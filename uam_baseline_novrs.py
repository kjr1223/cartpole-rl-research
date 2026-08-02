"""
uam_baseline_novrs.py
=====================
베이스라인: VRS를 전혀 고려하지 않은 순수 SAC

[제안 방법과의 차이]
  - VRS 패널티 없음 (이차 함수 패널티 제거)
  - 온톨로지 개입 없음 (DANGER/RECOVER에서 추력 강제 안 함)
  - Context DB 없음
  - 오직 "착지만 목표로" 학습

[측정 지표]
  - VRS 진입률: 전체 스텝 중 VRS 구역(vrs_ratio ≥ 0.5)에 있던 비율
  - 착지율: 1,000 에피소드 중 착지 성공 횟수

[목적]
  VRS를 모르는 에이전트가 착지를 추구할 때
  VRS에 얼마나 자주 빠지는지 측정 → 제안 방법과 비교
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "uam_checkpoints_baseline")
LOG_FILE = os.path.join(BASE, "uam_baseline_novrs.log")
CURVE_PNG= os.path.join(BASE, "uam_baseline_novrs_curves.png")

ALT_LOW     = 20.0
ALT_HIGH    = 40.0
MAX_STEPS   = 6_000    # Stage 1과 동일
EPISODES    = 1_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
EVAL_EVERY  = 100
SAVE_EVERY  = 200

os.makedirs(SAVE_DIR, exist_ok=True)
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')


class NoVRSEnv(VRSEnv):
    """
    VRS를 전혀 고려하지 않은 환경.
    - 온톨로지 개입 제거: DANGER/RECOVER에서도 에이전트 행동 그대로 사용
    - VRS 패널티 제거: 보상에서 VRS 관련 항목 없음
    - 착지 보상과 하강 진행 보상만 유지
    """

    def step(self, action):
        # 부모 step() 호출 전에 온톨로지 개입을 우회하기 위해
        # _ontology 결과를 무시하고 항상 에이전트 행동 그대로 전달
        from scipy.integrate import solve_ivp

        V_x, V_z, alt, T_norm = self.state
        dT = float(np.clip(action, -1.0, 1.0).flat[0])

        # 온톨로지 개입 없음 — dT 그대로 사용
        result = solve_ivp(
            fun=lambda t, s: self._derivatives(s, dT),
            t_span=(0.0, self.dt),
            y0=[V_x, V_z, alt, T_norm],
            method='RK45',
        )
        V_x_new, V_z_new, alt_new, T_norm_new = result.y[:, -1]

        V_x_new    = float(np.clip(V_x_new,    -20.0, 20.0))
        V_z_new    = float(np.clip(V_z_new,    -10.0, 10.0))
        alt_new    = float(np.clip(alt_new,      0.0, 200.0))
        T_norm_new = float(np.clip(T_norm_new,   0.0,   1.0))

        self.state = np.array([V_x_new, V_z_new, alt_new, T_norm_new], dtype=np.float32)
        self.state = self._safe_obs(self.state)
        self.step_count += 1

        v_i_new, v_h_new = self._compute_vi(V_z_new, T_norm_new)
        state_ok = bool(np.all(np.isfinite(self.state)))

        if not (state_ok and np.isfinite(v_i_new) and np.isfinite(v_h_new)):
            self.state = np.zeros(4, dtype=np.float32)
            return (self.state, -100.0, True, False, {"anomaly": True})

        vrs_ratio = V_z_new / v_h_new if v_h_new > 0.0 else 999.0
        zone_new  = self._ontology(V_z_new, T_norm_new)

        # VRS 패널티 없는 보상 계산
        reward, terminated = self._compute_reward_novrs(
            V_z_new, alt_new, T_norm_new, vrs_ratio, zone_new, alt_prev=alt
        )

        truncated = self.step_count >= self.max_steps

        self.history_alt.append(alt_new)
        self.history_vrs_ratio.append(vrs_ratio)
        self.history_ontology.append(zone_new)

        info = {
            "vrs_ratio": vrs_ratio,
            "zone"     : zone_new,
            "step"     : self.step_count,
            "in_vrs"   : vrs_ratio >= 0.5,  # VRS 진입 여부
        }

        return self.state, reward, terminated, truncated, info

    def _compute_reward_novrs(self, V_z, alt, T_norm, vrs_ratio, zone, alt_prev=None):
        """
        VRS 패널티 없는 보상 함수.
        착지 진행 보상 + 착지 성공/실패만 있음.
        """
        reward     = 0.0
        terminated = False

        # 1. 하강 진행 보상 (VRS 고려 없이 내려가면 보상)
        if alt_prev is not None:
            alt_drop = alt_prev - alt
            reward  += alt_drop * 1.0

        # 2. VRS 패널티 없음 ← 핵심 차이

        # 3. 착지 성공
        if alt <= 1.0 and V_z <= 2.0:
            reward    += 500.0
            terminated = True

        # 4. 추락
        if alt <= 0.0 and V_z > self.V_z_crash:
            reward    -= 100.0
            terminated = True

        # 5. 고도 초과
        if alt >= self.alt_max:
            reward    -= 50.0
            terminated = True

        if not np.isfinite(reward):
            reward = -100.0

        reward = float(np.clip(reward, -200.0, 600.0))
        return reward, terminated


def main():
    env = NoVRSEnv()
    env.reset_alt_low  = ALT_LOW
    env.reset_alt_high = ALT_HIGH
    env.max_steps      = MAX_STEPS

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    log(f"\n{'='*60}")
    log(f"베이스라인: VRS 고려 없는 순수 SAC")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS} ({MAX_STEPS*0.0025:.1f}초)")
    log(f"  에피소드: {EPISODES} | VRS 패널티: 없음 | 온톨로지 개입: 없음")
    log(f"{'='*60}\n")

    ep_rewards     = []
    landing_count  = 0
    total_steps    = 0
    total_vrs_steps= 0   # 전체 스텝 중 VRS 구역에 있던 스텝 수
    best_eval      = -float('inf')

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward  = 0.0
        done       = False
        landed     = False
        ep_vrs     = 0   # 이 에피소드에서 VRS 진입 스텝 수
        ep_steps   = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # VRS 진입 여부 카운트
            if info.get("in_vrs", False):
                ep_vrs      += 1
                total_vrs_steps += 1

            agent.buffer.push(obs, action, reward, next_obs, float(terminated))
            obs        = next_obs
            ep_reward += reward
            total_steps += 1
            ep_steps   += 1

            if terminated and next_obs[1] <= 2.0 and next_obs[2] <= 1.0:
                landed = True

            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)
        if landed:
            landing_count += 1

        if episode % 10 == 0:
            avg10      = np.mean(ep_rewards[-10:])
            land_r     = landing_count / episode * 100
            vrs_entry  = total_vrs_steps / total_steps * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | "
                f"착지율: {land_r:.1f}% | VRS진입률: {vrs_entry:.1f}% | Steps: {total_steps:,}")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"baseline_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "baseline_final.pth"))

    final_vrs_rate  = total_vrs_steps / total_steps * 100
    final_land_rate = landing_count / EPISODES * 100

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Baseline (No VRS) — Stage 1", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, EPISODES + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='#E74C3C', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='#E74C3C', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title('VRS Entry Rate & Landing Rate')
    axes[1].bar(['VRS 진입률', '착지율'], [final_vrs_rate, final_land_rate],
                color=['#E74C3C', '#3498DB'])
    axes[1].set_ylabel('%')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate([final_vrs_rate, final_land_rate]):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    log(f"\n{'='*60}")
    log(f"베이스라인 학습 완료")
    log(f"  착지율       : {final_land_rate:.1f}% ({landing_count}/{EPISODES})")
    log(f"  VRS 진입률   : {final_vrs_rate:.1f}% ({total_vrs_steps:,}/{total_steps:,} 스텝)")
    log(f"{'='*60}")
    log_fh.close()


if __name__ == '__main__':
    main()
