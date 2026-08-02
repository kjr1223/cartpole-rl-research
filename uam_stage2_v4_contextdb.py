"""
uam_stage2_v4_contextdb.py
==========================
Stage 2 (60~90m) — Context DB + Ontology 적용 학습

[이전 버전과의 차이]
  v3: Context DB 없이 순수 SAC만 사용 → 2.0% 고원
  v4: Context DB + Ontology 추가
      - CAUTION 구역(VRS 비율 0.5~0.8)에 진입하면 마킹 데이터와 가까운 상태에 보너스
      - 온톨로지는 이미 env에 내장 (DANGER/RECOVER 구역에서 추력 강제 증가)

[설정]
  - 시작점: stage2_v2_best.pth (ep200, Eval +63.80)
  - 고도: 60~90m, max_steps=10,000 (25초)
  - Context DB: uam_switching_markings 테이블 (1,084개)
  - 마킹 보너스: CAUTION 구역에서 가까운 마킹 있으면 +3.0
  - alpha=0.03 (고정), buffer=100k, episodes=1,000
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, evaluate, device, load_uam_markings, get_uam_bonus

BASE        = os.path.dirname(os.path.abspath(__file__))
CKPT_LOAD   = os.path.join(BASE, "uam_checkpoints_stage2_v2", "stage2_v2_best.pth")
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_stage2_v4")
LOG_FILE    = os.path.join(BASE, "uam_stage2_v4_training.log")
CURVE_PNG   = os.path.join(BASE, "uam_stage2_v4_curves.png")

ALT_LOW     = 60.0
ALT_HIGH    = 90.0
MAX_STEPS   = 10_000   # 25초
EPISODES    = 1_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
EVAL_EVERY  = 100
SAVE_EVERY  = 100

# Context DB 파라미터
MARKING_THRESHOLD = 0.3   # 정규화 거리 임계값 (이 거리 안에 마킹이 있으면 보너스)
MARKING_BONUS     = 3.0   # CAUTION 구역 마킹 근처일 때 추가 보상

os.makedirs(SAVE_DIR, exist_ok=True)
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')

def main():
    env = VRSEnv()
    env.reset_alt_low  = ALT_LOW
    env.reset_alt_high = ALT_HIGH
    env.max_steps      = MAX_STEPS

    # Context DB 마킹 데이터 로드 (V_x, V_z, alt, T_norm — 4개 변수)
    # uam_sac.py의 load_uam_markings()는 uam_switching_markings 테이블에서 읽는다
    markings = load_uam_markings()
    log(f"[Context DB] {len(markings)}개 마킹 로드 완료")

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    # v2 best 체크포인트에서 이어서 학습
    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic"])
        log(f"[Load] {CKPT_LOAD}")
    else:
        log("[경고] 체크포인트 없음 → 랜덤 초기화")

    log(f"\n{'='*60}")
    log(f"Stage 2 v4 — Context DB + Ontology 학습")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS} ({MAX_STEPS*0.0025:.1f}초)")
    log(f"  에피소드: {EPISODES} | buffer: {BUFFER_SIZE:,} | alpha: {FIXED_ALPHA}")
    log(f"  마킹 보너스: CAUTION 구역에서 거리<{MARKING_THRESHOLD} → +{MARKING_BONUS}")
    log(f"{'='*60}\n")

    ep_rewards    = []
    eval_rewards  = []
    landing_count = 0
    best_eval     = -float('inf')
    total_steps   = 0

    # 구역별 보너스 발생 횟수 기록 (얼마나 자주 마킹 근처에 갔는지 확인용)
    bonus_count   = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done      = False
        landed    = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ─────────────────────────────────────────────
            # Context DB 보너스: CAUTION 구역에서만 적용
            # DANGER/RECOVER는 env 온톨로지가 이미 추력을 강제로 올려서
            # 별도 보너스 없이도 회복 행동이 보장된다.
            # SAFE는 VRS와 무관하므로 보너스 불필요.
            # CAUTION(VRS 비율 0.5~0.8)만 학습 신호가 약하므로 여기서 보강.
            # ─────────────────────────────────────────────
            zone_before = env._ontology(obs[1], obs[3])
            if zone_before == "CAUTION":
                bonus = get_uam_bonus(obs, markings, MARKING_THRESHOLD, MARKING_BONUS)
                if bonus > 0:
                    reward  += bonus
                    bonus_count += 1

            agent.buffer.push(obs, action, reward, next_obs, float(terminated))
            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            # 착지 성공 여부 확인 (V_z ≤ 2.0, alt ≤ 1.0)
            if terminated and next_obs[1] <= 2.0 and next_obs[2] <= 1.0:
                landed = True

            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)
        if landed:
            landing_count += 1

        if episode % 10 == 0:
            avg10  = np.mean(ep_rewards[-10:])
            land_r = landing_count / episode * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | "
                f"착지율: {land_r:.1f}% | 보너스횟수: {bonus_count} | Steps: {total_steps:,}")

        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))

            if eval_r > best_eval:
                best_eval = eval_r
                agent.save(os.path.join(SAVE_DIR, "stage2_v4_best.pth"))
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} ★ 최고 → stage2_v4_best.pth\n")
            else:
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} (최고: {best_eval:.2f})\n")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"stage2_v4_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "stage2_v4_final.pth"))

    # 학습 곡선 저장
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Stage2 v4 (Context DB + Ontology) — Curves", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='#3182CE', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='#3182CE', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if eval_rewards:
        eps_e, rews_e = zip(*eval_rewards)
        axes[1].plot(eps_e, rews_e, color='#38A169', marker='o', linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title('Eval Reward (Deterministic)')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"[Save] {CURVE_PNG}")

    log(f"\n{'='*60}")
    log(f"학습 완료")
    log(f"  최종 Avg(20)  : {np.mean(ep_rewards[-20:]):.2f}")
    log(f"  최고 Eval     : {best_eval:.2f}")
    log(f"  총 착지 횟수  : {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    log(f"  총 보너스 발생: {bonus_count}회")
    log(f"{'='*60}")
    log_fh.close()


if __name__ == '__main__':
    main()
