"""
uam_pinn_train.py
=================
use_pinn=True: PINN 가상경험(Dyna 스타일) 추가 학습

[배경]
  stage3_ep1100.pth (Eval +1159) 을 시작점으로,
  PINN 가상 transition을 버퍼에 추가해 데이터 효율을 높인다.
  실제 env.step() 1회마다 PINN으로 n_imagined=3개 가상 경험을 추가 생성.

[변경 사항 vs 이전 run]
  - use_pinn=True (핵심)
  - 시작점: stage3_ep1100.pth
  - Stage 3 전용 (100~150m, max_steps=12000)
  - buffer_size=100k, alpha=0.03 (0.05에서 약간 낮춤 — 붕괴 방지)
  - episodes=1000
  - 체크포인트: uam_checkpoints_pinn/
  - 로그: uam_pinn_training.log
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uam_vrs_env import VRSEnv
from uam_pinn_model import UAMPINNTrainer
from uam_sac import SACAgent, evaluate, get_uam_bonus, device

BASE        = os.path.dirname(os.path.abspath(__file__))
CKPT_LOAD   = os.path.join(BASE, "uam_checkpoints_stage3", "stage3_ep1100.pth")
PINN_PATH   = os.path.join(BASE, "uam_pinn_model.pth")
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_pinn")
LOG_FILE    = os.path.join(BASE, "uam_pinn_training.log")
CURVE_PNG   = os.path.join(BASE, "uam_pinn_curves.png")

ALT_LOW     = 100.0
ALT_HIGH    = 150.0
MAX_STEPS   = 12_000
EPISODES    = 1_000
BUFFER_SIZE = 100_000
N_IMAGINED  = 3        # PINN 가상 transition 개수 (실제 1회당)
FIXED_ALPHA = 0.03     # 0.05에서 낮춤 (붕괴 방지)
EVAL_EVERY  = 100
SAVE_EVERY  = 100

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

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    # ep1100 체크포인트 로드
    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic"])
        log(f"[Load] {CKPT_LOAD}")
    else:
        log(f"[경고] 체크포인트 없음 → 랜덤 초기화")

    # PINN 로드
    pinn = UAMPINNTrainer()
    pinn.load(PINN_PATH)
    log(f"[PINN] {PINN_PATH} 로드 완료")

    log(f"\n{'='*55}")
    log(f"PINN 가상경험 학습 시작 (use_pinn=True)")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS}")
    log(f"  에피소드: {EPISODES} | buffer: {BUFFER_SIZE:,}")
    log(f"  n_imagined: {N_IMAGINED} | alpha: {FIXED_ALPHA}")
    log(f"{'='*55}\n")

    ep_rewards   = []
    eval_rewards = []
    total_steps  = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 실제 경험 저장
            agent.buffer.push(obs, action, reward, next_obs, float(terminated))

            # ── PINN 가상 transition 추가 (Dyna 스타일) ──────────────
            # DANGER/RECOVER 구역에서는 env.step()이 dT=1.0으로 강제하므로
            # PINN에도 동일한 override를 적용해야 분포가 맞음
            zone_obs = env._ontology(obs[1], obs[3])
            for _ in range(N_IMAGINED):
                a_img_raw = agent.select_action(obs)
                if zone_obs in ("DANGER", "RECOVER"):
                    a_img = np.array([1.0], dtype=np.float32)
                else:
                    a_img = a_img_raw

                ns_img = pinn.predict(obs, float(a_img[0]))
                ns_img = np.clip(ns_img,
                                 env.observation_space.low,
                                 env.observation_space.high)

                vrs_ratio_img = env._vrs_ratio(ns_img[1], ns_img[3])
                zone_img      = env._ontology(ns_img[1], ns_img[3])
                r_img, term_img = env._compute_reward(
                    ns_img[1], ns_img[2], ns_img[3], vrs_ratio_img, zone_img
                )
                agent.buffer.push(obs, a_img, r_img, ns_img, float(term_img))
            # ─────────────────────────────────────────────────────────

            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)

        if episode % 10 == 0:
            avg10 = np.mean(ep_rewards[-10:])
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | "
                f"Buffer: {len(agent.buffer):6d} | Steps: {total_steps:,}")

        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))
            log(f"\n[Eval] Episode {episode} | Eval Reward: {eval_r:.2f}\n")

        if episode % SAVE_EVERY == 0:
            path = os.path.join(SAVE_DIR, f"pinn_ep{episode}.pth")
            agent.save(path)

    agent.save(os.path.join(SAVE_DIR, "pinn_final.pth"))

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Stage3 + PINN Training Curves", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='purple', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='purple', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward (PINN)')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if eval_rewards:
        eps_e, rews_e = zip(*eval_rewards)
        axes[1].plot(eps_e, rews_e, color='#2ecc71', marker='o', linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title('Eval Reward (Deterministic)')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"[Save] {CURVE_PNG}")

    log(f"\n{'='*55}")
    log(f"PINN 학습 완료")
    log(f"  최종 Avg(20): {np.mean(ep_rewards[-20:]):.2f}")
    if eval_rewards:
        log(f"  최종 Eval   : {eval_rewards[-1][1]:.2f}")
    log(f"{'='*55}")
    log_fh.close()


if __name__ == '__main__':
    main()
