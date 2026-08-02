"""
uam_fresh_train.py
==================
처음부터 다시 시작하는 순수 SAC 학습

[교수님 피드백 반영]
  - 기존 체크포인트 없음 → 랜덤 초기화 (편향 제거)
  - PINN 없음, Reward Shaping 없음 → 순수 SAC
  - 보상 함수 완전 재설계:
      생존 보상 제거 / 하강 진행 보상 / VRS 이차 패널티 / 착지 보너스

[새 보상 함수 원칙]
  생존(버티기)에 아무 이득 없음
  내려갈수록 이득 (+alt_drop × 1.0)
  VRS 깊이 들어갈수록 급격히 손해 (-(ratio-0.5)² × 30)
  착지 = 가장 큰 보상 (+500)

[설정]
  고도: 20~40m (Stage 1 — 물리적으로 착지 가능한 범위)
  max_steps: 6,000
  alpha: 0.03, buffer: 100k, episodes: 1,000
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, evaluate, device

BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_fresh")
LOG_FILE    = os.path.join(BASE, "uam_fresh_training.log")
CURVE_PNG   = os.path.join(BASE, "uam_fresh_curves.png")

ALT_LOW     = 20.0
ALT_HIGH    = 40.0
MAX_STEPS   = 6_000
EPISODES    = 1_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
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

    # 랜덤 초기화 — 체크포인트 없음
    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    log(f"[Device] {device}")
    log(f"[Init] 랜덤 초기화 (기존 체크포인트 미사용)")
    log(f"\n{'='*55}")
    log(f"순수 SAC 재학습 (처음부터)")
    log(f"  보상: 생존보상 없음 | 하강(+alt_drop) | VRS 이차패널티 | 착지+500")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS} ({MAX_STEPS*0.0025:.1f}초)")
    log(f"  에피소드: {EPISODES} | buffer: {BUFFER_SIZE:,} | alpha: {FIXED_ALPHA}")
    log(f"{'='*55}\n")

    ep_rewards    = []
    eval_rewards  = []
    landing_count = 0
    best_eval     = -float('inf')
    total_steps   = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done      = False
        landed    = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, float(terminated))
            obs        = next_obs
            ep_reward += reward
            total_steps += 1

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
                f"착지율: {land_r:.1f}% | Steps: {total_steps:,}")

        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))

            if eval_r > best_eval:
                best_eval = eval_r
                agent.save(os.path.join(SAVE_DIR, "fresh_best.pth"))
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} ★ 최고 → fresh_best.pth\n")
            else:
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} (최고: {best_eval:.2f})\n")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"fresh_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "fresh_final.pth"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Fresh SAC — Curves", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='#6B46C1', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='#6B46C1', linewidth=1.8, label=f'MA({w})')
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

    log(f"\n{'='*55}")
    log(f"학습 완료")
    log(f"  최종 Avg(20)  : {np.mean(ep_rewards[-20:]):.2f}")
    log(f"  최고 Eval     : {best_eval:.2f}")
    log(f"  총 착지 횟수  : {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    log(f"{'='*55}")
    log_fh.close()


if __name__ == '__main__':
    main()
