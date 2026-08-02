"""
uam_shaped_train.py
===================
보상 함수 개선 후 Stage 3 재학습

[변경된 보상 함수]
  - 하강 shaping: +0.05 × max(alt_prev - alt, 0)  → "내려갈수록 이득"
  - 착지 보너스: +100 → +500
  - reward 클리핑 상한: +200 → +600

[설정]
  - 시작점: stage3_ep1100.pth
  - Stage 3 (100~150m), max_steps=12000
  - alpha=0.03, buffer=100k, episodes=1000
  - 체크포인트: uam_checkpoints_shaped/
  - 로그: uam_shaped_training.log
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
CKPT_LOAD   = os.path.join(BASE, "uam_checkpoints_stage3", "stage3_ep1100.pth")
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_shaped")
LOG_FILE    = os.path.join(BASE, "uam_shaped_training.log")
CURVE_PNG   = os.path.join(BASE, "uam_shaped_curves.png")

ALT_LOW     = 100.0
ALT_HIGH    = 150.0
MAX_STEPS   = 12_000
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

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic"])
        log(f"[Load] {CKPT_LOAD}")
    else:
        log("[경고] 체크포인트 없음 → 랜덤 초기화")

    log(f"\n{'='*55}")
    log(f"보상 함수 개선 후 Stage 3 재학습")
    log(f"  shaping: +0.05 × alt_drop/step")
    log(f"  착지 보너스: +500 (기존 +100)")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS}")
    log(f"  에피소드: {EPISODES} | buffer: {BUFFER_SIZE:,} | alpha: {FIXED_ALPHA}")
    log(f"{'='*55}\n")

    ep_rewards   = []
    eval_rewards = []
    landing_count = 0
    total_steps  = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        landed = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, float(terminated))
            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            # 안전 착지 감지 (V_z≤2 & alt≤1 → terminated)
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
                f"착지율: {land_r:.1f}% | Buffer: {len(agent.buffer):6d}")

        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))
            log(f"\n[Eval] Episode {episode} | Eval Reward: {eval_r:.2f}\n")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"shaped_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "shaped_final.pth"))

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Stage3 + Reward Shaping Curves", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='#e67e22', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='#e67e22', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward (Shaped)')
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
    log(f"학습 완료")
    log(f"  최종 Avg(20)  : {np.mean(ep_rewards[-20:]):.2f}")
    log(f"  총 착지 횟수  : {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    if eval_rewards:
        log(f"  최종 Eval     : {eval_rewards[-1][1]:.2f}")
    log(f"{'='*55}")
    log_fh.close()


if __name__ == '__main__':
    main()
