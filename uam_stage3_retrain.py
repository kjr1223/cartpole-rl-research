"""
uam_stage3_retrain.py
=====================
Stage 3 (100~150m) 집중 재학습

[배경]
  1~2차 전체 학습에서 Stage 1/2는 수렴했으나 Stage 3은 미수렴(-2000 고착).
  원인: 100~150m에서 max_steps=8000 내 착지가 물리적으로 빡빡,
        버퍼(50k)가 8000스텝짜리 에피소드에 금방 오염됨.

[변경 사항]
  - ep400 체크포인트 로드 (Stage 2 수렴 직후 상태)
  - max_steps: 8000 → 12000 (30초 시뮬레이션, 착지 여유 확보)
  - buffer_size: 50k → 100k (이전 run 종료로 RAM 여유 회복)
  - episodes: 600 → 1200 (2배 더 학습)
  - 로그: uam_stage3_retrain.log (실시간)
  - 체크포인트: uam_checkpoints_stage3/ (100ep마다)
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

# ─────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
CKPT_LOAD    = os.path.join(BASE, "uam_checkpoints_v2", "uam_sac_ep400.pth")
SAVE_DIR     = os.path.join(BASE, "uam_checkpoints_stage3")
LOG_FILE     = os.path.join(BASE, "uam_stage3_retrain.log")
CURVE_PNG    = os.path.join(BASE, "uam_stage3_curves.png")

ALT_LOW      = 100.0
ALT_HIGH     = 150.0
MAX_STEPS    = 12_000   # 8000 → 12000: 착지 여유
EPISODES     = 1_200    # 600 → 1200
BUFFER_SIZE  = 100_000  # 50k → 100k
EVAL_EVERY   = 100
SAVE_EVERY   = 100
WARMUP       = 0        # 이미 학습된 체크포인트 사용 → warmup 불필요
FIXED_ALPHA  = 0.05
# ─────────────────────────────────────────

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

    # 에이전트 생성 (큰 버퍼)
    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    # ep400 체크포인트 로드
    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        # critic_target도 critic 가중치로 초기화 (체크포인트에 없으므로)
        agent.critic_target.load_state_dict(ckpt["critic"])
        agent.alpha = ckpt.get("alpha", FIXED_ALPHA)
        log(f"[Load] {CKPT_LOAD}")
    else:
        log(f"[경고] 체크포인트 없음: {CKPT_LOAD} → 랜덤 초기화로 시작")

    log(f"\n{'='*55}")
    log(f"Stage 3 집중 재학습 시작")
    log(f"  고도 범위: {ALT_LOW}~{ALT_HIGH}m")
    log(f"  max_steps: {MAX_STEPS}")
    log(f"  에피소드 수: {EPISODES}")
    log(f"  버퍼 크기: {BUFFER_SIZE:,}")
    log(f"  체크포인트: {SAVE_DIR}")
    log(f"{'='*55}\n")

    ep_rewards   = []
    eval_rewards = []
    total_steps  = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # warmup=0이므로 처음부터 정책 사용
            if total_steps < WARMUP:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, float(terminated))

            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            # 버퍼 충분하면 업데이트
            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)

        # 진행 로그 (10ep마다)
        if episode % 10 == 0:
            avg10 = np.mean(ep_rewards[-10:])
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | "
                f"Buffer: {len(agent.buffer):6d} | Steps: {total_steps:,}")

        # 평가
        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))
            log(f"\n[Eval] Episode {episode} | Eval Reward: {eval_r:.2f}\n")

        # 체크포인트 저장
        if episode % SAVE_EVERY == 0:
            path = os.path.join(SAVE_DIR, f"stage3_ep{episode}.pth")
            agent.save(path)

    # 최종 저장
    agent.save(os.path.join(SAVE_DIR, "stage3_final.pth"))

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Stage 3 (100~150m) Retraining Curves", fontsize=13)

    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w      = 20
    ma     = np.convolve(ep_arr, np.ones(w)/w, mode='valid')

    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='steelblue', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='steelblue', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if eval_rewards:
        eps_e, rews_e = zip(*eval_rewards)
        axes[1].plot(eps_e, rews_e, color='#2ecc71', marker='o', linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title('Eval Reward (Deterministic, 5ep)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"\n[Save] {CURVE_PNG}")

    log(f"\n{'='*55}")
    log(f"Stage 3 재학습 완료")
    log(f"  최종 Avg(20): {np.mean(ep_rewards[-20:]):.2f}")
    log(f"  최종 Eval: {eval_rewards[-1][1]:.2f}" if eval_rewards else "")
    log(f"{'='*55}")
    log_fh.close()


if __name__ == '__main__':
    main()
