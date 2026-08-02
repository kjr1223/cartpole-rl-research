"""
uam_stage3_v1_contextdb.py
==========================
Stage 3 (100~150m) — Context DB + Ontology 학습

[커리큘럼]
  Stage 1: 20~40m  → 9.5%  (완료)
  Stage 2: 60~90m  → 53.3% (완료, v4 best ep300 사용)
  Stage 3: 100~150m → 이번 학습

[설정]
  - 시작점: stage2_v4_best.pth (ep300, 착지율 61% 피크)
  - 고도: 100~150m, max_steps=16,000 (40초)
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
CKPT_LOAD   = os.path.join(BASE, "uam_checkpoints_stage2_v4", "stage2_v4_best.pth")
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_stage3_v1")
LOG_FILE    = os.path.join(BASE, "uam_stage3_v1_training.log")
CURVE_PNG   = os.path.join(BASE, "uam_stage3_v1_curves.png")

ALT_LOW     = 100.0
ALT_HIGH    = 150.0
MAX_STEPS   = 16_000   # 40초 (150m 하강 여유 충분히)
EPISODES    = 1_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
EVAL_EVERY  = 100
SAVE_EVERY  = 100

MARKING_THRESHOLD = 0.3
MARKING_BONUS     = 3.0

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

    markings = load_uam_markings()
    log(f"[Context DB] {len(markings)}개 마킹 로드 완료")

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic"])
        log(f"[Load] {CKPT_LOAD}")
    else:
        log("[경고] 체크포인트 없음 → 랜덤 초기화")

    log(f"\n{'='*60}")
    log(f"Stage 3 v1 — Context DB + Ontology 학습")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS} ({MAX_STEPS*0.0025:.1f}초)")
    log(f"  에피소드: {EPISODES} | buffer: {BUFFER_SIZE:,} | alpha: {FIXED_ALPHA}")
    log(f"  마킹 보너스: CAUTION 구역에서 거리<{MARKING_THRESHOLD} → +{MARKING_BONUS}")
    log(f"{'='*60}\n")

    ep_rewards    = []
    eval_rewards  = []
    landing_count = 0
    best_eval     = -float('inf')
    total_steps   = 0
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

            # CAUTION 구역에서 Context DB 보너스 적용
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
                agent.save(os.path.join(SAVE_DIR, "stage3_v1_best.pth"))
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} ★ 최고 → stage3_v1_best.pth\n")
            else:
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} (최고: {best_eval:.2f})\n")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"stage3_v1_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "stage3_v1_final.pth"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("UAM Stage3 v1 (Context DB + Ontology) — Curves", fontsize=13)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, len(ep_rewards) + 1)
    w = 20
    ma = np.convolve(ep_arr, np.ones(w)/w, mode='valid')
    axes[0].plot(x_ep, ep_arr, alpha=0.3, color='#E67E22', linewidth=0.7)
    axes[0].plot(x_ep[w-1:], ma, color='#E67E22', linewidth=1.8, label=f'MA({w})')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if eval_rewards:
        eps_e, rews_e = zip(*eval_rewards)
        axes[1].plot(eps_e, rews_e, color='#E67E22', marker='o', linewidth=1.5)
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
