"""
uam_stage1_full.py
==================
Stage 1 (20~40m) — Context DB + Ontology + PINN 전체 프레임워크
제안 방법의 모든 컴포넌트를 Stage 1 고도에서 적용

[구성]
  - 시작점: fresh_best.pth (Stage 1 SAC+VRS Reward+Ontology 베스트)
  - Context DB: CAUTION 구역에서 마킹 근처 → +3.0 보너스
  - Ontology: VRSEnv에 내장 (DANGER/RECOVER 추력 강제)
  - PINN: 실제 step 1회당 가상 transition 3개 추가 (Dyna 스타일)
  - 고도: 20~40m | max_steps: 6,000 | episodes: 1,000
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
from uam_sac import SACAgent, evaluate, device, load_uam_markings, get_uam_bonus

BASE        = os.path.dirname(os.path.abspath(__file__))
CKPT_LOAD   = os.path.join(BASE, "uam_checkpoints_fresh", "fresh_best.pth")
PINN_PATH   = os.path.join(BASE, "uam_pinn_model.pth")
SAVE_DIR    = os.path.join(BASE, "uam_checkpoints_stage1_full")
LOG_FILE    = os.path.join(BASE, "uam_stage1_full.log")
CURVE_PNG   = os.path.join(BASE, "uam_stage1_full_curves.png")

ALT_LOW     = 20.0
ALT_HIGH    = 40.0
MAX_STEPS   = 6_000
EPISODES    = 1_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
N_IMAGINED  = 3       # PINN 가상 transition 개수
EVAL_EVERY  = 100
SAVE_EVERY  = 200

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

    # Context DB 마킹 로드
    markings = load_uam_markings()
    log(f"[Context DB] {len(markings)}개 마킹 로드 완료")

    # PINN 로드
    pinn = UAMPINNTrainer()
    pinn.load(PINN_PATH)
    log(f"[PINN] {PINN_PATH} 로드 완료")

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    # fresh_best 체크포인트 로드
    if os.path.exists(CKPT_LOAD):
        ckpt = torch.load(CKPT_LOAD, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic"])
        log(f"[Load] {CKPT_LOAD}")
    else:
        log("[경고] 체크포인트 없음 → 랜덤 초기화")

    log(f"\n{'='*60}")
    log(f"Stage 1 Full Framework — Context DB + Ontology + PINN")
    log(f"  고도: {ALT_LOW}~{ALT_HIGH}m | max_steps: {MAX_STEPS} ({MAX_STEPS*0.0025:.1f}초)")
    log(f"  에피소드: {EPISODES} | n_imagined: {N_IMAGINED} | alpha: {FIXED_ALPHA}")
    log(f"  마킹 보너스: CAUTION 구역 거리<{MARKING_THRESHOLD} → +{MARKING_BONUS}")
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

            # Context DB 보너스 (CAUTION 구역에서만)
            zone_before = env._ontology(obs[1], obs[3])
            if zone_before == "CAUTION":
                bonus = get_uam_bonus(obs, markings, MARKING_THRESHOLD, MARKING_BONUS)
                if bonus > 0:
                    reward     += bonus
                    bonus_count += 1

            # 실제 transition 저장
            agent.buffer.push(obs, action, reward, next_obs, float(terminated))

            # PINN 가상 transition 추가 (Dyna 스타일)
            # DANGER/RECOVER에서는 env와 동일하게 dT=1.0 강제
            zone_obs = env._ontology(obs[1], obs[3])
            for _ in range(N_IMAGINED):
                a_img_raw = agent.select_action(obs)
                a_img = np.array([1.0], dtype=np.float32) if zone_obs in ("DANGER", "RECOVER") else a_img_raw
                ns_img = pinn.predict(obs, float(a_img[0]))
                ns_img = np.clip(ns_img, env.observation_space.low, env.observation_space.high)
                vrs_ratio_img = env._vrs_ratio(ns_img[1], ns_img[3])
                zone_img      = env._ontology(ns_img[1], ns_img[3])
                r_img, term_img = env._compute_reward(
                    ns_img[1], ns_img[2], ns_img[3], vrs_ratio_img, zone_img
                )
                agent.buffer.push(obs, a_img, r_img, ns_img, float(term_img))

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
                f"착지율: {land_r:.1f}% | 보너스: {bonus_count} | Steps: {total_steps:,}")

        if episode % EVAL_EVERY == 0:
            eval_r = evaluate(agent, env, n_episodes=5)
            eval_rewards.append((episode, eval_r))
            if eval_r > best_eval:
                best_eval = eval_r
                agent.save(os.path.join(SAVE_DIR, "stage1_full_best.pth"))
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} ★ → stage1_full_best.pth\n")
            else:
                log(f"\n[Eval] Episode {episode} | Eval: {eval_r:.2f} (최고: {best_eval:.2f})\n")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(SAVE_DIR, f"stage1_full_ep{episode}.pth"))

    agent.save(os.path.join(SAVE_DIR, "stage1_full_final.pth"))

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Stage 1 Full Framework (Context DB + Ontology + PINN)", fontsize=12)
    ep_arr = np.array(ep_rewards)
    x_ep   = np.arange(1, EPISODES + 1)
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
        axes[1].plot(eps_e, rews_e, color='#2ECC71', marker='o', linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title('Eval Reward (Deterministic)')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    log(f"\n{'='*60}")
    log(f"학습 완료")
    log(f"  최종 Avg(20) : {np.mean(ep_rewards[-20:]):.2f}")
    log(f"  최고 Eval    : {best_eval:.2f}")
    log(f"  착지율       : {100*landing_count/EPISODES:.1f}% ({landing_count}/{EPISODES})")
    log(f"  총 보너스    : {bonus_count}회")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == '__main__':
    main()
