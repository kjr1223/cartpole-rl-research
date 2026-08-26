"""
uam_fresh_continue.py
=====================
fresh_final.pth (ep1000)에서 이어서 학습 — 수렴 확인용

[목적]
  reward v2(시간 페널티) 구조가 baseline_no_time_penalty(47.6% landed)를
  따라잡는지, 아니면 낮은 수준에서 수렴하는지 확인

[판단 기준]
  - ep1000~1300: 버퍼 워밍업 노이즈 구간 → 판단 제외
  - ep1500부터: 착지율 추세로 수렴 여부 판단
  - 정지 조건 (셋 중 하나):
      1. 최근 2~3회 체크에서 착지율 변화 ±2%p 이내 → 수렴
      2. 착지율이 47.6% 근처 or 초과 → 학습량 부족이 원인
      3. 30% 근처에서 정체 → reward v2가 성능을 실제로 제한

[구조]
  - fresh_final.pth 로드 (가중치만, 버퍼는 새로 시작)
  - 500에피소드씩 체크포인트 저장
  - 매 500ep마다 termination breakdown 평가 (N=100)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

BASE      = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE, "uam_checkpoints_fresh")
LOG_FILE  = os.path.join(BASE, "uam_fresh_continue.log")
START_PTH = os.path.join(SAVE_DIR, "fresh_final.pth")

ALT_LOW    = 20.0
ALT_HIGH   = 40.0
MAX_STEPS  = 6_000
FIXED_ALPHA = 0.03
BUFFER_SIZE = 100_000

START_EP   = 1001   # fresh_final.pth가 ep1000까지 학습된 것
TOTAL_ADD  = 2000   # 최대 2000에피소드 추가 (ep3000까지)
EVAL_N_SCREEN = 100   # 중간 스크리닝용 (500ep마다)
EVAL_N_FINAL  = 500   # 정지 조건 걸렸을 때 재검증용 (수동 실행)
CHUNK         = 500   # 체크포인트/평가 주기

os.makedirs(SAVE_DIR, exist_ok=True)
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')

def eval_breakdown(agent, env, n=100):
    reasons = Counter()
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        reasons[info.get('termination_reason', 'unknown')] += 1
    return reasons

def main():
    env = VRSEnv()
    env.reset_alt_low  = ALT_LOW
    env.reset_alt_high = ALT_HIGH
    env.max_steps      = MAX_STEPS

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)
    ckpt  = torch.load(START_PTH, map_location=device)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic.load_state_dict(ckpt['critic'])

    log(f"[Device] {device}")
    log(f"[Load] {START_PTH}")
    log(f"[Note] 버퍼 새로 시작 — ep{START_EP}~ep{START_EP+299} 구간은 노이즈로 판단 제외")
    log(f"\n{'='*60}")
    log(f"reward v2 수렴 확인 학습 (ep{START_EP}~ep{START_EP+TOTAL_ADD-1})")
    log(f"  비교 기준: baseline_no_time_penalty landed=47.6%, timeout=52.4%")
    log(f"  평가: {CHUNK}ep마다 스크리닝 N={EVAL_N_SCREEN} / 정지 조건 시 재검증 N={EVAL_N_FINAL}")
    log(f"{'='*60}\n")

    landing_count = 0
    ep_rewards    = []

    for ep_offset in range(TOTAL_ADD):
        episode = START_EP + ep_offset

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

            if info.get('termination_reason') == 'landed':
                landed = True

            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)
        if landed:
            landing_count += 1

        if episode % 10 == 0:
            avg10  = np.mean(ep_rewards[-10:])
            land_r = landing_count / (ep_offset + 1) * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | 착지율: {land_r:.1f}%")

        # 500ep마다 체크포인트 + termination breakdown 평가
        if episode % CHUNK == 0:
            pth_path = os.path.join(SAVE_DIR, f"fresh_ep{episode}.pth")
            agent.save(pth_path)

            log(f"\n[Checkpoint] ep{episode} 저장 → {pth_path}")
            log(f"[Eval] 스크리닝 (N={EVAL_N_SCREEN}) ...")
            reasons = eval_breakdown(agent, env, n=EVAL_N_SCREEN)
            landed_pct  = 100 * reasons['landed']  / EVAL_N_SCREEN
            timeout_pct = 100 * reasons['timeout'] / EVAL_N_SCREEN
            log(f"  landed : {reasons['landed']:3d}개 ({landed_pct:.1f}%)")
            log(f"  timeout: {reasons['timeout']:3d}개 ({timeout_pct:.1f}%)")
            log(f"  crash  : {reasons['crash']:3d}개")
            log(f"  [baseline 비교] landed=47.6%, timeout=52.4%")
            log(f"  ※ 정지 조건 해당 시 N={EVAL_N_FINAL}으로 재검증 필요\n")

    agent.save(os.path.join(SAVE_DIR, "fresh_continue_final.pth"))
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep{START_EP}~ep{START_EP+TOTAL_ADD-1})")
    log(f"  총 착지: {landing_count}/{TOTAL_ADD} ({100*landing_count/TOTAL_ADD:.1f}%)")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == '__main__':
    main()
