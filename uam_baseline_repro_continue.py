"""
uam_baseline_repro_continue.py
===============================
baseline_repro ep3000 체크포인트에서 이어받아 ep8000까지 학습

[목적]
  "3000ep로 재현 실패(7.2%)"가 학습량 부족인지, 구조적 불안정인지 검증
  원본 baseline(47.6%)이 더 오래 학습된 결과였다면 연장 시 수렴해야 함

[조건] — baseline_repro와 동일
  use_ontology_override = True
  use_vrs_penalty       = True
  use_time_penalty      = False

[정지 규칙]
  - ep4000: 워밍업 구간(버퍼 재충전) 포함 → 참고용만, 판단 보류
  - ep5000/6000/7000/8000: 착지율 40%+ 도달 시 즉시 중단 → 학습량 문제 확정
  - ep8000까지 진동/붕괴 반복 → 구조적 불안정, 중단 후 reward 구조 재검토

[주의]
  체크포인트 로드 = 네트워크 가중치만 복원, 리플레이 버퍼는 새로 채움
  → ep3000→ep4000 구간은 버퍼 워밍업으로 착지율 일시 불안정 가능
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

BASE       = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE, "uam_checkpoints_baseline_repro")   # 기존 폴더에 추가
LOG_FILE   = os.path.join(BASE, "uam_baseline_repro_continue.log")
LOAD_CKPT  = os.path.join(SAVE_DIR, "baseline_repro_ep4000.pth")

ALT_LOW     = 20.0
ALT_HIGH    = 40.0
MAX_STEPS   = 6_000
EP_START    = 4_001
EP_END      = 8_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
EVAL_N      = 100
STOP_THRESHOLD = 40.0   # eval 착지율 이 이상이면 조기 중단

EVAL_CHUNKS  = {5000, 6000, 7000, 8000}   # 판단 기준 체크포인트
WATCH_CHUNKS = {4000} | EVAL_CHUNKS        # 저장+eval 수행 시점

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
    env.reset_alt_low        = ALT_LOW
    env.reset_alt_high       = ALT_HIGH
    env.max_steps            = MAX_STEPS
    env.use_ontology_override = True
    env.use_vrs_penalty       = True
    env.use_time_penalty      = False

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)
    agent.load(LOAD_CKPT)

    log(f"[Device] {device}")
    log(f"[Method] baseline_repro_continue | ep{EP_START}~{EP_END} | 로드: {LOAD_CKPT}")
    log(f"\n{'='*60}")
    log(f"baseline 재현 연장 학습")
    log(f"  조건: ontology=ON | vrs_penalty=ON | time_penalty=OFF")
    log(f"  워밍업: ep4000 참고용만 (버퍼 재충전 구간)")
    log(f"  판단 기준: ep5000~8000, 착지율 {STOP_THRESHOLD}%+ 도달 시 조기 중단")
    log(f"  비교 기준: baseline_no_time_penalty 47.6%")
    log(f"{'='*60}\n")

    landing_count = 0
    ep_rewards    = []
    total_eps     = EP_END - EP_START + 1

    for episode in range(EP_START, EP_END + 1):
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
            elapsed   = episode - EP_START + 1
            avg10     = np.mean(ep_rewards[-10:])
            land_r    = landing_count / elapsed * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | 착지율: {land_r:.1f}%")

        if episode % 500 == 0 and episode in WATCH_CHUNKS:
            pth = os.path.join(SAVE_DIR, f"baseline_repro_ep{episode}.pth")
            agent.save(pth)
            log(f"\n[Checkpoint] ep{episode} → {pth}")

            label = "참고용 (워밍업 구간)" if episode == 4000 else "판단 기준"
            log(f"[Eval] N={EVAL_N} ... [{label}]")
            reasons = eval_breakdown(agent, env, n=EVAL_N)
            landed_pct = 100 * reasons['landed'] / EVAL_N
            log(f"  landed : {reasons['landed']:3d}개 ({landed_pct:.1f}%)")
            log(f"  timeout: {reasons['timeout']:3d}개 ({100*reasons['timeout']/EVAL_N:.1f}%)")
            log(f"  crash  : {reasons.get('crash',0):3d}개")
            log(f"  [비교] baseline 47.6%\n")

            if episode in EVAL_CHUNKS and landed_pct >= STOP_THRESHOLD:
                log(f"{'='*60}")
                log(f"[조기 중단] ep{episode} eval {landed_pct:.1f}% ≥ {STOP_THRESHOLD}%")
                log(f"  → 학습량 부족이 원인이었음 확정")
                log(f"{'='*60}")
                log_fh.close()
                return

    elapsed = EP_END - EP_START + 1
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep{EP_START}~{EP_END})")
    log(f"  총 착지: {landing_count}/{elapsed} ({100*landing_count/elapsed:.1f}%)")
    log(f"  → ep8000까지 수렴 실패 시 구조적 불안정으로 결론")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == "__main__":
    main()
