"""
uam_baseline_repro_train.py
===========================
baseline_no_time_penalty.pth 조건 재현 학습 — 로그 있는 버전

[조건]
  use_ontology_override = True   (DANGER/RECOVER → dT=1.0 강제)
  use_vrs_penalty       = True   (VRS 이차패널티 ON)
  use_time_penalty      = False  (시간패널티 OFF)

[목적]
  baseline(47.6%)이 "ontology ON 조건이 안정적으로 수렴한 결과"인지
  vs "우연히 좋은 지점에서 멈춘 케이스"인지 구분.
  이번엔 학습 과정 전체 로그 보존.

[보상 구조 — 현재 최종 버전]
  하강    : +alt_drop × 1.0
  VRS패널티: -(ratio-0.5)² × 30  (ratio ≥ 0.5)
  착지    : +500 (Vz≤2.0)
  Tier 1  : +100 (Vz≤1.5)
  Tier 2  : +100 (Vz≤1.0)
  Tier 3  : +100 (Vz≤0.5)
  clip    : [-200, 900]

[비교 기준]
  baseline_no_time_penalty.pth: 47.6% (로그 없음, 학습 조건 불명)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "uam_checkpoints_baseline_repro")
LOG_FILE = os.path.join(BASE, "uam_baseline_repro.log")

ALT_LOW     = 20.0
ALT_HIGH    = 40.0
MAX_STEPS   = 6_000
EPISODES    = 3_000
BUFFER_SIZE = 100_000
FIXED_ALPHA = 0.03
CHUNK       = 500
EVAL_N      = 100

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
    env.reset_alt_low       = ALT_LOW
    env.reset_alt_high      = ALT_HIGH
    env.max_steps           = MAX_STEPS
    env.use_ontology_override = True   # baseline 조건
    env.use_vrs_penalty       = True   # baseline 조건
    env.use_time_penalty      = False  # baseline 조건

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    log(f"[Device] {device}")
    log(f"[Method] baseline_repro | ontology=ON | vrs_penalty=ON | time_penalty=OFF")
    log(f"\n{'='*60}")
    log(f"baseline 조건 재현 학습 (ep1~{EPISODES}, method1과 동일 규모)")
    log(f"  보상: 하강(+alt_drop) | VRS패널티(×30) | 착지+500 | Tier+100×3")
    log(f"  clip: [-200, 900]")
    log(f"  비교 기준: baseline_no_time_penalty 47.6%")
    log(f"  평가: {CHUNK}ep마다 N={EVAL_N} termination breakdown")
    log(f"{'='*60}\n")

    landing_count = 0
    ep_rewards    = []

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

            if info.get('termination_reason') == 'landed':
                landed = True

            if len(agent.buffer) >= 256:
                agent.update()

        ep_rewards.append(ep_reward)
        if landed:
            landing_count += 1

        if episode % 10 == 0:
            avg10  = np.mean(ep_rewards[-10:])
            land_r = landing_count / episode * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | 착지율: {land_r:.1f}%")

        if episode % CHUNK == 0:
            pth = os.path.join(SAVE_DIR, f"baseline_repro_ep{episode}.pth")
            agent.save(pth)
            log(f"\n[Checkpoint] ep{episode} → {pth}")
            log(f"[Eval] N={EVAL_N} ...")
            reasons = eval_breakdown(agent, env, n=EVAL_N)
            log(f"  landed : {reasons['landed']:3d}개 ({100*reasons['landed']/EVAL_N:.1f}%)")
            log(f"  timeout: {reasons['timeout']:3d}개 ({100*reasons['timeout']/EVAL_N:.1f}%)")
            log(f"  crash  : {reasons.get('crash', 0):3d}개")
            log(f"  [비교] baseline 47.6%\n")

    agent.save(os.path.join(SAVE_DIR, "baseline_repro_final.pth"))
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep1~{EPISODES})")
    log(f"  총 착지: {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == "__main__":
    main()
