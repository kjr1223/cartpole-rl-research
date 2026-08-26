"""
uam_method0b_train.py
=====================
방법론 0-b: 시간 패널티 검증용
  use_ontology_override = False  (온톨로지 개입 없음)
  use_vrs_penalty       = False  (VRS 패널티 없음)
  use_time_penalty      = True   (시간 패널티 있음: -0.01/step)

[목적]
  "VRS 패널티 없이 시간 패널티만 있으면 착지율에 영향이 있는가?"
  방법론0-a(시간패널티 없음)와 비교 → 시간 패널티 단독 효과 격리.
  0-a ≈ 0-b이면 시간 패널티는 무해 → 방법론1~3에 포함 가능.
  0-b << 0-a이면 시간 패널티 자체가 문제 → 재설계 필요.

[비교 기준]
  baseline_no_time_penalty.pth: landed=47.6%, timeout=52.4%
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "uam_checkpoints_method0")
LOG_FILE = os.path.join(BASE, "uam_method0b.log")

ALT_LOW      = 20.0
ALT_HIGH     = 40.0
MAX_STEPS    = 6_000
EPISODES     = 3_000
BUFFER_SIZE  = 100_000
FIXED_ALPHA  = 0.03
CHUNK        = 500
EVAL_N       = 100

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
    env.use_ontology_override = False  # 온톨로지 개입 없음
    env.use_vrs_penalty       = False  # VRS 패널티 없음
    env.use_time_penalty      = True   # 시간 패널티 있음 (-0.01/step)

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    log(f"[Device] {device}")
    log(f"[Method] 0-b | ontology=OFF | vrs_penalty=OFF | time_penalty=ON(-0.01/step)")
    log(f"\n{'='*60}")
    log(f"방법론 0-b 학습 (ep1~{EPISODES})")
    log(f"  보상: 하강(+alt_drop) | 시간패널티(-0.01/step) | 착지+500")
    log(f"  비교 기준: baseline_no_time_penalty landed=47.6%")
    log(f"             방법론0-a (동시 학습 중)")
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
            pth = os.path.join(SAVE_DIR, f"method0b_ep{episode}.pth")
            agent.save(pth)
            log(f"\n[Checkpoint] ep{episode} → {pth}")
            log(f"[Eval] N={EVAL_N} ...")
            reasons = eval_breakdown(agent, env, n=EVAL_N)
            log(f"  landed : {reasons['landed']:3d}개 ({100*reasons['landed']/EVAL_N:.1f}%)")
            log(f"  timeout: {reasons['timeout']:3d}개 ({100*reasons['timeout']/EVAL_N:.1f}%)")
            log(f"  crash  : {reasons.get('crash', 0):3d}개")
            log(f"  [비교] baseline landed=47.6%\n")

    agent.save(os.path.join(SAVE_DIR, "method0b_final.pth"))
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep1~{EPISODES})")
    log(f"  총 착지: {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == "__main__":
    main()
