"""
uam_method1_v3_continue.py
===========================
method1-v3 ep1500 체크포인트에서 ep3000까지 이어받기

[이어받기 이유]
  ep1870에서 OOM으로 프로세스 종료 — ep1500 체크포인트가 마지막 저장 지점

[조건] — method1-v3(uam_method1_train.py)와 동일
  use_ontology_override = False
  use_vrs_penalty       = True
  use_time_penalty      = False
  마킹 보너스           = 에피소드당 1회 (bonus_given 플래그)

[비교 기준]
  baseline_repro_continue (ep4001~8000) 결과와 병렬 비교
  핵심 지표: vrs_entry_rate (방법론1 목표)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, load_uam_markings, get_uam_bonus, device

BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "uam_checkpoints_method1_v3")
LOG_FILE = os.path.join(BASE, "uam_method1_v3_continue.log")
LOAD_CKPT = os.path.join(SAVE_DIR, "method1_ep1500.pth")

ALT_LOW           = 20.0
ALT_HIGH          = 40.0
MAX_STEPS         = 6_000
EP_START          = 1_501
EP_END            = 3_000
BUFFER_SIZE       = 100_000
FIXED_ALPHA       = 0.03
EVAL_N            = 100
MARKING_THRESHOLD = 0.3
MARKING_BONUS     = 3.0

EVAL_ALT   = 30.0
EVAL_VZ    = 0.0
EVAL_VX    = 0.0
EVAL_TNORM = 0.5

log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')

def eval_fixed(agent, env, n=100):
    """고정 시나리오(alt=30, Vz=0, Vx=0, T_norm=0.5)로 n 에피소드 평가"""
    reasons     = Counter()
    vrs_entries = 0
    for _ in range(n):
        obs, _ = env.reset()
        env.state[0] = EVAL_VX
        env.state[1] = EVAL_VZ
        env.state[2] = EVAL_ALT
        env.state[3] = EVAL_TNORM
        obs = env.state.copy()
        done        = False
        vrs_entered = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('vrs_ratio', 0) >= 0.5:
                vrs_entered = True
        reasons[info.get('termination_reason', 'unknown')] += 1
        if vrs_entered:
            vrs_entries += 1
    return reasons, vrs_entries / n

def main():
    env = VRSEnv()
    env.reset_alt_low        = ALT_LOW
    env.reset_alt_high       = ALT_HIGH
    env.max_steps            = MAX_STEPS
    env.use_ontology_override = False
    env.use_vrs_penalty       = True
    env.use_time_penalty      = False

    os.chdir(BASE)
    markings = load_uam_markings()

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)
    agent.load(LOAD_CKPT)

    log(f"[Device] {device}")
    log(f"[Method] 1-v3 continue | ep{EP_START}~{EP_END} | 로드: {LOAD_CKPT}")
    log(f"[Markings] {len(markings)}개 로드")
    log(f"\n{'='*60}")
    log(f"method1-v3 이어받기 학습")
    log(f"  조건: ontology=OFF | vrs_penalty=ON | time_penalty=OFF | marking=ON(에피소드당 1회)")
    log(f"  평가: 고정 시나리오(alt={EVAL_ALT}m, Vz={EVAL_VZ}, T_norm={EVAL_TNORM}), N={EVAL_N}")
    log(f"{'='*60}\n")

    landing_count = 0
    ep_rewards    = []

    for episode in range(EP_START, EP_END + 1):
        obs, _ = env.reset()
        ep_reward   = 0.0
        done        = False
        landed      = False
        bonus_given = False  # 에피소드당 마킹 보너스 1회 제한

        while not done:
            zone_before = env._ontology(float(obs[1]), float(obs[3]))
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not bonus_given and zone_before in ("SAFE", "CAUTION"):
                b = get_uam_bonus(obs, markings, MARKING_THRESHOLD, MARKING_BONUS)
                if b > 0:
                    reward     += b
                    bonus_given = True
            reward = float(np.clip(reward, -200.0, 900.0))

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
            elapsed = episode - EP_START + 1
            avg10   = np.mean(ep_rewards[-10:])
            land_r  = landing_count / elapsed * 100
            log(f"Episode {episode:5d} | Avg(10): {avg10:9.2f} | 착지율: {land_r:.1f}%")

        if episode % 500 == 0:
            pth = os.path.join(SAVE_DIR, f"method1_ep{episode}.pth")
            agent.save(pth)
            log(f"\n[Checkpoint] ep{episode} → {pth}")
            log(f"[Eval] 고정 시나리오 N={EVAL_N} ...")
            reasons, vrs_rate = eval_fixed(agent, env, n=EVAL_N)
            log(f"  landed        : {reasons['landed']:3d}개 ({100*reasons['landed']/EVAL_N:.1f}%)")
            log(f"  timeout       : {reasons['timeout']:3d}개 ({100*reasons['timeout']/EVAL_N:.1f}%)")
            log(f"  crash         : {reasons.get('crash',0):3d}개")
            log(f"  vrs_entry_rate: {vrs_rate:.1%}  ← 방법론1 핵심 지표\n")

    elapsed = EP_END - EP_START + 1
    agent.save(os.path.join(SAVE_DIR, "method1_v3_continue_final.pth"))
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep{EP_START}~{EP_END})")
    log(f"  총 착지: {landing_count}/{elapsed} ({100*landing_count/elapsed:.1f}%)")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == "__main__":
    main()
