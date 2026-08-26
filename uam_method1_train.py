"""
uam_method1_train.py
====================
방법론 1 (Avoidance)
  use_ontology_override = False  (강제 개입 없음 — 회피는 패널티+마킹으로)
  use_vrs_penalty       = True   (VRS 이차패널티 ON)
  use_time_penalty      = False  (baseline 조건 통일)
  Context DB 마킹       = ON     (ratio≈0.38~0.42 경계 직전 보너스)

[마킹 보너스 설계]
  zone in (SAFE, CAUTION)이고 마킹과 정규화 거리 < 0.3이면 +3.0
  SAFE 전체를 포함하지만 실제 발동은 마킹(ratio≈0.4) 근처일 때만 — 거리 조건이 2차 필터

[핵심 지표]
  landed        : 착지율 (baseline 재현 학습과 비교)
  vrs_entry_rate: 에피소드 중 VRS 진입(ratio≥0.5) 발생 비율 — 방법론1 목표 지표

[평가 시나리오]
  학습: 랜덤 초기조건 (alt 20~40m 균등, Vz/T_norm 랜덤)
  평가: 고정 시나리오 (alt=30m, Vz=0, Vx=0, T_norm=0.5)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from collections import Counter

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, load_uam_markings, get_uam_bonus, device

BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "uam_checkpoints_method1_v3")
LOG_FILE = os.path.join(BASE, "uam_method1_v3.log")

ALT_LOW          = 20.0
ALT_HIGH         = 40.0
MAX_STEPS        = 6_000
EPISODES         = 3_000
BUFFER_SIZE      = 100_000
FIXED_ALPHA      = 0.03
CHUNK            = 500
EVAL_N           = 100
MARKING_THRESHOLD = 0.3
MARKING_BONUS     = 3.0

# 고정 평가 시나리오 초기조건
EVAL_ALT    = 30.0
EVAL_VZ     = 0.0
EVAL_VX     = 0.0
EVAL_TNORM  = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)
log_fh = open(LOG_FILE, 'w', buffering=1)

def log(msg):
    print(msg)
    log_fh.write(msg + '\n')

def eval_fixed(agent, env, n=100):
    """고정 시나리오(alt=30, Vz=0, Vx=0, T_norm=0.5)로 n 에피소드 평가"""
    reasons     = Counter()
    vrs_entries = 0  # VRS 진입(ratio≥0.5) 발생 에피소드 수

    for _ in range(n):
        obs, _ = env.reset()
        # 고정 초기조건 강제 설정
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
    env.reset_alt_low       = ALT_LOW
    env.reset_alt_high      = ALT_HIGH
    env.max_steps           = MAX_STEPS
    env.use_ontology_override = False  # 강제 개입 없음
    env.use_vrs_penalty       = True   # VRS 패널티 ON
    env.use_time_penalty      = False  # 시간 패널티 OFF

    # Context DB 마킹 로드
    os.chdir(BASE)  # marking_data.db 경로 기준
    markings = load_uam_markings()

    agent = SACAgent(auto_alpha=False, alpha=FIXED_ALPHA, buffer_size=BUFFER_SIZE)

    log(f"[Device] {device}")
    log(f"[Method] 1-v3 (Avoidance) | ontology=OFF | vrs_penalty=ON | time_penalty=OFF | marking=ON")
    log(f"[Markings] {len(markings)}개 로드")
    log(f"\n{'='*60}")
    log(f"방법론 1 학습 (ep1~{EPISODES})")
    log(f"  보상: 하강(+alt_drop) | VRS패널티(×30) | 마킹보너스(+3.0) | 착지+500 | Tier+100×3")
    log(f"  평가: 고정 시나리오(alt={EVAL_ALT}m, Vz={EVAL_VZ}, T_norm={EVAL_TNORM}), N={EVAL_N}")
    log(f"  비교: baseline_repro(ontology ON, vrs ON, marking OFF)")
    log(f"{'='*60}\n")

    landing_count = 0
    ep_rewards    = []

    for episode in range(1, EPISODES + 1):
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

            # 마킹 보너스: SAFE 후반부~CAUTION 구간, 에피소드당 최초 1회만
            if not bonus_given and zone_before in ("SAFE", "CAUTION"):
                b = get_uam_bonus(obs, markings, MARKING_THRESHOLD, MARKING_BONUS)
                if b > 0:
                    reward     += b
                    bonus_given = True
            reward = float(np.clip(reward, -200.0, 900.0))  # bonus 포함 후 clip

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
            pth = os.path.join(SAVE_DIR, f"method1_ep{episode}.pth")
            agent.save(pth)
            log(f"\n[Checkpoint] ep{episode} → {pth}")
            log(f"[Eval] 고정 시나리오 N={EVAL_N} ...")
            reasons, vrs_rate = eval_fixed(agent, env, n=EVAL_N)
            log(f"  landed        : {reasons['landed']:3d}개 ({100*reasons['landed']/EVAL_N:.1f}%)")
            log(f"  timeout       : {reasons['timeout']:3d}개 ({100*reasons['timeout']/EVAL_N:.1f}%)")
            log(f"  crash         : {reasons.get('crash', 0):3d}개")
            log(f"  vrs_entry_rate: {vrs_rate:.1%}  ← 방법론1 핵심 지표")
            log(f"  [비교] baseline_repro 결과 대기 중\n")

    agent.save(os.path.join(SAVE_DIR, "method1_final.pth"))
    log(f"\n{'='*60}")
    log(f"학습 완료 (ep1~{EPISODES})")
    log(f"  총 착지: {landing_count}/{EPISODES} ({100*landing_count/EPISODES:.1f}%)")
    log(f"{'='*60}")
    log_fh.close()

if __name__ == "__main__":
    main()
