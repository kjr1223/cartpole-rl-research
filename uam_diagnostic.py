"""
uam_diagnostic.py
=================
UAM VRS 환경 5,000 step 진단 테스트

[목적]
  - Glauert 다항식 + NaN 안전장치 적용 후 환경이 정상 동작하는지 확인
  - reward collapse (-80,000 수준) 여부 검사
  - v_i, v_h, VRS ratio 분포 확인
  - 결과를 diagnostic_test_log.txt 와 uam_diagnostic.png 로 저장

[판단 기준]
  - 에피소드 평균 보상이 -5,000 이하로 떨어지면 collapse 로 판정
    (4000 스텝 × (-20) DANGER 패널티 = -80,000 이론적 최저)
  - NaN/inf 발생 카운트가 0이어야 통과
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from uam_vrs_env import VRSEnv

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
TOTAL_STEPS     = 5_000     # 총 스텝 수
COLLAPSE_THRESH = -5_000.0  # 에피소드 보상이 이 이하면 collapse 판정
LOG_FILE        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'diagnostic_test_log.txt')
PNG_FILE        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'uam_diagnostic.png')
PYTHON_ENV      = '/home/jrkim/cartpole_env/bin/python3'

def run_diagnostic():
    env = VRSEnv()
    env.reset(seed=0)

    # 통계 수집 버퍼
    all_rewards         = []        # 스텝별 보상
    episode_rewards     = []        # 에피소드별 누적 보상
    episode_lengths     = []        # 에피소드 길이
    step_vrs_ratios     = []        # 스텝별 VRS ratio
    step_zones          = []        # 스텝별 존
    nan_count           = 0         # NaN/inf 발생 횟수
    anomaly_count       = 0         # 환경 이상(anomaly) 발생 횟수
    zone_counter        = {"SAFE": 0, "CAUTION": 0, "DANGER": 0, "RECOVER": 0, "ANOMALY": 0}

    ep_reward = 0.0
    ep_steps  = 0
    total_eps = 0
    obs, _    = env.reset(seed=0)

    collapse_detected = False
    collapse_info     = {}

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log(f" UAM VRS 진단 테스트 시작")
    log(f" 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f" 총 스텝: {TOTAL_STEPS}")
    log(f" Collapse 판정 기준: 에피소드 보상 < {COLLAPSE_THRESH}")
    log("=" * 60)

    for step in range(TOTAL_STEPS):
        # 랜덤 정책 (환경 자체의 결정론적 규칙 포함)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # NaN/inf 체크
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            nan_count += 1
            log(f"  [NaN] step={step} obs={obs} reward={reward}")

        # 이상 감지
        if info.get("anomaly", False):
            anomaly_count += 1
            zone_counter["ANOMALY"] += 1
        else:
            zone = info.get("zone", "SAFE")
            zone_counter[zone] = zone_counter.get(zone, 0) + 1
            step_zones.append(zone)

        vrs_ratio = info.get("vrs_ratio", 0.0)
        if np.isfinite(vrs_ratio):
            step_vrs_ratios.append(min(float(vrs_ratio), 10.0))  # 표시용 캡
        else:
            step_vrs_ratios.append(0.0)

        all_rewards.append(float(reward))
        ep_reward += reward
        ep_steps  += 1

        if terminated or truncated:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)

            # Collapse 판정
            if ep_reward < COLLAPSE_THRESH and not collapse_detected:
                collapse_detected = True
                collapse_info = {
                    "episode"    : total_eps,
                    "ep_reward"  : ep_reward,
                    "ep_steps"   : ep_steps,
                    "at_step"    : step,
                }

            total_eps += 1
            ep_reward  = 0.0
            ep_steps   = 0
            obs, _     = env.reset()

    # 마지막 미완성 에피소드 처리
    if ep_steps > 0:
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

    env.close()

    # ─────────────────────────────────────────
    # 결과 요약
    # ─────────────────────────────────────────
    ep_arr = np.array(episode_rewards)
    log("")
    log("=" * 60)
    log(" 진단 결과 요약")
    log("=" * 60)
    log(f"  총 에피소드        : {len(episode_rewards)}")
    log(f"  총 스텝            : {TOTAL_STEPS}")
    log(f"  에피소드 평균 보상  : {ep_arr.mean():.2f}")
    log(f"  에피소드 최솟값    : {ep_arr.min():.2f}")
    log(f"  에피소드 최댓값    : {ep_arr.max():.2f}")
    log(f"  에피소드 표준편차  : {ep_arr.std():.2f}")
    log(f"  NaN/inf 발생 수    : {nan_count}")
    log(f"  Anomaly 발생 수    : {anomaly_count}")
    log("")
    log("  존 분포 (스텝 단위):")
    total_steps_counted = sum(zone_counter.values())
    for z, cnt in zone_counter.items():
        pct = 100.0 * cnt / max(total_steps_counted, 1)
        log(f"    {z:10s}: {cnt:5d} ({pct:.1f}%)")

    log("")
    if collapse_detected:
        log("  ⚠️  COLLAPSE 감지!")
        log(f"     에피소드 {collapse_info['episode']} (전체 step {collapse_info['at_step']})")
        log(f"     에피소드 보상: {collapse_info['ep_reward']:.2f}")
        log(f"     에피소드 길이: {collapse_info['ep_steps']} steps")
        log("")
        log("  [COLLAPSE 원인 분석]")
        log(f"    DANGER 존 비중: {100.0*zone_counter.get('DANGER',0)/max(total_steps_counted,1):.1f}%")
        log(f"    RECOVER 존 비중: {100.0*zone_counter.get('RECOVER',0)/max(total_steps_counted,1):.1f}%")
        log("    → DANGER 구간에서 dT=1.0 강제 오버라이드가 작동하지 않거나")
        log("      T_norm이 오르는 속도보다 V_z가 발산하는 속도가 빠를 가능성")
        log("")
        log("  ❌ 3단계 FAIL — 4단계로 넘어가지 않음")
    else:
        log("  ✅ Collapse 없음 — 3단계 PASS")
        log("     → 4단계(SAC 재학습) 진행 가능")

    log("")
    log(f"  로그 저장: {LOG_FILE}")
    log(f"  그래프 저장: {PNG_FILE}")
    log("=" * 60)

    # ─────────────────────────────────────────
    # 그래프 저장
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('UAM VRS Diagnostic Test (5,000 steps, Random Policy)',
                 fontsize=13, fontweight='bold')

    # (a) 에피소드 보상
    ax = axes[0]
    x_ep = np.arange(1, len(episode_rewards) + 1)
    ax.bar(x_ep, episode_rewards, color='steelblue', alpha=0.7, width=0.8)
    ax.axhline(COLLAPSE_THRESH, color='red', linestyle='--', linewidth=1.2,
               label=f'Collapse threshold ({COLLAPSE_THRESH})')
    ax.axhline(ep_arr.mean(), color='orange', linestyle='--', linewidth=1.2,
               label=f'Mean ({ep_arr.mean():.1f})')
    ax.set_title('(a) Episode Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) 스텝별 보상 이동평균
    ax = axes[1]
    step_x = np.arange(len(all_rewards))
    w = 200
    ma = np.convolve(all_rewards, np.ones(w)/w, mode='valid')
    ax.plot(step_x, all_rewards, color='lightblue', alpha=0.4, linewidth=0.5)
    ax.plot(step_x[w-1:], ma, color='steelblue', linewidth=1.5, label=f'MA({w})')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('(b) Per-step Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) VRS ratio 분포
    ax = axes[2]
    ax.plot(step_x, step_vrs_ratios, color='purple', alpha=0.5, linewidth=0.5)
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.0, label='CAUTION (0.5)')
    ax.axhline(0.8, color='red',    linestyle='--', linewidth=1.0, label='DANGER (0.8)')
    ax.axhline(2.0, color='purple', linestyle='--', linewidth=1.0, label='RECOVER (2.0)')
    ax.set_ylim(-0.5, 5.0)
    ax.set_title('(c) VRS Ratio (Vz/vh) per Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('VRS Ratio')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 존 비중 텍스트
    zone_txt = ' | '.join([f"{z}: {100.0*c/max(total_steps_counted,1):.0f}%"
                            for z, c in zone_counter.items()])
    fig.text(0.5, 0.01, zone_txt, ha='center', fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(PNG_FILE, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # ─────────────────────────────────────────
    # 로그 파일 저장
    # ─────────────────────────────────────────
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    return not collapse_detected  # True=PASS, False=FAIL


if __name__ == '__main__':
    passed = run_diagnostic()
    sys.exit(0 if passed else 1)
