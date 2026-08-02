"""
uam_eval_baseline.py
====================
베이스라인 (VRS 고려 없는 순수 SAC) best 모델 결정론적 평가
제안 방법(uam_eval_vrs_rate.py)과 동일한 방식으로 비교
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from uam_baseline_novrs import NoVRSEnv   # 같은 파일에서 가져옴
from uam_sac import SACAgent, device

BASE   = os.path.dirname(os.path.abspath(__file__))
CKPT   = os.path.join(BASE, "uam_checkpoints_baseline", "baseline_final.pth")
N_EVAL = 200

ALT_LOW  = 20.0
ALT_HIGH = 40.0
MAX_STEPS = 6_000

def main():
    env = NoVRSEnv()
    env.reset_alt_low  = ALT_LOW
    env.reset_alt_high = ALT_HIGH
    env.max_steps      = MAX_STEPS

    agent = SACAgent(auto_alpha=False, alpha=0.03)
    ckpt  = torch.load(CKPT, map_location=device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    print(f"[Device] {device}")
    print(f"[Load] {CKPT}")
    print(f"[평가 시작] {N_EVAL} 에피소드\n")

    total_steps      = 0
    total_vrs_steps  = 0
    landing_count    = 0
    vrs_episode_count = 0

    for ep in range(1, N_EVAL + 1):
        obs, _ = env.reset()
        done   = False
        landed = False
        ep_vrs = 0
        ep_steps = 0

        while not done:
            # deterministic=True: 탐험 노이즈 없음 (공정 비교)
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            vrs_ratio = info.get("vrs_ratio", 0)
            if vrs_ratio >= 0.5:
                ep_vrs          += 1
                total_vrs_steps += 1

            obs       = next_obs
            ep_steps += 1
            total_steps += 1

            if terminated and next_obs[1] <= 2.0 and next_obs[2] <= 1.0:
                landed = True

        if landed:
            landing_count += 1
        if ep_vrs > 0:
            vrs_episode_count += 1

        if ep % 20 == 0:
            vrs_rate  = total_vrs_steps / total_steps * 100
            land_rate = landing_count / ep * 100
            print(f"ep {ep:4d} | 착지율: {land_rate:.1f}% | "
                  f"VRS진입률(스텝): {vrs_rate:.1f}% | "
                  f"VRS진입 에피소드: {vrs_episode_count/ep*100:.1f}%")

    vrs_step_rate = total_vrs_steps / total_steps * 100
    vrs_ep_rate   = vrs_episode_count / N_EVAL * 100
    land_rate     = landing_count / N_EVAL * 100

    print(f"\n{'='*55}")
    print(f"[베이스라인 평가 완료] ep400 체크포인트, {N_EVAL} 에피소드")
    print(f"  착지율               : {land_rate:.1f}%")
    print(f"  VRS 진입률 (스텝 기준): {vrs_step_rate:.1f}%")
    print(f"  VRS 진입 에피소드 비율: {vrs_ep_rate:.1f}%")
    print(f"{'='*55}")
    print()
    print("=== 제안 방법 vs 베이스라인 비교 ===")
    print(f"  구분          | 착지율  | VRS진입률(스텝)")
    print(f"  제안 방법      | 59.0%  | 0.9%")
    print(f"  베이스라인     | {land_rate:.1f}%  | {vrs_step_rate:.1f}%")

if __name__ == "__main__":
    main()
