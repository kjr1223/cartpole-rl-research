"""
uam_marking_auto.py
===================
완전 자동 마킹 수집 — headless, 사람 개입 없음

[설계]
  uam_marking_semi_auto.py와 동일한 수집 조건이지만
  HIT 감지 시 자동으로 DB 저장 (Y 자동 입력)
  pygame 없음 → 백그라운드 실행 가능

[수집 조건]
  초기: Vx=0, alt uniform(20~40m), Vz uniform(2.0~3.0), T_norm uniform(0.3~0.6)
  후보: vrs_ratio 0.38~0.42 구간에 처음 진입하는 순간 (step 0 포함)
  정책: baseline_no_time_penalty.pth (deterministic)
  목표: TARGET_MARKS개 수집 후 자동 종료
"""

import os, sys, sqlite3, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device

CHECKPOINT   = os.path.join(os.path.dirname(__file__),
               "uam_checkpoints_fresh/baseline_no_time_penalty.pth")
DB_PATH      = os.path.join(os.path.dirname(__file__), "marking_data.db")

TARGET_LO    = 0.38
TARGET_HI    = 0.42
ALT_LOW      = 20.0
ALT_HIGH     = 40.0
VZ_LOW       = 2.0
VZ_HIGH      = 3.0
TNORM_LOW    = 0.3
TNORM_HIGH   = 0.6
MAX_STEPS_EP = 6000
TARGET_MARKS = 200   # 이 수 달성하면 자동 종료


def setup_db():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uam_switching_markings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            step      INTEGER,
            V_x       REAL,
            V_z       REAL,
            alt       REAL,
            T_norm    REAL,
            vrs_ratio REAL,
            zone      TEXT,
            source    TEXT,
            marked_at TEXT
        )
    """)
    conn.commit()
    return conn, cursor


def main():
    conn, cursor = setup_db()
    cursor.execute("SELECT COUNT(*) FROM uam_switching_markings")
    existing = cursor.fetchone()[0]
    print(f"[DB] 기존 마킹 {existing}개")

    agent = SACAgent(auto_alpha=False, alpha=0.03)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic.load_state_dict(ckpt['critic'])
    print(f"[Agent] {CHECKPOINT} 로드 완료")

    env = VRSEnv()
    env.reset_alt_low  = ALT_LOW
    env.reset_alt_high = ALT_HIGH
    env.max_steps      = MAX_STEPS_EP

    total_marks = 0
    ep          = 0

    print(f"\n=== 자동 마킹 시작 (목표: {TARGET_MARKS}개) ===\n")

    while total_marks < TARGET_MARKS:
        ep += 1

        obs, _ = env.reset()
        init_vz    = float(np.random.uniform(VZ_LOW, VZ_HIGH))
        init_tnorm = float(np.random.uniform(TNORM_LOW, TNORM_HIGH))
        env.state[0] = 0.0
        env.state[1] = init_vz
        env.state[3] = init_tnorm
        obs = env.state.copy()

        step           = 0
        done           = False
        in_target_zone = False

        vrs_ratio = env._vrs_ratio(init_vz, init_tnorm)
        zone      = env._ontology(init_vz, init_tnorm)

        while not done and total_marks < TARGET_MARKS:
            # HIT 감지 → 자동 저장
            if not in_target_zone and TARGET_LO <= vrs_ratio <= TARGET_HI:
                in_target_zone = True
                V_x, V_z, alt_val, T_norm = obs
                cursor.execute("""
                    INSERT INTO uam_switching_markings
                        (step, V_x, V_z, alt, T_norm, vrs_ratio, zone, source, marked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (step, float(V_x), float(V_z), float(alt_val), float(T_norm),
                      float(vrs_ratio), zone, "auto", time.strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                total_marks += 1
                print(f"  ✅ [{total_marks:3d}/{TARGET_MARKS}] "
                      f"ep={ep:4d} step={step:4d} | "
                      f"Vz={V_z:.2f} T={T_norm:.2f} alt={alt_val:.1f} ratio={vrs_ratio:.3f} {zone}")
                break  # 에피소드당 1개 마킹 후 바로 다음 에피소드로

            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            V_x, V_z, alt_val, T_norm = obs
            vrs_ratio = env._vrs_ratio(float(V_z), float(T_norm))
            zone      = env._ontology(float(V_z), float(T_norm))

        if ep % 50 == 0:
            print(f"  [진행] ep={ep} | 누적 마킹: {total_marks}/{TARGET_MARKS}")

    env.close()
    conn.close()
    print(f"\n=== 완료 ===")
    print(f"  수집한 마킹: {total_marks}개")
    print(f"  총 에피소드: {ep}개 (적중률: {100*total_marks/ep:.1f}%)")
    print(f"  전체 마킹 (기존 포함): {existing + total_marks}개")
    print(f"  파일: {DB_PATH}")


if __name__ == "__main__":
    main()
