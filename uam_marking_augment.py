"""
uam_marking_augment.py
=======================
uam_marking_db.py로 모은 "manual" 마킹 데이터 주변에 랜덤 perturbation을 줘서
데이터를 늘리고(2단계: 증강), 늘어난 데이터가 여전히 같은 온톨로지 zone에
속하는지 VRSEnv의 규칙으로 재검증(3단계: 자동 라벨 검증)하는 스크립트.

왜 이 단계가 필요한가:
    사람이 직접 마킹한 데이터(manual)는 정확하지만 양이 적다(예: 82개).
    SAC 학습에서 get_uam_bonus()가 "현재 상태가 마킹된 점들 중 가장 가까운
    것과 얼마나 비슷한가"를 거리로 비교하는 방식이라, 마킹 데이터가 너무
    적으면 커버하는 상태 공간이 듬성듬성해서 보너스가 거의 안 터진다.

    그래서 각 manual 마킹 주변에 작은 노이즈를 더해서 비슷한 점들을 여러 개
    만들어낸다(perturbation). 그런데 노이즈를 더하다 보면 운 나쁘게 zone이
    바뀌어버릴 수 있다(예: CAUTION으로 마킹했는데 노이즈 때문에 SAFE나
    DANGER 경계를 넘어가버림). 이런 점을 그대로 "CAUTION"이라고 우기면서
    DB에 넣으면 잘못된 라벨이 섞이게 된다.

    그래서 perturbation 후에는 반드시 VRSEnv._ontology()로 zone을 다시
    계산해서, 원래 마킹의 zone과 같을 때만 살리고(자동 라벨 검증),
    다르면 버린다.
"""

import sqlite3

import numpy as np

from uam_vrs_env import VRSEnv

# ================================
# Perturbation 설정
# 마킹 1개당 몇 개의 가상 주변 데이터를 만들지, 그리고 각 상태변수에
# 얼마나 노이즈를 줄지(표준편차). 변수마다 값의 범위가 다르므로
# (V_x,V_z: -20~20/-10~10, alt: 0~200, T_norm: 0~1) 노이즈 크기도 다르게 잡았다.
# ================================
N_PERTURB_PER_MARK = 15     # manual 마킹 1개당 시도할 perturbation 개수

NOISE_STD = {
    "V_x":    0.3,    # m/s
    "V_z":    0.3,    # m/s  (ratio에 직접 영향이라 너무 크게 흔들면 안 됨)
    "alt":    5.0,    # m
    "T_norm": 0.03,   # [0,1] 스케일이라 작게
}


def load_manual_markings():
    """marking_data.db에서 source='manual'인 마킹만 불러온다."""
    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT V_x, V_z, alt, T_norm, zone
        FROM uam_switching_markings
        WHERE source = 'manual'
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"manual 마킹 {len(rows)}개 불러옴")
    return rows


def perturb_and_validate(rows, env):
    """
    각 manual 마킹 주변에 N_PERTURB_PER_MARK개의 perturbation을 만들고,
    zone이 원본과 같은 것만 골라서 반환한다.

    반환: [(V_x, V_z, alt, T_norm, vrs_ratio, zone), ...] 검증 통과한 것들
    """
    accepted = []
    n_tried = 0

    for V_x, V_z, alt, T_norm, orig_zone in rows:
        for _ in range(N_PERTURB_PER_MARK):
            n_tried += 1

            # 각 변수에 가우시안 노이즈를 더해서 "비슷하지만 똑같지는 않은" 점을 만든다
            new_V_x    = V_x    + np.random.normal(0, NOISE_STD["V_x"])
            new_V_z    = V_z    + np.random.normal(0, NOISE_STD["V_z"])
            new_alt    = alt    + np.random.normal(0, NOISE_STD["alt"])
            new_T_norm = T_norm + np.random.normal(0, NOISE_STD["T_norm"])

            # 환경이 허용하는 범위를 벗어나지 않게 클리핑
            new_V_x    = float(np.clip(new_V_x,    -20.0, 20.0))
            new_V_z    = float(np.clip(new_V_z,    -10.0, 10.0))
            new_alt    = float(np.clip(new_alt,      0.0, 200.0))
            new_T_norm = float(np.clip(new_T_norm,   0.0, 1.0))

            # 자동 라벨 검증: perturbation으로 만든 점의 zone을 실제 온톨로지
            # 규칙으로 다시 계산해서, 원래 마킹의 zone과 같을 때만 채택한다.
            new_ratio = env._vrs_ratio(new_V_z, new_T_norm)
            new_zone  = env._ontology(new_V_z, new_T_norm)

            if new_zone == orig_zone:
                accepted.append((new_V_x, new_V_z, new_alt, new_T_norm,
                                  new_ratio, new_zone))

    print(f"perturbation 시도: {n_tried}개 → 검증 통과: {len(accepted)}개 "
          f"({100*len(accepted)/n_tried:.1f}%)")
    return accepted


def save_augmented(accepted):
    """검증 통과한 perturbation 데이터를 source='augmented'로 DB에 저장."""
    import time

    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()

    for V_x, V_z, alt, T_norm, vrs_ratio, zone in accepted:
        cursor.execute("""
            INSERT INTO uam_switching_markings
                (step, V_x, V_z, alt, T_norm, vrs_ratio, zone, source, marked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            -1,   # 증강 데이터는 실제 에피소드 step이 없으므로 -1로 표시
            V_x, V_z, alt, T_norm, vrs_ratio, zone,
            "augmented", time.strftime("%Y-%m-%d %H:%M:%S")
        ))

    conn.commit()
    conn.close()
    print(f"augmented 마킹 {len(accepted)}개 저장 완료 (source='augmented')")


if __name__ == "__main__":
    env = VRSEnv()

    rows = load_manual_markings()
    if not rows:
        print("manual 마킹이 없습니다. 먼저 uam_marking_db.py로 마킹을 모아주세요.")
    else:
        accepted = perturb_and_validate(rows, env)
        save_augmented(accepted)

        # zone별 분포 확인 (manual + augmented 합쳐서 최종 몇 개씩인지)
        conn = sqlite3.connect("marking_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT source, zone, COUNT(*)
            FROM uam_switching_markings
            GROUP BY source, zone
        """)
        print("\n최종 DB 분포 (source, zone, count):")
        for row in cursor.fetchall():
            print(" ", row)
        conn.close()
