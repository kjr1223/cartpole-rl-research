"""
uam_marking_db.py
==================
사람이 화살표 키로 UAM의 추력(dT)을 직접 조작해서 비행시키면서,
"이 상황은 중요한 전환/주의 시점이다"라고 판단되는 순간을 스페이스바로
직접 마킹하는 도구. (1단계: 핵심 데이터 수집)

cartpole_marking_db.py와 똑같은 아이디어(사람이 직접 컨텍스트를 DB에 남김)를
UAM에 적용한 것. 다른 점은:
    - CartPole 마킹 도구는 랜덤 행동을 구경하다 마킹하는 방식이었지만,
      여기서는 사람이 직접 위/아래 화살표로 추력을 조작해서 "내가 지금
      반응해야겠다"고 직접 판단한 시점을 마킹한다 (더 진짜 사람의 직관에 가까움).
    - UAM 환경은 물리 적분이 400Hz라 그대로 보여주면 너무 빨라서 못 따라감.
      그래서 물리는 400Hz 그대로 돌리되, 화면 갱신은 STEPS_PER_FRAME 스텝마다
      한 번씩만 해서 사람 눈에 보이는 "슬로우 리플레이"로 만든다.
    - pygame 창에는 cartpole처럼 실제 캐릭터 이미지가 아니라, 고도/속도/추력을
      막대그래프 + 텍스트로 보여주는 단순한 대시보드를 그린다.

조작법:
    위쪽 화살표 (누르고 있는 동안): 추력 증가 (dT=+1.0) → 하강 감속/상승
    아래쪽 화살표 (누르고 있는 동안): 추력 감소 (dT=-1.0) → 하강 가속
    아무 키도 안 누르면: dT=0 (추력 유지)
    스페이스바: 지금 이 순간을 마킹 (source='manual'로 DB에 저장)
    Q: 종료

이렇게 모은 "manual" 마킹 데이터는 uam_marking_augment.py에서 주변에
perturbation을 줘서 데이터를 늘리는 2단계(증강), VRS 온톨로지 규칙으로
증강 데이터가 여전히 같은 zone인지 검증하는 3단계(자동 라벨 검증)의
출발점(앵커)이 된다.

마킹된 데이터는 marking_data.db의 uam_switching_markings 테이블에 쌓이고,
이후 uam_sac.py 학습 루프에서 CAUTION 구간 보상 보너스(get_uam_bonus)로 쓰인다.
(DANGER/RECOVER 구간의 행동 강제는 마킹과 무관하게 uam_vrs_env.py의 온톨로지
규칙이 이미 처리하고 있어서, 마킹 DB는 "아직 강제되지 않은 애매한 구간"을
보완하는 역할만 한다.)
"""

import sqlite3
import time

import numpy as np
import pygame

from uam_vrs_env import VRSEnv

# ================================
# 재생 속도 설정
# 한 프레임(화면 갱신 1번)마다 물리를 몇 스텝 진행할지.
# 클수록 빨리 지나가고, 작을수록 느리게(자세히) 볼 수 있다.
# ================================
STEPS_PER_FRAME = 8     # 8스텝 * dt(1/400s) = 0.02s 시뮬레이션 시간 / 프레임
FPS             = 30    # 화면 갱신 속도 (사람이 보기 편한 정도)

# 화면 크기
WIN_W, WIN_H = 700, 500

# 온톨로지 구역별 색깔 (uam_vrs_env.py render()와 동일하게 맞춤)
ZONE_COLORS = {
    "SAFE":    (46, 204, 113),
    "CAUTION": (243, 156, 18),
    "DANGER":  (231, 76, 60),
    "RECOVER": (155, 89, 182),
}


# ================================
# DB 준비
# ================================
def setup_db():
    """
    marking_data.db에 UAM용 마킹 테이블을 만든다.
    CartPole의 switching_markings 테이블과 별도로 uam_switching_markings를 둔다
    (상태 변수 종류가 다르기 때문).

    source 컬럼: 'manual'(이 도구로 사람이 직접 마킹) vs 'augmented'
    (uam_marking_augment.py가 manual 마킹 주변에 perturbation을 줘서 만든 데이터).
    나중에 두 종류를 구분해서 쓸 수 있도록 처음부터 구분해서 저장한다.
    """
    conn = sqlite3.connect("marking_data.db")
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
    print("DB 준비 완료: marking_data.db (uam_switching_markings)")
    return conn, cursor


def draw_bar(screen, x, y, w, h, value, max_value, color, label, font):
    """
    값(value)을 max_value 기준으로 채운 막대그래프 하나를 그리는 헬퍼 함수.
    V_x, V_z, alt, T_norm을 똑같은 방식으로 그리기 위해 공통화함.
    """
    # 바깥 테두리
    pygame.draw.rect(screen, (60, 60, 60), (x, y, w, h), 2)
    # 채워진 부분 (value/max_value 비율만큼)
    ratio = max(0.0, min(1.0, value / max_value))
    fill_h = int(h * ratio)
    pygame.draw.rect(screen, color, (x, y + (h - fill_h), w, fill_h))
    # 라벨 텍스트
    text = font.render(f"{label}: {value:.2f}", True, (20, 20, 20))
    screen.blit(text, (x, y + h + 5))


def main():
    conn, cursor = setup_db()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("UAM VRS Marking Tool (SPACE: mark, Q: quit)")
    font      = pygame.font.SysFont(None, 22)
    big_font  = pygame.font.SysFont(None, 32)
    clock     = pygame.time.Clock()

    env = VRSEnv()
    obs, _ = env.reset()

    step = 0
    session_marks = 0
    running = True

    print("\n=== UAM VRS 마킹 시작 ===")
    print("위쪽 화살표: 추력 증가 (하강 감속)")
    print("아래쪽 화살표: 추력 감소 (하강 가속)")
    print("스페이스바: 지금 이 순간 마킹!")
    print("Q키: 종료")
    print("=========================\n")

    while running:
        # ----------------------------------------
        # 키보드 입력 확인 (프레임당 한 번)
        # ----------------------------------------
        space_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = True
                if event.key == pygame.K_q:
                    running = False

        # ----------------------------------------
        # 마킹! (스페이스바를 누른 순간의 상태를 DB에 저장)
        # ----------------------------------------
        if space_pressed:
            V_x, V_z, alt, T_norm = obs
            vrs_ratio = env._vrs_ratio(V_z, T_norm)
            zone      = env._ontology(V_z, T_norm)

            cursor.execute("""
                INSERT INTO uam_switching_markings
                    (step, V_x, V_z, alt, T_norm, vrs_ratio, zone, source, marked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                step, float(V_x), float(V_z), float(alt), float(T_norm),
                float(vrs_ratio), zone, "manual", time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            session_marks += 1
            print(f"⭐ 마킹 {session_marks}번째 저장! "
                  f"step={step} | alt={alt:.1f}m | V_z={V_z:.2f} | "
                  f"ratio={vrs_ratio:.2f} | zone={zone}")

        # ----------------------------------------
        # 화살표 키로 추력(dT) 조작
        # KEYDOWN 이벤트가 아니라 get_pressed()를 쓰는 이유: 키를 "누르고 있는 동안"
        # 계속 추력이 바뀌어야 하는데, KEYDOWN은 처음 누른 순간 한 번만 잡히기 때문.
        # ----------------------------------------
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            manual_dT = 1.0
        elif keys[pygame.K_DOWN]:
            manual_dT = -1.0
        else:
            manual_dT = 0.0

        # ----------------------------------------
        # 물리 시뮬레이션을 STEPS_PER_FRAME번 진행
        # (화면은 한 번만 갱신하지만 물리는 400Hz 그대로 유지)
        # 한 프레임 안에서는 사용자가 누른 키 상태를 그대로 유지해서 적용한다.
        # ----------------------------------------
        for _ in range(STEPS_PER_FRAME):
            action = np.array([manual_dT])
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                print(f"  (에피소드 종료 step={step}, 새 에피소드 시작)")
                obs, _ = env.reset()
                step = 0
                break

        # ----------------------------------------
        # 화면 그리기
        # ----------------------------------------
        V_x, V_z, alt, T_norm = obs
        vrs_ratio = env._vrs_ratio(V_z, T_norm)
        zone      = env._ontology(V_z, T_norm)
        zone_color = ZONE_COLORS.get(zone, (200, 200, 200))

        screen.fill((245, 245, 245))

        # 상단: 온톨로지 구역 표시 (배경색 + 텍스트)
        pygame.draw.rect(screen, zone_color, (0, 0, WIN_W, 50))
        zone_text = big_font.render(f"ZONE: {zone}  (V_z/v_h = {vrs_ratio:.2f})",
                                     True, (255, 255, 255))
        screen.blit(zone_text, (20, 10))

        # 상태 막대그래프들
        draw_bar(screen, 60,  100, 50, 250, V_x,    20.0, (52, 152, 219), "V_x",    font)
        draw_bar(screen, 180, 100, 50, 250, V_z,    10.0, (231, 76, 60),  "V_z",    font)
        draw_bar(screen, 300, 100, 50, 250, alt,   200.0, (46, 204, 113), "alt",    font)
        draw_bar(screen, 420, 100, 50, 250, T_norm,  1.0, (155, 89, 182), "T_norm", font)

        # 안내 텍스트 + 마킹 카운트 (pygame 기본 폰트가 한글을 지원 안 해서 영어로 표시)
        info_text = font.render(
            f"step={step}  |  Marks: {session_marks}  |  "
            f"UP/DOWN=thrust, SPACE=mark, Q=quit",
            True, (20, 20, 20))
        screen.blit(info_text, (20, 420))

        dT_text = font.render(f"current dT: {manual_dT:+.1f}", True, (20, 20, 20))
        screen.blit(dT_text, (20, 445))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    conn.close()

    print(f"\n이번 세션에서 {session_marks}개 마킹 저장 완료!")
    print("파일: marking_data.db (테이블: uam_switching_markings)")


if __name__ == "__main__":
    main()
