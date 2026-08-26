"""
uam_marking_semi_auto.py
========================
반자동 마킹 수집 도구 — Context DB 재수집용 (방법론 1 학습 준비)

[기존 uam_marking_db.py와의 차이]
  기존: 사람이 직접 추력 조종하면서 아무 때나 스페이스바 마킹
  신규: 스크립트가 에피소드 자동 실행 → ratio 0.38~0.42 감지 → 일시정지
        → 사람이 "이게 위험 전환점이 맞나?" 판단 → Y(마킹) / N(넘기기)

[수집 조건]
  초기조건: Vx=0, alt uniform(20~40m), Vz uniform(0~3m/s), T_norm=0.5
  실행 정책: baseline_no_time_penalty.pth (deterministic)
  후보 조건: vrs_ratio가 0.38~0.42 구간에 처음 진입하는 순간
  목표: Vz 다양성 확보 (기존 마킹이 Vz≈3.86에 몰렸던 문제 해결)

[조작법]
  Y 또는 스페이스바: 현재 후보 상태를 마킹 (DB에 저장)
  N 또는 엔터:       넘기기 (저장 안 함)
  Q:                 종료
"""

import os, sys, sqlite3, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame

from uam_vrs_env import VRSEnv
from uam_sac import SACAgent, device
import torch

# ================================
# 마킹 설정 
# ================================
CHECKPOINT    = os.path.join(os.path.dirname(__file__),
                "uam_checkpoints_fresh/baseline_no_time_penalty.pth")
DB_PATH       = os.path.join(os.path.dirname(__file__), "marking_data.db")
TARGET_LO     = 0.38   # ratio 후보 구간 하한
TARGET_HI     = 0.42   # ratio 후보 구간 상한
ALT_LOW       = 20.0
ALT_HIGH      = 40.0
VZ_LOW        = 2.0   # target ratio 도달 가능한 Vz 범위 (계산 근거: T_norm=0.3~0.6 기준 ratio=0.38에 필요한 Vz=2.1~2.97)
VZ_HIGH       = 3.0
TNORM_LOW     = 0.3   # T_norm 랜덤화 → (Vz, T_norm) 조합 다양성 확보
TNORM_HIGH    = 0.6
MAX_STEPS_EP  = 6000

WIN_W, WIN_H  = 750, 520
FPS           = 30

ZONE_COLORS = {
    "SAFE":    (46, 204, 113),
    "CAUTION": (243, 156, 18),
    "DANGER":  (231, 76, 60),
    "RECOVER": (155, 89, 182),
}


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


def draw_bar(screen, x, y, w, h, value, max_value, color, label, font):
    pygame.draw.rect(screen, (60, 60, 60), (x, y, w, h), 2)
    ratio  = max(0.0, min(1.0, value / max_value))
    fill_h = int(h * ratio)
    pygame.draw.rect(screen, color, (x, y + (h - fill_h), w, fill_h))
    text = font.render(f"{label}: {value:.3f}", True, (20, 20, 20))
    screen.blit(text, (x, y + h + 5))


def draw_candidate(screen, font, big_font, obs, vrs_ratio, zone, ep, total_marks, init_vz, init_tnorm):
    """후보 상태를 화면에 그리고 Y/N 입력 대기 UI 표시"""
    V_x, V_z, alt, T_norm = obs
    zone_color = ZONE_COLORS.get(zone, (200, 200, 200))

    screen.fill((245, 245, 245))

    # 상단 zone 배너
    pygame.draw.rect(screen, zone_color, (0, 0, WIN_W, 55))
    screen.blit(big_font.render(
        f"ZONE: {zone}   ratio = {vrs_ratio:.3f}   [후보 감지!]",
        True, (255, 255, 255)), (15, 12))

    # 상태 막대그래프
    draw_bar(screen,  60, 100, 55, 250, V_x,    20.0, (52, 152, 219),  "V_x",    font)
    draw_bar(screen, 185, 100, 55, 250, V_z,    10.0, (231, 76, 60),   "V_z",    font)
    draw_bar(screen, 310, 100, 55, 250, alt,   200.0, (46, 204, 113),  "alt",    font)
    draw_bar(screen, 435, 100, 55, 250, T_norm,  1.0, (155, 89, 182),  "T_norm", font)

    # VRS ratio 강조 표시
    pygame.draw.rect(screen, (230, 230, 230), (560, 100, 150, 80))
    screen.blit(big_font.render("VRS ratio", True, (60, 60, 60)), (565, 108))
    screen.blit(big_font.render(f"{vrs_ratio:.3f}", True, (200, 50, 50)), (590, 140))

    # 초기 조건 표시 (다양성 확인용)
    screen.blit(font.render(f"초기 Vz: {init_vz:.2f} m/s", True, (80, 80, 80)), (560, 195))
    screen.blit(font.render(f"초기 T_norm: {init_tnorm:.2f}", True, (80, 80, 80)), (560, 212))

    # 하단 안내
    pygame.draw.rect(screen, (200, 240, 200), (0, 420, WIN_W, 50))
    screen.blit(big_font.render(
        "Y / SPACE: 마킹 저장    N / ENTER: 넘기기    Q: 종료",
        True, (20, 80, 20)), (60, 432))

    screen.blit(font.render(
        f"ep={ep}  |  누적 마킹: {total_marks}개",
        True, (60, 60, 60)), (15, 480))

    pygame.display.flip()


def main():
    conn, cursor = setup_db()

    # 기존 마킹 수 확인
    cursor.execute("SELECT COUNT(*) FROM uam_switching_markings")
    existing = cursor.fetchone()[0]
    print(f"[DB] 기존 마킹 {existing}개 (새 마킹은 추가됩니다)")

    # 에이전트 로드
    agent = SACAgent(auto_alpha=False, alpha=0.03)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic.load_state_dict(ckpt['critic'])
    print(f"[Agent] {CHECKPOINT} 로드 완료")

    # 환경
    env = VRSEnv()
    env.max_steps = MAX_STEPS_EP

    pygame.init()
    screen   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("UAM Semi-Auto Marking (Y=마킹 / N=넘기기 / Q=종료)")
    font     = pygame.font.SysFont(None, 22)
    big_font = pygame.font.SysFont(None, 30)
    clock    = pygame.time.Clock()

    total_marks = 0
    ep          = 0
    running     = True

    print("\n=== 반자동 마킹 시작 ===")
    print("후보 상태가 감지되면 화면이 멈춥니다.")
    print("Y 또는 스페이스: 마킹 저장 / N 또는 엔터: 넘기기 / Q: 종료\n")

    while running:
        ep += 1

        # 에피소드 초기조건: Vx=0, Vz 랜덤, alt 랜덤
        env.reset_alt_low  = ALT_LOW
        env.reset_alt_high = ALT_HIGH
        obs, _ = env.reset()

        init_vz    = float(np.random.uniform(VZ_LOW, VZ_HIGH))
        init_tnorm = float(np.random.uniform(TNORM_LOW, TNORM_HIGH))
        env.state[0] = 0.0         # Vx = 0
        env.state[1] = init_vz     # Vz 랜덤 (2.0~3.0)
        env.state[3] = init_tnorm  # T_norm 랜덤 (0.3~0.6) → Vh 다양화로 마킹 다양성 확보
        obs = env.state.copy()

        step           = 0
        done           = False
        in_target_zone = False

        # reset 직후 ratio를 먼저 계산 (step 0 체크)
        vrs_ratio = env._vrs_ratio(init_vz, init_tnorm)
        zone      = env._ontology(init_vz, init_tnorm)
        hit_str   = "★ HIT" if TARGET_LO <= vrs_ratio <= TARGET_HI else "miss"
        print(f"ep{ep:04d} | Vz={init_vz:.2f} T={init_tnorm:.2f} "
              f"ratio={vrs_ratio:.3f} {hit_str}")

        while not done and running:
            # ── step 시작 전: 현재 obs의 ratio 체크 (step=0이면 reset 직후 값) ──

            # pygame 종료 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            # 후보 감지: 처음으로 0.38~0.42 구간에 진입한 순간 멈춤
            if not in_target_zone and TARGET_LO <= vrs_ratio <= TARGET_HI:
                in_target_zone = True  # 에피소드당 첫 번째 진입만 처리

                # 화면 표시 + Y/N 대기
                draw_candidate(screen, font, big_font,
                               obs, vrs_ratio, zone, ep, total_marks, init_vz, init_tnorm)

                waiting = True
                while waiting and running:
                    clock.tick(FPS)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_y, pygame.K_SPACE):
                                # 마킹 저장
                                V_x, V_z, alt_val, T_norm = obs
                                cursor.execute("""
                                    INSERT INTO uam_switching_markings
                                        (step, V_x, V_z, alt, T_norm,
                                         vrs_ratio, zone, source, marked_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (step, float(V_x), float(V_z),
                                      float(alt_val), float(T_norm),
                                      float(vrs_ratio), zone,
                                      "manual", time.strftime("%Y-%m-%d %H:%M:%S")))
                                conn.commit()
                                total_marks += 1
                                print(f"  ✅ 마킹 {total_marks}번째 | "
                                      f"ep={ep} step={step} | "
                                      f"init_Vz={init_vz:.2f} → "
                                      f"Vz={V_z:.2f} alt={alt_val:.1f} ratio={vrs_ratio:.3f}")
                                waiting = False

                            elif event.key in (pygame.K_n, pygame.K_RETURN):
                                V_x, V_z, alt_val, T_norm = obs
                                print(f"  ⏭  넘김 | ep={ep} step={step} | "
                                      f"init_Vz={init_vz:.2f} → "
                                      f"Vz={V_z:.2f} ratio={vrs_ratio:.3f}")
                                waiting = False

                            elif event.key == pygame.K_q:
                                running = False
                                waiting = False

            # 실행 중 화면 (후보 아닐 때는 간단히 상태만 표시)
            else:
                V_x, V_z, alt_val, T_norm = obs  # info 대신 obs 직접 사용
                screen.fill((240, 240, 240))
                zone_color = ZONE_COLORS.get(zone, (200, 200, 200))
                pygame.draw.rect(screen, zone_color, (0, 0, WIN_W, 45))
                screen.blit(big_font.render(
                    f"ZONE: {zone}   ratio={vrs_ratio:.3f}   "
                    f"ep={ep} step={step}",
                    True, (255, 255, 255)), (15, 10))
                screen.blit(font.render(
                    f"alt={alt_val:.1f}m   "
                    f"Vz={V_z:.2f}m/s   "
                    f"누적 마킹: {total_marks}개   "
                    f"init_Vz={init_vz:.2f}",
                    True, (40, 40, 40)), (15, 55))
                pygame.display.flip()
                clock.tick(FPS)

            # ── 실제 step 실행 (ratio 체크 이후) ──
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # 다음 루프 반복 전에 ratio/zone 갱신
            V_x, V_z, alt_val, T_norm = obs
            vrs_ratio = env._vrs_ratio(float(V_z), float(T_norm))
            zone      = env._ontology(float(V_z), float(T_norm))

    env.close()
    pygame.quit()
    conn.close()
    print(f"\n이번 세션 마킹 수: {total_marks}개")
    print(f"전체 마킹 (기존 포함): {existing + total_marks}개")
    print(f"파일: {DB_PATH}")


if __name__ == "__main__":
    main()
