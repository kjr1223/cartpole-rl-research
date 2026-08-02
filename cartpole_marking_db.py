import gymnasium as gym
import pygame
import sqlite3
import time

# ================================
# DB 준비
# ================================
conn = sqlite3.connect("marking_data.db")
cursor = conn.cursor()

# 테이블 없으면 새로 만들기
cursor.execute("""
    CREATE TABLE IF NOT EXISTS markings (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        step      INTEGER,
        cart_pos  REAL, # 카트 위치
        cart_vel  REAL, # 카트 속도
        pole_ang  REAL, # 막대 각도
        pole_vel  REAL, # 막대 각속도
        marked_at TEXT
    )
""")
conn.commit()
print("DB 준비 완료: marking_data.db")

# ================================
# 카트폴 환경
# ================================
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
pygame.init()

print("\n=== 카트폴 마킹 시작 ===")
print("스페이스바: 지금 이 순간 마킹!")
print("Q키: 종료")
print("========================\n")

step = 0
session_marks = 0
running = True

while running:
    # 키보드 입력 확인
    space_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_pressed = True
            if event.key == pygame.K_q:
                running = False

    # 마킹!
    if space_pressed:
        cart_pos, cart_vel, pole_ang, pole_vel = observation

        # DB에 저장
        cursor.execute("""
            INSERT INTO markings (step, cart_pos, cart_vel, pole_ang, pole_vel, marked_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            step,
            float(cart_pos),
            float(cart_vel),
            float(pole_ang),
            float(pole_vel),
            time.strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        session_marks += 1

        print(f"⭐ 마킹 {session_marks}번째 저장! "
              f"step={step} | "
              f"각도={pole_ang:.3f} | "
              f"카트위치={cart_pos:.3f}")

    # 랜덤 행동
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    step += 1
    time.sleep(0.05)

    if terminated or truncated:
        print(f"  (막대 넘어짐 step={step})")
        observation, info = env.reset()

env.close()
conn.close()

print(f"\n이번 세션에서 {session_marks}개 마킹 저장 완료!")
print("파일: marking_data.db")
