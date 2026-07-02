import gymnasium as gym
import pygame
import time

# 마킹된 데이터를 저장할 리스트
marked_data = []

# 환경 초기화
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

# pygame 창 가져오기 (키보드 입력 받으려고)
pygame.init()

print("=== 카트폴 마킹 시작 ===")
print("스페이스바: 지금 이 순간 마킹!")
print("Q키: 종료")
print("========================")

step = 0
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

    # 스페이스바 눌렸으면 마킹!
    if space_pressed:
        mark = {
            "step": step,
            "state": observation.tolist(),
            "time": time.time()
        }
        marked_data.append(mark)
        print(f"⭐ 마킹! step={step} | 상태={[round(x,3) for x in observation]}")

    # 랜덤 행동
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    step += 1
    time.sleep(0.05)

    if terminated or truncated:
        print(f"--- 막대 넘어짐 (step {step}) ---")
        observation, info = env.reset()

env.close()

print(f"\n총 {len(marked_data)}개 마킹됨")
for i, m in enumerate(marked_data):
    print(f"  마킹 {i+1}: step={m['step']} | 상태={[round(x,3) for x in m['state']]}")
