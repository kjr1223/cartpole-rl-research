import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print("카트폴 시작!")
print(f"상태 의미: [카트위치, 카트속도, 막대각도, 막대각속도]")
print(f"초기 상태: {observation}")

for step in range(200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"step {step:3d} | 상태: {observation} | 보상: {reward}")
    time.sleep(0.05)
    
    if terminated or truncated:
        print(f"\n막대 넘어짐! {step+1}번째 스텝에서 종료")
        observation, info = env.reset()

env.close()
print("종료!")
