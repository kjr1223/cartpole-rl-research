import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

# ================================
# 신경망 정의
# ================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),   # 입력: state 4개
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),   # 출력: 왼쪽/오른쪽 2개
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # 출력: 이 상태가 얼마나 좋은지 점수
        )
    
    def forward(self, x):
        return self.network(x)

# ================================
# PPO 학습
# ================================
def train():
    env = gym.make("CartPole-v1")
    
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    episode_rewards = []
    
    print("=== 기본 PPO 학습 시작 (마킹 없음) ===")
    print(f"{'에피소드':>8} | {'이번 reward':>12} | {'최근 10평균':>12}")
    print("-" * 45)
    
    for episode in range(300):
        # 데이터 수집
        states, actions, rewards, log_probs = [], [], [], []
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(obs)
            
            with torch.no_grad():
                probs = policy(state_tensor)
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            
            obs = next_obs
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # PPO 업데이트
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(log_probs)
        
        # 리턴 계산 (감가율 0.99)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        # Advantage 계산
        with torch.no_grad():
            values = value_net(states_t).squeeze()
        advantages = returns_t - values
        
        # PPO 업데이트 3번 반복
        for _ in range(3):
            probs = policy(states_t)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            
            value_loss = nn.MSELoss()(value_net(states_t).squeeze(), returns_t)
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        
        # 10 에피소드마다 출력
        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")
    
    # 모델 저장
    torch.save(policy.state_dict(), "ppo_baseline.pth")
    print("\n학습 완료! 저장: ppo_baseline.pth")
    print(f"최종 10에피소드 평균: {np.mean(episode_rewards[-10:]):.1f}")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    train()
