import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import sqlite3
from torch.distributions import Categorical

# ================================
# 신경망 (기본 PPO랑 동일)
# ================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
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
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

# ================================
# DB에서 마킹 데이터 불러오기
# ================================
def load_markings():
    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT cart_pos, cart_vel, pole_ang, pole_vel FROM markings")
    rows = cursor.fetchall()
    conn.close()
    markings = np.array(rows)
    print(f"마킹 데이터 {len(markings)}개 불러옴")
    return markings

# ================================
# 마킹 보너스 계산
# ================================
def get_marking_bonus(state, markings, threshold=0.3, bonus=2.0):
    """
    현재 상태가 마킹된 상태랑 비슷하면 보너스 reward 지급
    threshold: 이 거리 이내면 '비슷하다'고 판단
    bonus: 보너스 reward 크기
    """
    distances = np.linalg.norm(markings - state, axis=1)
    min_dist = np.min(distances)
    
    if min_dist < threshold:
        return bonus
    return 0.0

# ================================
# 마킹 반영 PPO 학습
# ================================
def train():
    env = gym.make("CartPole-v1")
    markings = load_markings()
    
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    
    episode_rewards = []
    bonus_counts = []  # 에피소드마다 보너스 몇 번 받았는지
    
    print("\n=== 마킹 반영 PPO 학습 시작 ===")
    print(f"{'에피소드':>8} | {'이번 reward':>12} | {'최근 10평균':>12} | {'보너스 횟수':>10}")
    print("-" * 55)
    
    for episode in range(300):
        states, actions, rewards, log_probs = [], [], [], []
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        bonus_count = 0
        
        while not done:
            state_tensor = torch.FloatTensor(obs)
            
            with torch.no_grad():
                probs = policy(state_tensor)
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # 마킹 보너스 추가!
            bonus = get_marking_bonus(obs, markings)
            if bonus > 0:
                bonus_count += 1
            reward = reward + bonus
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            
            obs = next_obs
            total_reward += reward
        
        episode_rewards.append(total_reward)
        bonus_counts.append(bonus_count)
        
        # PPO 업데이트 (기본이랑 동일)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(log_probs)
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        with torch.no_grad():
            values = value_net(states_t).squeeze()
        advantages = returns_t - values
        
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
        
        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            avg_bonus = np.mean(bonus_counts[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f} | {avg_bonus:>10.1f}")
    
    torch.save(policy.state_dict(), "ppo_marking.pth")
    print("\n학습 완료! 저장: ppo_marking.pth")
    print(f"최종 10에피소드 평균: {np.mean(episode_rewards[-10:]):.1f}")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    train()
