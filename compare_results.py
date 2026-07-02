import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

# ================================
# 학습 다시 돌려서 데이터 수집
# ================================
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

def load_markings():
    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT cart_pos, cart_vel, pole_ang, pole_vel FROM markings")
    rows = cursor.fetchall()
    conn.close()
    return np.array(rows)

def get_marking_bonus(state, markings, threshold=0.3, bonus=2.0):
    distances = np.linalg.norm(markings - state, axis=1)
    return bonus if np.min(distances) < threshold else 0.0

def run_ppo(use_marking=False, episodes=300):
    env = gym.make("CartPole-v1")
    markings = load_markings() if use_marking else None

    policy = PolicyNetwork()
    value_net = ValueNetwork()
    p_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    all_rewards = []
    label = "마킹 PPO" if use_marking else "기본 PPO"
    print(f"{label} 학습 중...")

    for episode in range(episodes):
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

            if use_marking:
                reward += get_marking_bonus(obs, markings)

            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_lp_t = torch.FloatTensor(log_probs)

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
            new_lp = dist.log_prob(actions_t)
            ratio = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            p_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            v_loss = nn.MSELoss()(value_net(states_t).squeeze(), returns_t)
            p_opt.zero_grad(); p_loss.backward(); p_opt.step()
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        if (episode + 1) % 50 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"  {episode+1}에피소드 | 최근 10평균: {avg:.1f}")

    env.close()
    return all_rewards

# ================================
# 두 버전 학습
# ================================
baseline_rewards = run_ppo(use_marking=False)
marking_rewards  = run_ppo(use_marking=True)

# ================================
# 이동 평균 계산 (그래프 부드럽게)
# ================================
def moving_avg(data, window=10):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

baseline_avg = moving_avg(baseline_rewards)
marking_avg  = moving_avg(marking_rewards)
episodes = range(1, 301)

# ================================
# 그래프 그리기
# ================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PPO Baseline vs Marking PPO Comparison', fontsize=16, fontweight='bold')

# 1. 학습 곡선 비교
ax1 = axes[0, 0]
ax1.plot(episodes, baseline_avg, color='steelblue', linewidth=2, label='Baseline PPO')
ax1.plot(episodes, marking_avg,  color='tomato',    linewidth=2, label='Marking PPO')
ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Max (500)')
ax1.set_title('Learning Curve (Moving Avg 10)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 구간별 평균 비교 막대그래프
ax2 = axes[0, 1]
sections = ['1-100', '101-200', '201-300']
baseline_sections = [
    np.mean(baseline_rewards[0:100]),
    np.mean(baseline_rewards[100:200]),
    np.mean(baseline_rewards[200:300])
]
marking_sections = [
    np.mean(marking_rewards[0:100]),
    np.mean(marking_rewards[100:200]),
    np.mean(marking_rewards[200:300])
]
x = np.arange(len(sections))
bars1 = ax2.bar(x - 0.2, baseline_sections, 0.4, label='Baseline PPO', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + 0.2, marking_sections,  0.4, label='Marking PPO',  color='tomato',    alpha=0.8)
ax2.set_title('Average Reward by Section')
ax2.set_xlabel('Episode Section')
ax2.set_ylabel('Average Reward')
ax2.set_xticks(x)
ax2.set_xticklabels(sections)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

# 3. 원본 reward (점)
ax3 = axes[1, 0]
ax3.scatter(episodes, baseline_rewards, color='steelblue', alpha=0.3, s=10, label='Baseline PPO')
ax3.scatter(episodes, marking_rewards,  color='tomato',    alpha=0.3, s=10, label='Marking PPO')
ax3.plot(episodes, baseline_avg, color='steelblue', linewidth=1.5)
ax3.plot(episodes, marking_avg,  color='tomato',    linewidth=1.5)
ax3.set_title('Raw Rewards + Trend')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Reward')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 차이값 (마킹 - 기본)
ax4 = axes[1, 1]
diff = [m - b for m, b in zip(marking_avg, baseline_avg)]
colors = ['tomato' if d > 0 else 'steelblue' for d in diff]
ax4.bar(episodes, diff, color=colors, alpha=0.6, width=1.0)
ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_title('Marking PPO - Baseline PPO (Positive = Marking Wins)')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Reward Difference')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장 완료: comparison.png")

# 최종 요약
print("\n===== 최종 요약 =====")
print(f"기본 PPO  전체 평균: {np.mean(baseline_rewards):.1f}")
print(f"마킹 PPO  전체 평균: {np.mean(marking_rewards):.1f}")
print(f"초반 100에피소드: 기본={baseline_sections[0]:.1f} / 마킹={marking_sections[0]:.1f}")
print(f"중반 100에피소드: 기본={baseline_sections[1]:.1f} / 마킹={marking_sections[1]:.1f}")
print(f"후반 100에피소드: 기본={baseline_sections[2]:.1f} / 마킹={marking_sections[2]:.1f}")
