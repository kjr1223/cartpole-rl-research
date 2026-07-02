import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

# ================================
# SAC 신경망 3개
# Actor: 행동 결정
# Critic 1, 2: 행동 평가 (2개 써서 안정적)
# ================================

class Actor(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        # state 4개 → 64 → 64 → action 확률 2개
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # softmax로 확률 변환
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, x):
        probs = self.forward(x)
        dist  = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs


class Critic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        # state 4개 → 64 → 64 → 각 action의 Q값 2개
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, action_dim) # 각 행동의 Q값 출력
        )

    def forward(self, x):
        return self.network(x)


# ================================
# SAC 에이전트
# ================================
class SACAgent:
    def __init__(self, state_dim=4, action_dim=2):
        self.actor    = Actor(state_dim, action_dim)
        self.critic1  = Critic(state_dim, action_dim)
        self.critic2  = Critic(state_dim, action_dim)

        # Target Critic: 학습 안정화용 (가중치 천천히 업데이트)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=3e-4)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 엔트로피 온도 파라미터
        # alpha가 클수록 다양한 행동 시도
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -0.5  # 목표 엔트로피

        # 리플레이 버퍼: 과거 경험 저장해두고 랜덤으로 꺼내서 학습
        self.buffer = []
        self.buffer_size = 10000
        self.batch_size  = 64

        self.gamma = 0.99   # 감가율
        self.tau   = 0.005  # target network 업데이트 속도

    @property
    def alpha(self):
        # 엔트로피 가중치 (항상 양수)
        return self.log_alpha.exp()

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, probs = self.actor.get_action(state_t)
        return action.item()

    def store(self, state, action, reward, next_state, done):
        # 경험을 버퍼에 저장
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)  # 버퍼 꽉 차면 오래된 것 삭제

    def update(self):
        # 버퍼에 충분한 데이터 없으면 학습 안 함
        if len(self.buffer) < self.batch_size:
            return

        # 랜덤으로 배치 샘플링
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        states     = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions    = torch.LongTensor(np.array([b[1] for b in batch]))
        rewards    = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones      = torch.FloatTensor(np.array([b[4] for b in batch]))

        with torch.no_grad():
            # 다음 상태에서 행동 샘플링
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)

            # Target Q값 계산 (두 critic 중 작은 값 사용 → 과대평가 방지)
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_q  = torch.min(next_q1, next_q2)

            # 엔트로피 보너스 포함한 target Q값
            next_v  = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * next_v

        # Critic 업데이트
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze()
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze()

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        # Actor 업데이트
        probs     = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q  = torch.min(q1, q2)

        # reward 최대화 + 엔트로피 최대화
        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # 엔트로피 온도 alpha 업데이트
        entropy    = -(probs * log_probs).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Target Critic 천천히 업데이트 (tau=0.005)
        for p, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ================================
# PPO (기존 코드 그대로)
# ================================
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

def run_ppo(episodes=300):
    env = gym.make("CartPole-v1")
    policy    = PolicyNetwork()
    value_net = ValueNetwork()
    p_opt = torch.optim.Adam(policy.parameters(),    lr=3e-4)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    all_rewards = []
    print("\nPPO 학습 시작")
    print(f"{'에피소드':>8} | {'이번reward':>12} | {'최근10평균':>12}")
    print("-" * 40)

    for episode in range(episodes):
        states, actions, rewards, log_probs = [], [], [], []
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_t = torch.FloatTensor(obs)
            with torch.no_grad():
                probs = policy(state_t)
            dist   = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(obs); actions.append(action.item())
            rewards.append(reward); log_probs.append(log_prob.item())
            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        states_t  = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_lp_t  = torch.FloatTensor(log_probs)

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
            probs   = policy(states_t)
            dist    = Categorical(probs)
            new_lp  = dist.log_prob(actions_t)
            ratio   = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            p_loss  = -torch.min(ratio * advantages, clipped * advantages).mean()
            v_loss  = nn.MSELoss()(value_net(states_t).squeeze(), returns_t)
            p_opt.zero_grad(); p_loss.backward(); p_opt.step()
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        if (episode + 1) % 30 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")

    env.close()
    return all_rewards


# ================================
# SAC 학습
# ================================
def run_sac(episodes=300):
    env   = gym.make("CartPole-v1")
    agent = SACAgent()

    all_rewards = []
    print("\nSAC 학습 시작")
    print(f"{'에피소드':>8} | {'이번reward':>12} | {'최근10평균':>12}")
    print("-" * 40)

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 경험 버퍼에 저장
            agent.store(obs, action, reward, next_obs, float(done))

            # 매 스텝마다 업데이트
            agent.update()

            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        if (episode + 1) % 30 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")

    env.close()
    return all_rewards


# ================================
# 학습 실행 및 비교
# ================================
ppo_rewards = run_ppo(episodes=300)
sac_rewards = run_sac(episodes=300)

def moving_avg(data, window=10):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

ppo_avg = moving_avg(ppo_rewards)
sac_avg = moving_avg(sac_rewards)
eps = range(1, 301)

# ================================
# 그래프
# ================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CartPole: PPO vs SAC', fontsize=16, fontweight='bold')

# 학습 곡선
ax1 = axes[0]
ax1.plot(eps, ppo_avg, color='steelblue', linewidth=2, label='PPO')
ax1.plot(eps, sac_avg, color='tomato',    linewidth=2, label='SAC')
ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Max(500)')
ax1.set_title('Learning Curve (Moving Avg 10)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 구간별 막대
ax2 = axes[1]
sections = ['1-100', '101-200', '201-300']
p_sec = [np.mean(ppo_rewards[i*100:(i+1)*100]) for i in range(3)]
s_sec = [np.mean(sac_rewards[i*100:(i+1)*100]) for i in range(3)]
x = np.arange(len(sections))
bars1 = ax2.bar(x - 0.2, p_sec, 0.4, label='PPO', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + 0.2, s_sec, 0.4, label='SAC', color='tomato',    alpha=0.8)
ax2.set_title('Average Reward by Section')
ax2.set_xticks(x)
ax2.set_xticklabels(sections)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1 + bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('ppo_vs_sac.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: ppo_vs_sac.png")

print("\n===== 최종 요약 =====")
print(f"PPO 전체 평균: {np.mean(ppo_rewards):.1f}")
print(f"SAC 전체 평균: {np.mean(sac_rewards):.1f}")
for i, sec in enumerate(sections):
    print(f"구간 {sec}: PPO={p_sec[i]:.1f} / SAC={s_sec[i]:.1f}")
