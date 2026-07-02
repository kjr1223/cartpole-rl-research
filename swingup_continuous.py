import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ================================
# 연속 행동 Swing-up 환경
# 기존이랑 다른 점:
#   action = -1.0 ~ +1.0 사이 연속값
#   force  = action * 10.0 (세기 조절 가능)
# ================================
class CartPoleSwingUpContinuous(gym.Env):
    def __init__(self):
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 0.5
        self.dt        = 0.02
        self.force_mag = 10.0
        self.x_limit   = 2.4

        # 상태 공간: [카트위치, 카트속도, 막대각도, 막대각속도]
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 행동 공간: -1.0 ~ +1.0 연속값
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.state = None
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # 막대가 아래에서 시작
        self.state = np.array([
            0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1),
            0.0
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        # 연속 행동: -1~+1 값을 실제 힘으로 변환
        force = float(action) * self.force_mag

        # 라그랑지안 동역학 계산
        total_mass = self.mass_cart + self.mass_pole
        ml    = self.mass_pole * self.length
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        temp      = (force + ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / \
                    (self.length * (4/3 - self.mass_pole * cos_t**2 / total_mass))
        x_acc     = temp - ml * theta_acc * cos_t / total_mass

        # 상태 업데이트
        x         = x         + self.dt * x_dot
        x_dot     = x_dot     + self.dt * x_acc
        theta     = theta     + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        theta     = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # 개선된 reward
        upright_reward = np.cos(theta)           # 위에 있을수록 +1
        energy_bonus   = 0.1 * self.length * (1 + np.cos(theta))  # 높을수록 보너스
        swing_bonus    = 0.5 if abs(theta) < np.pi / 2 else 0.0   # 위쪽 절반이면 보너스
        x_penalty      = -0.1 * abs(x)           # 카트 중앙 벗어나면 감점
        vel_penalty    = -0.001 * theta_dot**2   # 막대 흔들림 감점

        reward = upright_reward + energy_bonus + swing_bonus + x_penalty + vel_penalty

        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, terminated, False, {}


# ================================
# 연속 행동 PPO 신경망
# 기존 PPO랑 다른 점:
#   출력 = [평균, 표준편차]
#   Categorical 대신 Gaussian 분포 사용
# ================================
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        # 평균값 출력 (-1 ~ +1)
        self.mean_layer = nn.Linear(64, 1)
        # 표준편차 출력 (항상 양수)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = self.network(x)
        mean     = torch.tanh(self.mean_layer(features))  # -1 ~ +1 범위로
        std      = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        # Gaussian 분포에서 행동 샘플링
        dist     = Normal(mean, std)
        action   = dist.sample()
        action   = torch.clamp(action, -1.0, 1.0)  # -1~+1 범위 유지
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

class ContinuousValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)


# ================================
# 연속 행동 SAC 신경망
# 기존 SAC랑 다른 점:
#   Actor 출력 = [평균, 표준편차]
#   Critic 입력 = state + action 같이 받음
# ================================
class SACActorContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mean_layer    = nn.Linear(64, 1)
        self.log_std_layer = nn.Linear(64, 1)

    def forward(self, x):
        features = self.network(x)
        mean     = self.mean_layer(features)
        # 표준편차 범위 제한 (-20 ~ 2)
        log_std  = self.log_std_layer(features).clamp(-20, 2)
        std      = log_std.exp()
        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        dist      = Normal(mean, std)
        x_t       = dist.rsample()  # reparameterization trick
        action    = torch.tanh(x_t)  # -1 ~ +1 범위로

        # log_prob 보정 (tanh 변환에 의한 보정)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class SACCriticContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        # state 4개 + action 1개 = 5개 입력
        self.network = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # state와 action을 합쳐서 입력
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# ================================
# 연속 행동 PPO 학습
# ================================
def run_ppo_continuous(episodes=500):
    env       = CartPoleSwingUpContinuous()
    policy    = ContinuousPolicyNetwork()
    value_net = ContinuousValueNetwork()
    p_opt = torch.optim.Adam(policy.parameters(),    lr=3e-4)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    all_rewards = []
    print("\n연속 행동 PPO 학습 시작")
    print(f"{'에피소드':>8} | {'이번reward':>12} | {'최근10평균':>12}")
    print("-" * 40)

    for episode in range(episodes):
        states, actions, rewards, log_probs = [], [], [], []
        obs, _ = env.reset()
        done  = False
        total_reward = 0

        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = policy.get_action(state_t)

            action_val = action.squeeze().item()
            next_obs, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated

            states.append(obs)
            actions.append(action_val)
            rewards.append(reward)
            log_probs.append(log_prob.item())

            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        # PPO 업데이트
        states_t  = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(actions).unsqueeze(1)
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
            mean, std = policy.forward(states_t)
            dist      = Normal(mean, std)
            new_lp    = dist.log_prob(actions_t).sum(-1)

            ratio   = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            p_loss  = -torch.min(ratio * advantages,
                                 clipped * advantages).mean()
            v_loss  = nn.MSELoss()(value_net(states_t).squeeze(), returns_t)

            p_opt.zero_grad(); p_loss.backward(); p_opt.step()
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        if (episode + 1) % 50 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")

    return all_rewards


# ================================
# 연속 행동 SAC 학습
# ================================
def run_sac_continuous(episodes=500):
    env    = CartPoleSwingUpContinuous()
    actor  = SACActorContinuous()
    critic1 = SACCriticContinuous()
    critic2 = SACCriticContinuous()

    # Target Critic: 학습 안정화용
    target_critic1 = SACCriticContinuous()
    target_critic2 = SACCriticContinuous()
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor_opt   = torch.optim.Adam(actor.parameters(),   lr=3e-4)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    # 엔트로피 온도 파라미터 자동 조절
    log_alpha      = torch.zeros(1, requires_grad=True)
    alpha_opt      = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -1.0  # 연속 행동이므로 -action_dim

    # 리플레이 버퍼: 과거 경험 저장
    buffer      = []
    buffer_size = 10000
    batch_size  = 64
    gamma       = 0.99
    tau         = 0.005  # target network 업데이트 속도

    all_rewards = []
    print("\n연속 행동 SAC 학습 시작")
    print(f"{'에피소드':>8} | {'이번reward':>12} | {'최근10평균':>12}")
    print("-" * 40)

    for episode in range(episodes):
        obs, _ = env.reset()
        done   = False
        total_reward = 0

        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t)
            action_val = action.squeeze().item()

            next_obs, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated

            # 경험 버퍼에 저장
            buffer.append((obs, action_val, reward, next_obs, float(done)))
            if len(buffer) > buffer_size:
                buffer.pop(0)

            # 버퍼에 충분한 데이터 쌓이면 학습
            if len(buffer) >= batch_size:
                idx   = np.random.choice(len(buffer), batch_size, replace=False)
                batch = [buffer[i] for i in idx]

                s  = torch.FloatTensor(np.array([b[0] for b in batch]))
                a  = torch.FloatTensor(np.array([b[1] for b in batch])).unsqueeze(1)
                r  = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
                ns = torch.FloatTensor(np.array([b[3] for b in batch]))
                d  = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1)

                with torch.no_grad():
                    next_action, next_log_prob = actor.get_action(ns)
                    # Target Q값 계산
                    target_q1 = target_critic1(ns, next_action)
                    target_q2 = target_critic2(ns, next_action)
                    target_q  = torch.min(target_q1, target_q2)
                    # 엔트로피 보너스 포함
                    target_v  = r + gamma * (1 - d) * \
                                (target_q - log_alpha.exp() * next_log_prob)

                # Critic 업데이트
                c1_loss = F.mse_loss(critic1(s, a), target_v)
                c2_loss = F.mse_loss(critic2(s, a), target_v)
                critic1_opt.zero_grad(); c1_loss.backward(); critic1_opt.step()
                critic2_opt.zero_grad(); c2_loss.backward(); critic2_opt.step()

                # Actor 업데이트
                new_action, new_log_prob = actor.get_action(s)
                q1 = critic1(s, new_action)
                q2 = critic2(s, new_action)
                q  = torch.min(q1, q2)
                actor_loss = (log_alpha.exp() * new_log_prob - q).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                # Alpha 업데이트 (엔트로피 자동 조절)
                alpha_loss = -(log_alpha * (new_log_prob +
                               target_entropy).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                # Target Critic 천천히 업데이트
                for p, tp in zip(critic1.parameters(),
                                 target_critic1.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
                for p, tp in zip(critic2.parameters(),
                                 target_critic2.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")

    return all_rewards


# ================================
# 학습 실행 및 그래프 비교
# ================================
ppo_rewards = run_ppo_continuous(episodes=500)
sac_rewards = run_sac_continuous(episodes=500)

def moving_avg(data, window=10):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

ppo_avg = moving_avg(ppo_rewards)
sac_avg = moving_avg(sac_rewards)
eps = range(1, 501)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Swing-up Continuous: PPO vs SAC', fontsize=16, fontweight='bold')

# 학습 곡선
ax1 = axes[0, 0]
ax1.plot(eps, ppo_avg, color='steelblue', linewidth=2, label='PPO')
ax1.plot(eps, sac_avg, color='tomato',    linewidth=2, label='SAC')
ax1.set_title('Learning Curve (Moving Avg 10)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 구간별 막대그래프
ax2 = axes[0, 1]
sections = ['1-100', '101-200', '201-300', '301-400', '401-500']
p_sec = [np.mean(ppo_rewards[i*100:(i+1)*100]) for i in range(5)]
s_sec = [np.mean(sac_rewards[i*100:(i+1)*100]) for i in range(5)]
x = np.arange(len(sections))
bars1 = ax2.bar(x - 0.2, p_sec, 0.4, label='PPO', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + 0.2, s_sec, 0.4, label='SAC', color='tomato',    alpha=0.8)
ax2.set_title('Average Reward by Section')
ax2.set_xticks(x)
ax2.set_xticklabels(sections)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1 + bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# 원본 reward
ax3 = axes[1, 0]
ax3.scatter(eps, ppo_rewards, color='steelblue', alpha=0.2, s=8)
ax3.scatter(eps, sac_rewards, color='tomato',    alpha=0.2, s=8)
ax3.plot(eps, ppo_avg, color='steelblue', linewidth=1.5, label='PPO')
ax3.plot(eps, sac_avg, color='tomato',    linewidth=1.5, label='SAC')
ax3.set_title('Raw Rewards + Trend')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Reward')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 차이값
ax4 = axes[1, 1]
diff   = [s - p for p, s in zip(ppo_avg, sac_avg)]
colors = ['tomato' if d > 0 else 'steelblue' for d in diff]
ax4.bar(eps, diff, color=colors, alpha=0.6, width=1.0)
ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_title('SAC - PPO (Red = SAC Wins)')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Reward Difference')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('swingup_continuous_comparison.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: swingup_continuous_comparison.png")

print("\n===== 최종 요약 =====")
print(f"PPO 전체 평균: {np.mean(ppo_rewards):.1f}")
print(f"SAC 전체 평균: {np.mean(sac_rewards):.1f}")
for i, sec in enumerate(sections):
    print(f"구간 {sec}: PPO={p_sec[i]:.1f} / SAC={s_sec[i]:.1f}")
