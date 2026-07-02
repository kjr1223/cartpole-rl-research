import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================
# Swing-up 환경 (시각화 없는 버전)
# 학습할 때는 화면 안 띄워야 빠르게 돌아가요
# ================================
import gymnasium as gym
from gymnasium import spaces

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self):
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 0.5
        self.dt        = 0.02
        self.force_mag = 10.0
        self.x_limit   = 2.4

        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = None

        # numpy 랜덤 시드
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # 막대가 아래에서 시작 (pi = 180도)
        self.state = np.array([
            0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1),
            0.0
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        # 행동에 따라 힘 방향 결정
        force = self.force_mag if action == 1 else -self.force_mag

        # 라그랑지안 동역학 계산
        # 카트+막대 시스템의 운동방정식
        total_mass = self.mass_cart + self.mass_pole
        ml = self.mass_pole * self.length
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        temp = (force + ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / \
                    (self.length * (4/3 - self.mass_pole * cos_t**2 / total_mass))
        x_acc = temp - ml * theta_acc * cos_t / total_mass

        # 상태 업데이트 (오일러 적분)
        x         = x         + self.dt * x_dot
        x_dot     = x_dot     + self.dt * x_acc
        theta     = theta     + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        # 각도를 -pi ~ pi 범위로 정규화
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # reward 설계
        # cos(theta)=1 이면 위(좋음), -1이면 아래(나쁨)
        upright_reward = np.cos(theta)
        x_penalty      = -0.1 * abs(x)        # 카트 중앙 벗어나면 감점
        vel_penalty    = -0.01 * theta_dot**2  # 막대 흔들림 감점
        reward = upright_reward + x_penalty + vel_penalty

        # 카트가 범위 벗어나면 종료
        terminated = bool(abs(x) > self.x_limit)

        return self.state.copy(), reward, terminated, False, {}


# ================================
# 신경망 정의
# 기본 카트폴이랑 구조 동일
# ================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # state 4개 입력 → 64 → 64 → action 2개 출력
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
        # state 4개 입력 → 64 → 64 → 점수 1개 출력
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)


# ================================
# 마킹 데이터 불러오기
# DB에서 swingup_markings 테이블 읽어옴
# ================================
def load_markings():
    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cart_pos, cart_vel, pole_ang, pole_vel
        FROM swingup_markings
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"마킹 데이터 {len(rows)}개 불러옴")
    return np.array(rows)


# ================================
# 마킹 보너스 계산
# 현재 state가 마킹된 state랑 가까우면 보너스 지급
# ================================
def get_marking_bonus(state, markings, threshold=0.5, bonus=3.0):
    # 마킹된 state들과의 거리 계산
    distances = np.linalg.norm(markings - state, axis=1)
    min_dist = np.min(distances)

    # threshold 이내면 보너스
    if min_dist < threshold:
        return bonus
    return 0.0


# ================================
# PPO 학습 함수
# use_marking=True면 마킹 보너스 반영
# ================================
def run_ppo(use_marking=False, episodes=3000):
    env = CartPoleSwingUpEnv()
    markings = load_markings() if use_marking else None

    policy  = PolicyNetwork()
    value_net = ValueNetwork()
    p_opt = torch.optim.Adam(policy.parameters(),   lr=3e-4)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    all_rewards = []
    label = "마킹 PPO" if use_marking else "기본 PPO"
    print(f"\n{label} 학습 시작 (총 {episodes} 에피소드)")
    print(f"{'에피소드':>8} | {'이번reward':>12} | {'최근10평균':>12}")
    print("-" * 40)

    for episode in range(episodes):
        states, actions, rewards, log_probs = [], [], [], []
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(obs)

            with torch.no_grad():
                probs = policy(state_tensor)

            dist   = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # 마킹 보너스 추가 (마킹 PPO만)
            if use_marking:
                reward += get_marking_bonus(obs, markings)

            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())

            obs = next_obs
            total_reward += reward

        all_rewards.append(total_reward)

        # PPO 업데이트
        states_t  = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_lp_t  = torch.FloatTensor(log_probs)

        # 감가율 0.99로 미래 보상 계산
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Advantage: 기대보다 얼마나 좋았는지
        with torch.no_grad():
            values = value_net(states_t).squeeze()
        advantages = returns_t - values

        # 3번 반복 업데이트
        for _ in range(3):
            probs    = policy(states_t)
            dist     = Categorical(probs)
            new_lp   = dist.log_prob(actions_t)

            # PPO 핵심: 너무 급격하게 바뀌지 않도록 clamp
            ratio   = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 0.8, 1.2)
            p_loss  = -torch.min(ratio * advantages,
                                 clipped * advantages).mean()

            v_loss = nn.MSELoss()(value_net(states_t).squeeze(), returns_t)

            p_opt.zero_grad(); p_loss.backward(); p_opt.step()
            v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        if (episode + 1) % 50 == 0:
            avg = np.mean(all_rewards[-10:])
            print(f"{episode+1:>8} | {total_reward:>12.1f} | {avg:>12.1f}")

    # 모델 저장
    fname = "swingup_marking.pth" if use_marking else "swingup_baseline.pth"
    torch.save(policy.state_dict(), fname)
    print(f"\n저장 완료: {fname}")
    print(f"최종 10에피소드 평균: {np.mean(all_rewards[-10:]):.1f}")

    return all_rewards


# ================================
# 두 버전 학습 후 그래프 비교
# ================================
baseline_rewards = run_ppo(use_marking=False, episodes=3000)
marking_rewards  = run_ppo(use_marking=True,  episodes=3000)

# 이동 평균 (그래프 부드럽게)
def moving_avg(data, window=10):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

baseline_avg = moving_avg(baseline_rewards)
marking_avg  = moving_avg(marking_rewards)
episodes = range(1, 3001)

# ================================
# 그래프 저장
# ================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Swing-up PPO: Baseline vs Marking', fontsize=16, fontweight='bold')

# 1. 학습 곡선
ax1 = axes[0, 0]
ax1.plot(episodes, baseline_avg, color='steelblue', linewidth=2, label='Baseline PPO')
ax1.plot(episodes, marking_avg,  color='tomato',    linewidth=2, label='Marking PPO')
ax1.set_title('Learning Curve (Moving Avg 10)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 구간별 막대그래프
ax2 = axes[0, 1]
sections = ['1-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000']
b_sec = [np.mean(baseline_rewards[i*500:(i+1)*500]) for i in range(6)]
m_sec = [np.mean(marking_rewards[i*500:(i+1)*500])  for i in range(6)]
x = np.arange(len(sections))
bars1 = ax2.bar(x - 0.2, b_sec, 0.4, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + 0.2, m_sec, 0.4, label='Marking',  color='tomato',    alpha=0.8)
ax2.set_title('Average Reward by Section')
ax2.set_xticks(x)
ax2.set_xticklabels(sections)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# 3. 원본 reward
ax3 = axes[1, 0]
ax3.scatter(episodes, baseline_rewards, color='steelblue', alpha=0.2, s=8)
ax3.scatter(episodes, marking_rewards,  color='tomato',    alpha=0.2, s=8)
ax3.plot(episodes, baseline_avg, color='steelblue', linewidth=1.5, label='Baseline')
ax3.plot(episodes, marking_avg,  color='tomato',    linewidth=1.5, label='Marking')
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
ax4.set_title('Marking - Baseline (Red = Marking Wins)')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Reward Difference')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('swingup_comparison.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: swingup_comparison.png")

# 최종 요약
print("\n===== 최종 요약 =====")
print(f"기본 PPO 전체 평균: {np.mean(baseline_rewards):.1f}")
print(f"마킹 PPO 전체 평균: {np.mean(marking_rewards):.1f}")
for i, sec in enumerate(sections):
    print(f"구간 {sec}: 기본={b_sec[i]:.1f} / 마킹={m_sec[i]:.1f}")
