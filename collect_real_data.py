import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
})

# ================================
# 신경망 정의
# ================================
class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2), nn.Softmax(dim=-1))
    def forward(self, x): return self.network(x)
    def get_action(self, x):
        return Categorical(self.forward(x)).sample().item()

class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU())
        self.mu      = nn.Linear(128,1)
        self.log_std = nn.Linear(128,1)
    def get_action(self, x):
        f = self.net(x)
        dist = Normal(self.mu(f), self.log_std(f).clamp(-20,2).exp())
        return torch.tanh(dist.rsample())

class BalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2), nn.Softmax(dim=-1))
    def get_action(self, x):
        return (Categorical(self.net(x)).sample().float()*2-1)

# ================================
# 스윙업 환경
# ================================
class SwingupEnv(gym.Env):
    def __init__(self, start_angle_deg=180):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle = np.radians(20)
        self.start_angle  = np.radians(start_angle_deg)
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.np_random = np.random.default_rng()

    def reset(self):
        self.state = np.array([0.0, 0.0,
            self.start_angle + self.np_random.uniform(-0.1, 0.1),
            0.0], dtype=np.float32)
        self.mode = "swingup"; self.switch_step = None
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole*self.length, np.cos(theta), np.sin(theta)
        tmp    = (force + ml*theta_dot**2*s) / mt
        th_acc = (self.gravity*s - c*tmp) / \
                 (self.length*(4/3 - self.mass_pole*c**2/mt))
        x_acc  = tmp - ml*th_acc*c/mt
        x         += self.dt*x_dot;    x_dot     += self.dt*x_acc
        theta     += self.dt*theta_dot; theta_dot += self.dt*th_acc
        theta      = ((theta+np.pi)%(2*np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode = "balance"; just_switched = True

        E_goal = self.mass_pole * self.gravity * self.length * 2
        pe = self.mass_pole*self.gravity*self.length*(1-np.cos(theta))
        ke = 0.5*self.mass_pole*(self.length*theta_dot)**2
        E_now = pe + ke

        if self.mode == "swingup":
            reward = (-0.5*abs(E_goal-E_now) + np.cos(theta)
                     + 0.1*abs(theta_dot) - 0.1*abs(x))
        else:
            reward = (np.cos(theta) - 0.1*abs(x)
                     - 0.1*theta_dot**2 + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, terminated, just_switched


# ================================
# 실제 데이터 수집 함수들
# ================================
def collect_cartpole_rewards(model_path, episodes=300, use_marking=False):
    """카트폴 PPO 실제 학습 재실행"""
    from torch.distributions import Categorical
    import torch.nn.functional as F

    env    = gym.make("CartPole-v1")
    policy = PPOPolicy()
    vnet   = nn.Sequential(
        nn.Linear(4,64), nn.Tanh(),
        nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))

    # 저장된 모델 불러오기
    policy.load_state_dict(torch.load(model_path, weights_only=True))

    all_rewards = []
    print(f"카트폴 평가 중: {model_path}")

    # 저장된 모델로 평가만
    policy.eval()
    for ep in range(episodes):
        obs, _ = env.reset(); done = False; total_r = 0
        while not done:
            with torch.no_grad():
                action = policy.get_action(torch.FloatTensor(obs))
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; total_r += r
        all_rewards.append(total_r)
        if (ep+1) % 50 == 0:
            print(f"  ep={ep+1} | 평균={np.mean(all_rewards[-50:]):.1f}")

    env.close()
    return np.array(all_rewards)


def evaluate_swingup(sac_path, balance_path, episodes=200,
                     start_angle_deg=180):
    """스윙업 전환 성공률 평가"""
    sac_actor = SACActor()
    sac_actor.load_state_dict(
        torch.load(sac_path, weights_only=True))
    sac_actor.eval()

    balance_actor = BalanceActor()
    balance_actor.load_state_dict(
        torch.load(balance_path, weights_only=True))
    balance_actor.eval()

    env = SwingupEnv(start_angle_deg=start_angle_deg)
    switched_list = []

    for ep in range(episodes):
        obs = env.reset(); done = False; switched = False
        while not done:
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0)
                if env.mode == "swingup":
                    a = sac_actor.get_action(st)
                else:
                    a = balance_actor.get_action(st)
            obs, r, term, just_switched = env.step(a.squeeze().item())
            done = term
            if just_switched: switched = True; done = True
        switched_list.append(1 if switched else 0)

    return np.array(switched_list)


# ================================
# 1. 카트폴 PPO 실제 데이터 수집
# ================================
print("\n=== 카트폴 PPO 데이터 수집 ===")
base_cp = collect_cartpole_rewards("ppo_baseline.pth", episodes=300)
mark_cp = collect_cartpole_rewards("ppo_marking.pth",  episodes=300)
np.save("real_cartpole_baseline.npy", base_cp)
np.save("real_cartpole_marking.npy",  mark_cp)
print(f"기본 PPO 평균: {np.mean(base_cp):.1f}")
print(f"마킹 PPO 평균: {np.mean(mark_cp):.1f}")


# ================================
# 2. Swing-up Curriculum 실제 데이터 수집
# ================================
print("\n=== Swing-up Curriculum 데이터 수집 ===")
stages      = [45, 90, 135, 180]
base_all_sw = []
mark_all_sw = []

for angle in stages:
    print(f"\n  [{angle}도] 평가 중...")
    base_sw = evaluate_swingup(
        f"stage_{angle}_baseline.pth", "balance_ppo.pth",
        episodes=200, start_angle_deg=angle)
    mark_sw = evaluate_swingup(
        f"stage_{angle}_marking.pth", "balance_ppo.pth",
        episodes=200, start_angle_deg=angle)
    base_all_sw.extend(base_sw.tolist())
    mark_all_sw.extend(mark_sw.tolist())
    print(f"  기본 SAC {angle}도: {np.mean(base_sw)*100:.1f}%")
    print(f"  마킹 SAC {angle}도: {np.mean(mark_sw)*100:.1f}%")

base_all_sw = np.array(base_all_sw)
mark_all_sw = np.array(mark_all_sw)
np.save("real_swingup_baseline.npy", base_all_sw)
np.save("real_swingup_marking.npy",  mark_all_sw)


# ================================
# 3. Residual RL 실제 데이터 수집
# ================================
print("\n=== Residual RL 데이터 수집 ===")

# Curriculum SAC (마킹) vs Residual RL 비교
# curriculum_marking.pth vs residual_baseline.pth

class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU())
        self.mu      = nn.Linear(128,1)
        self.log_std = nn.Linear(128,1)
    def get_action(self, x):
        f = self.net(x)
        dist = Normal(self.mu(f), self.log_std(f).clamp(-20,2).exp())
        return torch.tanh(dist.rsample())

res_all_sw = []
for angle in stages:
    print(f"\n  Residual [{angle}도] 평가 중...")
    res_actor = ResidualActor()
    res_actor.load_state_dict(
        torch.load(f"residual_{angle}_baseline.pth", weights_only=True))
    res_actor.eval()

    balance_actor = BalanceActor()
    balance_actor.load_state_dict(
        torch.load("balance_ppo.pth", weights_only=True))
    balance_actor.eval()

    env = SwingupEnv(start_angle_deg=angle)
    switched = []
    for ep in range(200):
        obs = env.reset(); done = False; sw = False
        while not done:
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0)
                if env.mode == "swingup":
                    # Residual: 물리 제어기 + SAC 보정
                    _, _, theta, theta_dot = obs
                    E_goal = 0.1*9.8*0.5*2
                    pe = 0.1*9.8*0.5*(1-np.cos(theta))
                    ke = 0.5*0.1*(0.5*theta_dot)**2
                    E_error = E_goal - (pe+ke)
                    base_a = np.clip(
                        2.0*E_error*np.cos(theta)*theta_dot/10.0,
                        -1.0, 1.0)
                    res_a = res_actor.get_action(st).squeeze().item()
                    a_val = np.clip(0.7*base_a + 0.3*res_a, -1.0, 1.0)
                else:
                    a_val = balance_actor.get_action(st).squeeze().item()
            obs, r, term, just_switched = env.step(a_val)
            done = term
            if just_switched: sw = True; done = True
        switched.append(1 if sw else 0)
    res_all_sw.extend(switched)
    print(f"  Residual RL {angle}도: {np.mean(switched)*100:.1f}%")

res_all_sw = np.array(res_all_sw)
np.save("real_residual_baseline.npy", res_all_sw)


# ================================
# 그래프 그리기
# ================================
def smooth(data, w=15):
    return np.array([np.mean(data[max(0,i-w):i+1])
                     for i in range(len(data))])

def success_rate(data, w=30):
    return np.array([np.mean(data[max(0,i-w):i+1])*100
                     for i in range(len(data))])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'CartPole & Swing-up: Human-in-the-Loop Marking Effect\n(Real Experimental Data)',
    fontsize=14, fontweight='bold', y=1.02)

# ─── 그래프 1: 카트폴 PPO ───
ax1 = axes[0]
x1  = range(1, 301)
b_s = smooth(base_cp, 15)
m_s = smooth(mark_cp, 15)

# 실제 표준편차 계산
b_std = np.array([np.std(base_cp[max(0,i-15):i+1])
                  for i in range(300)])
m_std = np.array([np.std(mark_cp[max(0,i-15):i+1])
                  for i in range(300)])
ax1.fill_between(x1, b_s-b_std*0.5, b_s+b_std*0.5,
                color='steelblue', alpha=0.15)
ax1.fill_between(x1, m_s-m_std*0.5, m_s+m_std*0.5,
                color='tomato', alpha=0.15)
ax1.plot(x1, b_s, color='steelblue', label='Baseline PPO')
ax1.plot(x1, m_s, color='tomato',    label='Marking PPO')
ax1.axhline(y=500, color='gray', linestyle='--',
           alpha=0.5, label='Max (500)')
ax1.set_title('(a) CartPole: Baseline vs Marking PPO')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward (steps survived)')
ax1.legend(loc='upper left')
ax1.set_xlim(0, 300); ax1.set_ylim(0, 550)

# ─── 그래프 2: Swing-up Curriculum ───
ax2   = axes[1]
x2    = range(1, len(base_all_sw)+1)
b_r   = success_rate(base_all_sw, 30)
m_r   = success_rate(mark_all_sw, 30)
ax2.plot(x2, b_r, color='steelblue', label='Baseline SAC')
ax2.plot(x2, m_r, color='tomato',    label='Marking SAC')

# 단계 구분선
for xb, lbl in [(200,'45→90°'),(400,'90→135°'),(600,'135→180°')]:
    ax2.axvline(x=xb, color='gray', linestyle='--', alpha=0.6)
    ax2.text(xb+5, 92, lbl, fontsize=8, color='gray')

ax2.set_title('(b) Swing-up: Curriculum SAC Baseline vs Marking')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Switch Success Rate (%)')
ax2.legend(loc='upper left')
ax2.set_xlim(0, len(base_all_sw)); ax2.set_ylim(0, 105)

# ─── 그래프 3: Curriculum vs Residual RL ───
ax3 = axes[2]
x3  = range(1, len(mark_all_sw)+1)
r_r = success_rate(res_all_sw, 30)
ax3.plot(x3, m_r, color='tomato',  label='Curriculum SAC (Marking)',
        linewidth=2.5)
ax3.plot(x3, r_r, color='purple',  label='Residual RL (Baseline)',
        linewidth=2.0, linestyle='--')

for xb, lbl in [(200,'45→90°'),(400,'90→135°'),(600,'135→180°')]:
    ax3.axvline(x=xb, color='gray', linestyle='--', alpha=0.6)
    ax3.text(xb+5, 92, lbl, fontsize=8, color='gray')

ax3.set_title('(c) Curriculum SAC vs Residual RL')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Switch Success Rate (%)')
ax3.legend(loc='upper left')
ax3.set_xlim(0, len(mark_all_sw)); ax3.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('paper_real_results.png', dpi=200,
           bbox_inches='tight', facecolor='white')
print("\n저장 완료: paper_real_results.png")

# ─── 막대 그래프 ───
fig2, ax4 = plt.subplots(figsize=(10, 5))
stage_labels = ['Stage1\n(45°)','Stage2\n(90°)',
                'Stage3\n(135°)','Stage4\n(180°)']

# 각 단계별 실제 성공률 계산
base_rates, mark_rates, res_rates = [], [], []
n = 200  # 단계당 에피소드
for i in range(4):
    base_rates.append(np.mean(base_all_sw[i*n:(i+1)*n])*100)
    mark_rates.append(np.mean(mark_all_sw[i*n:(i+1)*n])*100)
    res_rates.append(np.mean(res_all_sw[i*n:(i+1)*n])*100)

x   = np.arange(4)
w   = 0.25
b1  = ax4.bar(x-w,   base_rates, w, label='Baseline SAC',
             color='steelblue', alpha=0.85)
b2  = ax4.bar(x,     mark_rates, w, label='Marking SAC',
             color='tomato',    alpha=0.85)
b3  = ax4.bar(x+w,   res_rates,  w, label='Residual RL',
             color='purple',    alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.5:
            ax4.text(bar.get_x()+bar.get_width()/2., h+1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

ax4.set_title('Switch Success Rate by Stage (Real Data)',
             fontsize=13, fontweight='bold')
ax4.set_xlabel('Curriculum Stage')
ax4.set_ylabel('Switch Success Rate (%)')
ax4.set_xticks(x); ax4.set_xticklabels(stage_labels)
ax4.legend(); ax4.set_ylim(0, 115)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper_real_bar.png', dpi=200,
           bbox_inches='tight', facecolor='white')
print("저장 완료: paper_real_bar.png")

print("\n===== 실제 데이터 요약 =====")
print(f"카트폴 기본 PPO 평균: {np.mean(base_cp):.1f}")
print(f"카트폴 마킹 PPO 평균: {np.mean(mark_cp):.1f}")
for i, angle in enumerate(stages):
    print(f"스윙업 {angle}도: "
          f"기본={base_rates[i]:.1f}% / "
          f"마킹={mark_rates[i]:.1f}% / "
          f"Residual={res_rates[i]:.1f}%")
