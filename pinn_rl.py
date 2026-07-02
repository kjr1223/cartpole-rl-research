import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sqlite3
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pinn_model import PINNTrainer

# ================================
# 하이퍼파라미터
# ================================
EPISODES      = 300
MAX_STEPS     = 500
BATCH_SIZE    = 256
BUFFER_SIZE   = 50000
LR            = 3e-4
GAMMA         = 0.99
IMAGINARY_K   = 3
CONTEXT_BONUS = 2.0
SWITCH_BONUS  = 5.0  # 전환 시점 보너스

# ================================
# SwitchingEnv (swingup_switching.py에서 가져옴)
# ================================
import gymnasium as gym
from gymnasium import spaces

class SwitchingEnv(gym.Env):
    def __init__(self):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle = np.radians(30)
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state = np.array([0.0, 0.0, np.pi + np.random.uniform(-0.1, 0.1), 0.0],
                               dtype=np.float32)
        self.mode = "swingup"
        self.total_steps = 0
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action[0]) * self.force_mag
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole*self.length, np.cos(theta), np.sin(theta)
        tmp    = (force + ml*theta_dot**2*s) / mt
        th_acc = (self.gravity*s - c*tmp) / \
                 (self.length*(4/3 - self.mass_pole*c**2/mt))
        x_acc  = tmp - ml*th_acc*c/mt
        x += self.dt*x_dot;     x_dot     += self.dt*x_acc
        theta += self.dt*theta_dot; theta_dot += self.dt*th_acc
        theta  = ((theta+np.pi)%(2*np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.total_steps += 1

        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode = "balance"
            just_switched = True

        if self.mode == "swingup":
            reward = (np.cos(theta)
                      + 0.1*self.length*(1+np.cos(theta))
                      + (0.5 if abs(theta) < np.pi/2 else 0)
                      - 0.1*abs(x) - 0.001*theta_dot**2)
        else:
            reward = (np.cos(theta) - 0.1*abs(x)
                      - 0.1*theta_dot**2
                      + (1.0 if just_switched else 0))

        done = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, done, False, {"just_switched": just_switched}

# ================================
# Context DB
# ================================
def load_context_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos, cart_vel, pole_ang, pole_vel "
            "FROM switching_markings"
        )
        rows = cursor.fetchall()
        conn.close()
        if len(rows) > 0:
            return np.array(rows, dtype=np.float32)
    except:
        pass
    # switching_markings 없으면 기본 markings 사용
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos, cart_vel, pole_ang, pole_vel FROM markings"
        )
        rows = cursor.fetchall()
        conn.close()
        if len(rows) > 0:
            return np.array(rows, dtype=np.float32)
    except:
        pass
    return np.zeros((1, 4))

def context_reward(state, context_states, threshold=0.3):
    if len(context_states) == 0:
        return 0.0
    dists = np.linalg.norm(context_states - state, axis=1)
    if dists.min() < threshold:
        return CONTEXT_BONUS
    return 0.0

# ================================
# SAC 네트워크
# ================================
class Actor(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        raw     = dist.rsample()
        action  = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - \
                   torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)

# ================================
# Replay Buffer
# ================================
class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buf = deque(maxlen=size)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n=BATCH_SIZE):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self):
        return len(self.buf)

# ================================
# SAC 에이전트
# ================================
class SACAgent:
    def __init__(self):
        self.actor  = Actor()
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)
        self.log_alpha  = torch.zeros(1, requires_grad=True)
        self.alpha_opt  = optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -1.0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor(s)
        return a.squeeze(0).numpy()

    def update(self, buffer):
        if len(buffer) < BATCH_SIZE:
            return
        s, a, r, ns, d = buffer.sample()
        with torch.no_grad():
            na, log_p = self.actor(ns)
            q1t, q2t  = self.critic_target(ns, na)
            qt        = torch.min(q1t, q2t) - self.alpha*log_p
            target_q  = r + GAMMA*(1-d)*qt
        q1, q2 = self.critic(s, a)
        c_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()
        na, log_p  = self.actor(s)
        q1, q2     = self.critic(s, na)
        a_loss     = (self.alpha.detach()*log_p - torch.min(q1,q2)).mean()
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()
        al_loss = -(self.log_alpha*(log_p+self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); al_loss.backward(); self.alpha_opt.step()
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(0.005*p.data + 0.995*tp.data)

# ================================
# 학습 함수
# ================================
def train(use_pinn=True, use_context=True, episodes=EPISODES):
    env    = SwitchingEnv()
    agent  = SACAgent()
    buffer = ReplayBuffer()
    rewards, switch_counts = [], []

    # PINN 로드
    pinn = None
    if use_pinn:
        pinn = PINNTrainer()
        try:
            pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
            print("  PINN 모델 로드 완료")
        except:
            print("  PINN 모델 없음")
            use_pinn = False

    # Context DB 로드
    context_states = np.zeros((1, 4))
    if use_context:
        context_states = load_context_db(
            "/home/jrkim/cartpole_project/marking_data.db"
        )
        print(f"  Context DB: {len(context_states)}개 마킹")

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward  = 0
        switch_cnt = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            just_switched = info.get("just_switched", False)

            # 전환 시점 보너스 (Context 핵심)
            if just_switched:
                switch_cnt += 1
                if use_context:
                    reward += SWITCH_BONUS

            # Context DB 유사도 보너스
            if use_context:
                reward += context_reward(next_obs, context_states)

            buffer.push(obs, action, reward, next_obs, float(done))

            # PINN 가상 경험
            if use_pinn and pinn is not None and len(buffer.buf) > 10:
                for _ in range(IMAGINARY_K):
                    idx   = random.randint(0, len(buffer.buf)-1)
                    s_img = buffer.buf[idx][0]
                    a_img = agent.select_action(s_img)
                    ns_img = pinn.predict(s_img, a_img[0])
                    r_img  = float(np.cos(ns_img[2]))  # 간단한 보상
                    if use_context:
                        r_img += context_reward(ns_img, context_states)
                    buffer.push(s_img, a_img, r_img, ns_img, 0.0)

            agent.update(buffer)
            ep_reward += reward
            obs = next_obs
            if done:
                break

        rewards.append(ep_reward)
        switch_counts.append(switch_cnt)

        if (ep+1) % 50 == 0:
            avg_r = np.mean(rewards[-50:])
            avg_s = np.mean(switch_counts[-50:])
            print(f"  EP {ep+1:4d} | Reward: {avg_r:7.2f} | "
                  f"전환 성공: {avg_s:.2f}회/에피소드")

    env.close()
    return rewards, switch_counts

# ================================
# 메인: 3가지 비교
# ================================
if __name__ == "__main__":
    print("=" * 55)
    print("비교: SAC vs PINN+SAC vs PINN+Context+SAC")
    print("환경: 스윙업 → 카트폴 전환 구조")
    print("=" * 55)

    print("\n[1] 기본 SAC")
    r1, s1 = train(use_pinn=False, use_context=False)

    print("\n[2] PINN + SAC")
    r2, s2 = train(use_pinn=True, use_context=False)

    print("\n[3] PINN + Context DB + SAC")
    r3, s3 = train(use_pinn=True, use_context=True)

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(smooth(r1), label='SAC', color='gray')
    ax1.plot(smooth(r2), label='PINN+SAC', color='blue')
    ax1.plot(smooth(r3), label='PINN+Context+SAC', color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curve')
    ax1.legend()

    ax2.plot(smooth(s1), label='SAC', color='gray')
    ax2.plot(smooth(s2), label='PINN+SAC', color='blue')
    ax2.plot(smooth(s3), label='PINN+Context+SAC', color='red')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Switch Count')
    ax2.set_title('Switching Success')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('/home/jrkim/cartpole_project/pinn_comparison.png')
    print("\n그래프 저장: pinn_comparison.png")

    print("\n=== 최종 성능 (마지막 50 에피소드) ===")
    print(f"{'':25s} {'Reward':>10s} {'Switch':>10s}")
    print(f"{'SAC 기본':25s} {np.mean(r1[-50:]):10.2f} {np.mean(s1[-50:]):10.2f}")
    print(f"{'PINN + SAC':25s} {np.mean(r2[-50:]):10.2f} {np.mean(s2[-50:]):10.2f}")
    print(f"{'PINN + Context + SAC':25s} {np.mean(r3[-50:]):10.2f} {np.mean(s3[-50:]):10.2f}")
