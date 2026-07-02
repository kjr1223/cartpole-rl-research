import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
import sqlite3
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_model import PINNTrainer

# ================================
# 환경 (curriculum_sac.py와 동일)
# ================================
class CurriculumEnv(gym.Env):
    def __init__(self, start_angle_deg=180):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle    = np.radians(20)
        self.start_angle     = np.radians(start_angle_deg)
        self.start_angle_deg = start_angle_deg
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.np_random = np.random.default_rng()

    def reset(self):
        start_theta = self.start_angle + self.np_random.uniform(-0.1, 0.1)
        self.state = np.array([0.0, 0.0, start_theta, 0.0], dtype=np.float32)
        self.mode = "swingup"
        self.switch_step = None
        self.total_steps = 0
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
        x += self.dt*x_dot;     x_dot     += self.dt*x_acc
        theta += self.dt*theta_dot; theta_dot += self.dt*th_acc
        theta  = ((theta+np.pi)%(2*np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.total_steps += 1
        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode = "balance"
            self.switch_step = self.total_steps
            just_switched = True
        E_goal = self.mass_pole * self.gravity * self.length * 2
        pe = self.mass_pole * self.gravity * self.length * (1 - np.cos(theta))
        ke = 0.5 * self.mass_pole * (self.length * theta_dot)**2
        E_now = pe + ke
        if self.mode == "swingup":
            reward = (-0.5*abs(E_goal-E_now) + np.cos(theta)
                      + 0.1*abs(theta_dot) - 0.1*abs(x))
        else:
            reward = (np.cos(theta) - 0.1*abs(x)
                      - 0.1*theta_dot**2
                      + (3.0 if just_switched else 0.0))
        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, terminated, just_switched

# ================================
# SAC 네트워크 (curriculum_sac.py와 동일)
# ================================
class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU())
        self.mu      = nn.Linear(128,1)
        self.log_std = nn.Linear(128,1)

    def get_action(self, x, with_logprob=False):
        f    = self.net(x)
        dist = Normal(self.mu(f), self.log_std(f).clamp(-20,2).exp())
        xt   = dist.rsample()
        a    = torch.tanh(xt)
        if not with_logprob: return a
        lp = (dist.log_prob(xt)-torch.log(1-a.pow(2)+1e-6)).sum(-1,keepdim=True)
        return a, lp

class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,1))
    def forward(self, s, a):
        return self.net(torch.cat([s,a],dim=-1))

# ================================
# Context DB
# ================================
def load_markings():
    try:
        conn   = sqlite3.connect(
            "/home/jrkim/cartpole_project/marking_data.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos,cart_vel,pole_ang,pole_vel "
            "FROM switching_markings")
        rows = cursor.fetchall()
        conn.close()
        if len(rows) > 0:
            print(f"switching_markings: {len(rows)}개")
            return np.array(rows, dtype=np.float32)
    except: pass
    try:
        conn   = sqlite3.connect(
            "/home/jrkim/cartpole_project/marking_data.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos,cart_vel,pole_ang,pole_vel FROM markings")
        rows = cursor.fetchall()
        conn.close()
        print(f"markings: {len(rows)}개")
        return np.array(rows, dtype=np.float32)
    except:
        return np.zeros((1,4))

def get_bonus(state, markings, threshold=0.5, bonus=3.0):
    if markings is None: return 0.0
    distances = np.linalg.norm(markings - state, axis=1)
    return bonus if np.min(distances) < threshold else 0.0

# ================================
# 파인튜닝: 기존 모델 불러와서
# PINN 가상 경험으로 추가 학습
# ================================
def finetune(model_path, label, use_pinn=True,
             use_marking=False, episodes=1000):
    print(f"\n{'='*45}")
    print(f"파인튜닝: {label}")
    print(f"  모델: {model_path}")
    print(f"  PINN: {use_pinn} | Context: {use_marking}")
    print(f"{'='*45}")

    # 모델 로드
    actor = SACActor()
    actor.load_state_dict(
        torch.load(model_path, weights_only=True))
    c1, c2   = SACCritic(), SACCritic()
    tc1, tc2 = SACCritic(), SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(), lr=1e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),   lr=1e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),   lr=1e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha], lr=1e-4)
    buf = []

    # PINN 로드
    pinn = None
    if use_pinn:
        pinn = PINNTrainer()
        try:
            pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
            print("  PINN 로드 완료")
        except:
            print("  PINN 없음")
            pinn = None

    # Context DB 로드
    markings = load_markings() if use_marking else None

    env = CurriculumEnv(start_angle_deg=180)
    gamma, tau = 0.99, 0.005
    all_switched = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        switched = False

        while not done:
            with torch.no_grad():
                a, _ = actor.get_action(
                    torch.FloatTensor(obs).unsqueeze(0),
                    with_logprob=True)
            av = a.squeeze().item()
            no, r, term, just_switched = env.step(av)
            done = term

            # Context 보너스
            if use_marking and markings is not None:
                r += get_bonus(obs, markings)
            if just_switched:
                switched = True
                r += 5.0

            buf.append((obs, av, r, no, float(done)))
            if len(buf) > 20000: buf.pop(0)

            # PINN 가상 경험
            if pinn is not None and len(buf) > 10:
                for _ in range(3):
                    idx   = random.randint(0, len(buf)-1)
                    s_img = buf[idx][0]
                    with torch.no_grad():
                        a_img = actor.get_action(
                            torch.FloatTensor(s_img).unsqueeze(0)
                        ).squeeze().item()
                    ns_img = pinn.predict(s_img, a_img)
                    r_img  = float(-0.5 * abs(
                        self_energy(ns_img) -
                        self_energy_goal()) + np.cos(ns_img[2]))
                    if use_marking and markings is not None:
                        r_img += get_bonus(ns_img, markings)
                    buf.append((s_img, a_img, r_img, ns_img, 0.0))
                    if len(buf) > 20000: buf.pop(0)

            # SAC 업데이트
            if len(buf) >= 64:
                idx = np.random.choice(len(buf), 64, replace=False)
                b   = [buf[i] for i in idx]
                s_  = torch.FloatTensor(np.array([x[0] for x in b]))
                a_  = torch.FloatTensor(np.array([x[1] for x in b])).unsqueeze(1)
                r_  = torch.FloatTensor(np.array([x[2] for x in b])).unsqueeze(1)
                ns_ = torch.FloatTensor(np.array([x[3] for x in b]))
                d_  = torch.FloatTensor(np.array([x[4] for x in b])).unsqueeze(1)
                with torch.no_grad():
                    na, nlp = actor.get_action(ns_, with_logprob=True)
                    tv = r_ + gamma*(1-d_)*(
                        torch.min(tc1(ns_,na), tc2(ns_,na))
                        - log_alpha.exp()*nlp)
                c1_opt.zero_grad()
                F.mse_loss(c1(s_,a_), tv).backward()
                c1_opt.step()
                c2_opt.zero_grad()
                F.mse_loss(c2(s_,a_), tv).backward()
                c2_opt.step()
                na2, nlp2 = actor.get_action(s_, with_logprob=True)
                q = torch.min(c1(s_,na2), c2(s_,na2))
                a_loss = (log_alpha.exp()*nlp2 - q).mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                al_loss = -(log_alpha*(nlp2+-1.0).detach()).mean()
                al_opt.zero_grad(); al_loss.backward(); al_opt.step()
                for p, tp in zip(c1.parameters(), tc1.parameters()):
                    tp.data.copy_(tau*p + (1-tau)*tp)
                for p, tp in zip(c2.parameters(), tc2.parameters()):
                    tp.data.copy_(tau*p + (1-tau)*tp)
            obs = no

        all_switched.append(1 if switched else 0)
        if (ep+1) % 100 == 0:
            rate = sum(all_switched[-100:])/100*100
            print(f"  ep={ep+1:4d} | 전환성공률={rate:.0f}%/100")

    env.close()
    return all_switched

def self_energy(state):
    mass_pole, gravity, length = 0.1, 9.8, 0.5
    _, _, theta, theta_dot = state
    pe = mass_pole * gravity * length * (1 - np.cos(theta))
    ke = 0.5 * mass_pole * (length * theta_dot)**2
    return pe + ke

def self_energy_goal():
    return 0.1 * 9.8 * 0.5 * 2

# ================================
# 메인: 3가지 비교
# ================================
if __name__ == "__main__":
    BASE  = "/home/jrkim/cartpole_project/curriculum_baseline.pth"
    MARK  = "/home/jrkim/cartpole_project/curriculum_marking.pth"
    EPS   = 500

    print("기존 학습된 모델 불러와서 180도 파인튜닝")
    print("빠르게 PINN 효과 비교\n")

    # 1. 기본 SAC (PINN X, Context X)
    r1 = finetune(BASE,  "SAC 기본",
                  use_pinn=False, use_marking=False, episodes=EPS)

    # 2. PINN + SAC (Context X)
    r2 = finetune(BASE,  "PINN + SAC",
                  use_pinn=True,  use_marking=False, episodes=EPS)

    # 3. PINN + Context + SAC
    r3 = finetune(MARK,  "PINN + Context + SAC",
                  use_pinn=True,  use_marking=True,  episodes=EPS)

    # 결과 시각화
    def moving_avg(sw, w=50):
        return [sum(sw[max(0,i-w):i+1])/min(i+1,w)*100
                for i in range(len(sw))]

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg(r1), color='gray', linewidth=2,
             label='SAC (기본)')
    plt.plot(moving_avg(r2), color='blue', linewidth=2,
             label='PINN + SAC')
    plt.plot(moving_avg(r3), color='red',  linewidth=2,
             label='PINN + Context + SAC')
    plt.xlabel('Episode')
    plt.ylabel('Switch Success Rate (%)')
    plt.title('PINN + Context DB Effect (180deg Finetuning)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        '/home/jrkim/cartpole_project/pinn_finetune_comparison.png',
        dpi=150)
    print("\n그래프 저장: pinn_finetune_comparison.png")

    print("\n===== 최종 결과 (마지막 100 에피소드) =====")
    print(f"{'SAC 기본':25s} {sum(r1[-100:])}%")
    print(f"{'PINN + SAC':25s} {sum(r2[-100:])}%")
    print(f"{'PINN + Context + SAC':25s} {sum(r3[-100:])}%")
