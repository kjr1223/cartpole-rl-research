import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sqlite3
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_model import PINNTrainer

# ================================
# 환경
# ================================
from gymnasium import spaces
import gymnasium as gym

class CurriculumEnv(gym.Env):
    def __init__(self, start_angle_deg=45):
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
        self.state  = np.array([0.0, 0.0, start_theta, 0.0], dtype=np.float32)
        self.mode   = "swingup"
        self.switch_step  = None
        self.total_steps  = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        mt    = self.mass_cart + self.mass_pole
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
            just_switched    = True
        E_goal = self.mass_pole * self.gravity * self.length * 2
        pe     = self.mass_pole * self.gravity * self.length * (1-np.cos(theta))
        ke     = 0.5 * self.mass_pole * (self.length*theta_dot)**2
        E_now  = pe + ke
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
# SAC 네트워크
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
        lp   = (dist.log_prob(xt) -
                torch.log(1-a.pow(2)+1e-6)).sum(-1, keepdim=True)
        return a, lp

class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,1))
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=-1))

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
        return np.array(rows, dtype=np.float32)
    except:
        return None

def get_bonus(state, markings, threshold=0.5, bonus=3.0):
    if markings is None: return 0.0
    return bonus if np.min(
        np.linalg.norm(markings-state, axis=1)) < threshold else 0.0

# ================================
# 단계별 학습
# ================================
def train_stage(actor, c1, c2, tc1, tc2,
                a_opt, c1_opt, c2_opt, log_alpha, al_opt,
                buf, angle, episodes,
                pinn=None, markings=None, label=""):

    env   = CurriculumEnv(start_angle_deg=angle)
    gamma, tau = 0.99, 0.005
    print(f"\n  [{angle}도] {episodes}ep | "
          f"PINN={'ON' if pinn else 'OFF'} | "
          f"Context={'ON' if markings is not None else 'OFF'}")
    all_sw = []

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

            if markings is not None:
                r += get_bonus(obs, markings)
            if just_switched:
                switched = True
                r += 5.0

            buf.append((obs, av, r, no, float(done)))
            if len(buf) > 20000: buf.pop(0)

            # PINN 가상 경험
            if pinn is not None and len(buf) > 10:
                for _ in range(1):
                    idx   = random.randint(0, len(buf)-1)
                    s_img = buf[idx][0]
                    with torch.no_grad():
                        a_img = actor.get_action(
                            torch.FloatTensor(s_img).unsqueeze(0)
                        ).squeeze().item()
                    ns_img = pinn.predict(s_img, a_img)
                    _, _, th, thd = ns_img
                    E_goal = 0.1*9.8*0.5*2
                    pe = 0.1*9.8*0.5*(1-np.cos(th))
                    ke = 0.5*0.1*(0.5*thd)**2
                    r_img = -0.5*abs(E_goal-(pe+ke)) + np.cos(th)
                    if markings is not None:
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
                a_opt.zero_grad()
                (log_alpha.exp()*nlp2 - q).mean().backward()
                a_opt.step()
                al_loss = -(log_alpha*(nlp2+-1.0).detach()).mean()
                al_opt.zero_grad()
                al_loss.backward()
                al_opt.step()
                for p, tp in zip(c1.parameters(), tc1.parameters()):
                    tp.data.copy_(tau*p + (1-tau)*tp)
                for p, tp in zip(c2.parameters(), tc2.parameters()):
                    tp.data.copy_(tau*p + (1-tau)*tp)
            obs = no

        all_sw.append(1 if switched else 0)
        if (ep+1) % 100 == 0:
            rate = sum(all_sw[-100:])/100*100
            print(f"    ep={ep+1:4d} | 전환성공률={rate:.0f}%")

    env.close()
    rate = sum(all_sw)/len(all_sw)*100
    print(f"  → {angle}도 완료: {rate:.1f}%")
    return all_sw

# ================================
# 전체 실험
# ================================
def run_experiment(use_pinn, use_marking, stages):
    actor = SACActor()
    c1, c2   = SACCritic(), SACCritic()
    tc1, tc2 = SACCritic(), SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),   lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),   lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha],     lr=3e-4)
    buf = []

    pinn = None
    if use_pinn:
        pinn = PINNTrainer()
        try:
            pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
        except:
            pinn = None

    markings = load_markings() if use_marking else None

    all_sw = []
    for angle, eps in stages:
        sw = train_stage(
            actor, c1, c2, tc1, tc2,
            a_opt, c1_opt, c2_opt, log_alpha, al_opt,
            buf, angle, eps,
            pinn=pinn, markings=markings)
        all_sw.extend(sw)
    return all_sw

# ================================
# 메인
# ================================
if __name__ == "__main__":
    STAGES = [(45, 300), (90, 300), (135, 400), (180, 500)]  # 45→90→135→180도

    print("="*50)
    print("빠른 비교: 45도 → 90도")
    print("="*50)

    print("\n[1] SAC 기본")
    r1 = run_experiment(use_pinn=False, use_marking=False, stages=STAGES)

    print("\n[2] PINN + SAC")
    r2 = run_experiment(use_pinn=True,  use_marking=False, stages=STAGES)

    print("\n[3] PINN + Context + SAC")
    r3 = run_experiment(use_pinn=True,  use_marking=True,  stages=STAGES)

    # 시각화
    def moving_avg(sw, w=30):
        return [sum(sw[max(0,i-w):i+1])/min(i+1,w)*100
                for i in range(len(sw))]

    plt.figure(figsize=(10, 5))
    eps = range(1, len(r1)+1)
    plt.plot(moving_avg(r1), color='gray', linewidth=2, label='SAC')
    plt.plot(moving_avg(r2), color='blue', linewidth=2, label='PINN+SAC')
    plt.plot(moving_avg(r3), color='red',  linewidth=2,
             label='PINN+Context+SAC')
    plt.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    plt.text(305, 5, '45→90', fontsize=9, color='gray')
    plt.axvline(x=600, color='gray', linestyle='--', alpha=0.5)
    plt.text(605, 5, '90→135', fontsize=9, color='gray')
    plt.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    plt.text(1005, 5, '135→180', fontsize=9, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Switch Success Rate (%)')
    plt.title('PINN + Context DB: 45→90→135→180deg Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        '/home/jrkim/cartpole_project/pinn_180_comparison.png',
        dpi=150)
    print("\n그래프 저장: pinn_180_comparison.png")

    print("\n===== 결과 요약 =====")
    print(f"{'':25s} {'45도':>8s} {'90도':>8s} {'135도':>8s}")
    for name, r in [('SAC 기본', r1),
                    ('PINN+SAC', r2),
                    ('PINN+Context+SAC', r3)]:
        s45  = sum(r[:300])/300*100
        s90  = sum(r[300:600])/300*100
        s135 = sum(r[600:])/400*100
        print(f"{name:25s} {s45:7.1f}%  {s90:7.1f}%  {s135:7.1f}%")
