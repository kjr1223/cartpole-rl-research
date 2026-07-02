import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_model import PINNTrainer
import gymnasium as gym
from gymnasium import spaces

# ================================
# 하이퍼파라미터
# ================================
MAX_STEPS     = 500
BATCH_SIZE    = 256
BUFFER_SIZE   = 50000
LR            = 3e-4
GAMMA         = 0.99
IMAGINARY_K   = 10
SWITCH_BONUS  = 5.0
CONTEXT_BONUS = 2.0

# ================================
# 환경
# ================================
class SwitchingEnv(gym.Env):
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

    def reset(self, seed=None, options=None):
        self.state = np.array([
            0.0, 0.0,
            self.start_angle + self.np_random.uniform(-0.1, 0.1),
            0.0], dtype=np.float32)
        self.mode = "swingup"
        self.total_steps = 0
        return self.state.copy(), {}

    def _energy(self):
        _, _, theta, theta_dot = self.state
        pe = self.mass_pole * self.gravity * self.length * (1 - np.cos(theta))
        ke = 0.5 * self.mass_pole * (self.length * theta_dot)**2
        return pe + ke

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
        self.total_steps += 1

        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode = "balance"; just_switched = True

        E_goal = self.mass_pole * self.gravity * self.length * 2
        E_now  = self._energy()

        if self.mode == "swingup":
            reward = (-0.5*abs(E_goal-E_now)
                     + np.cos(theta)
                     + 0.1*abs(theta_dot)
                     - 0.1*abs(x))
        else:
            reward = (np.cos(theta) - 0.1*abs(x)
                     - 0.1*theta_dot**2
                     + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, terminated, False, {
            "just_switched": just_switched}


# ================================
# curriculum_sac.py와 동일한 SAC 구조
# curriculum_marking.pth 불러오기 위해
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
    def forward(self,s,a):
        return self.net(torch.cat([s,a],dim=-1))


# ================================
# Replay Buffer
# ================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        # action을 항상 1D array로 통일
        a = np.array([float(a)]) if np.isscalar(a) else np.array(a).flatten()[:1]
        self.buf.append((
            np.array(s, dtype=np.float32),
            a.astype(np.float32),
            float(r),
            np.array(ns, dtype=np.float32),
            float(d)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self): return len(self.buf)


# ================================
# SAC 업데이트
# ================================
class SACTrainer:
    def __init__(self, actor):
        self.actor   = actor
        self.critic1 = SACCritic()
        self.critic2 = SACCritic()
        self.tc1     = SACCritic()
        self.tc2     = SACCritic()
        self.tc1.load_state_dict(self.critic1.state_dict())
        self.tc2.load_state_dict(self.critic2.state_dict())
        self.a_opt  = optim.Adam(self.actor.parameters(),   lr=LR)
        self.c1_opt = optim.Adam(self.critic1.parameters(), lr=LR)
        self.c2_opt = optim.Adam(self.critic2.parameters(), lr=LR)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.al_opt    = optim.Adam([self.log_alpha], lr=LR)
        self.tau = 0.005

    def update(self, buffer):
        if len(buffer) < BATCH_SIZE: return
        s,a,r,ns,d = buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            na, nlp = self.actor.get_action(ns, with_logprob=True)
            tq  = torch.min(self.tc1(ns,na), self.tc2(ns,na))
            tv  = r + GAMMA*(1-d)*(tq - self.log_alpha.exp()*nlp)

        for opt, critic in [(self.c1_opt,self.critic1),
                            (self.c2_opt,self.critic2)]:
            loss = F.mse_loss(critic(s,a), tv)
            opt.zero_grad(); loss.backward(); opt.step()

        na2, nlp2 = self.actor.get_action(s, with_logprob=True)
        q2  = torch.min(self.critic1(s,na2), self.critic2(s,na2))
        al  = (self.log_alpha.exp()*nlp2 - q2).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step()

        al2 = -(self.log_alpha*(nlp2+-1.0).detach()).mean()
        self.al_opt.zero_grad(); al2.backward(); self.al_opt.step()

        for p,tp in zip(self.critic1.parameters(),self.tc1.parameters()):
            tp.data.copy_(self.tau*p+(1-self.tau)*tp)
        for p,tp in zip(self.critic2.parameters(),self.tc2.parameters()):
            tp.data.copy_(self.tau*p+(1-self.tau)*tp)


# ================================
# Context DB
# ================================
def load_context():
    try:
        conn   = sqlite3.connect("marking_data.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos,cart_vel,pole_ang,pole_vel "
            "FROM switching_markings")
        rows = cursor.fetchall()
        conn.close()
        if rows:
            print(f"  Context DB: {len(rows)}개 로드")
            return np.array(rows, dtype=np.float32)
    except: pass
    return None

def ctx_bonus(state, ctx):
    if ctx is None: return 0.0
    return CONTEXT_BONUS if np.linalg.norm(
        ctx - state, axis=1).min() < 0.5 else 0.0


# ================================
# 180도 PINN 파인튜닝
# curriculum_marking.pth에서 시작
# ================================
def finetune_180(model_path, use_pinn=True,
                 use_context=False, episodes=500, label=""):

    # 이전 성공 모델 불러오기
    actor = SACActor()
    actor.load_state_dict(
        torch.load(model_path, weights_only=True))
    print(f"  모델 불러옴: {model_path}")

    trainer = SACTrainer(actor)
    buffer  = ReplayBuffer(BUFFER_SIZE)
    context = load_context() if use_context else None

    # PINN 로드
    pinn = None
    if use_pinn:
        try:
            pinn = PINNTrainer()
            pinn.model.load_state_dict(
                torch.load("pinn_model.pth", weights_only=True))
            pinn.model.eval()
            print(f"  PINN 로드 완료!")
        except Exception as e:
            print(f"  PINN 로드 실패: {e}")

    env = SwitchingEnv(start_angle_deg=180)
    switched_list = []

    print(f"\n  180도 파인튜닝 ({episodes}에피소드)")
    print(f"  PINN: {use_pinn} | Context: {use_context}")
    print(f"  IMAGINARY_K: {IMAGINARY_K}")

    for ep in range(episodes):
        obs, _   = env.reset()
        switched = False

        for step in range(MAX_STEPS):
            with torch.no_grad():
                a = actor.get_action(
                    torch.FloatTensor(obs).unsqueeze(0))
            action = a.squeeze().item()

            next_obs, reward, done, _, info = env.step(action)
            just_switched = info.get("just_switched", False)

            if just_switched:
                switched = True
                reward  += SWITCH_BONUS

            if use_context:
                reward += ctx_bonus(next_obs, context)

            buffer.push(obs, action, reward, next_obs, float(done))

            # PINN 가상 경험 (핵심!)
            if use_pinn and pinn and len(buffer) > 10:
                for _ in range(IMAGINARY_K):
                    idx   = random.randint(0, len(buffer.buf)-1)
                    s_img = buffer.buf[idx][0]
                    with torch.no_grad():
                        a_img = actor.get_action(
                            torch.FloatTensor(s_img).unsqueeze(0))
                    a_val  = a_img.squeeze().item()
                    ns_img = pinn.predict(s_img, a_val)

                    # 에너지 기반 가상 reward
                    _, _, th, td = ns_img
                    E_goal = 0.1*9.8*0.5*2
                    E_now  = (0.1*9.8*0.5*(1-np.cos(th))
                             + 0.5*0.1*(0.5*td)**2)
                    r_img  = (-0.5*abs(E_goal-E_now)
                             + np.cos(th)
                             + 0.1*abs(td))
                    if use_context:
                        r_img += ctx_bonus(ns_img, context)

                    buffer.push(s_img, a_val, r_img, ns_img, 0.0)

            trainer.update(buffer)
            obs = next_obs
            if done: break

        switched_list.append(1 if switched else 0)

        if (ep+1) % 100 == 0:
            rate = np.mean(switched_list[-100:])*100
            print(f"  ep={ep+1} | 전환성공률={rate:.1f}%/100")

    # 저장
    fname = f"pinn_180_finetune_{label}.pth"
    torch.save(actor.state_dict(), fname)
    final_rate = np.mean(switched_list[-100:])*100
    print(f"  → 최종 성공률: {final_rate:.1f}% | 저장: {fname}")
    return switched_list, final_rate


# ================================
# 메인: 세 가지 비교
# ================================
if __name__ == "__main__":
    BASE_MODEL = "curriculum_marking.pth"

    print("="*55)
    print("180도 PINN 파인튜닝 (curriculum_marking.pth 기반)")
    print("="*55)

    print("\n[1] SAC only (파인튜닝)")
    sw1, r1 = finetune_180(
        BASE_MODEL, use_pinn=False,
        use_context=False, episodes=500, label="sac")

    print("\n[2] PINN + SAC (파인튜닝)")
    sw2, r2 = finetune_180(
        BASE_MODEL, use_pinn=True,
        use_context=False, episodes=500, label="pinn_sac")

    print("\n[3] PINN + Context + SAC (파인튜닝)")
    sw3, r3 = finetune_180(
        BASE_MODEL, use_pinn=True,
        use_context=True, episodes=500, label="pinn_ctx")

    # ================================
    # 결과 그래프
    # ================================
    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        '180° Swing-up: PINN Fine-tuning\n'
        '(Starting from curriculum_marking.pth)',
        fontsize=13, fontweight='bold')

    colors = {'sac':'#4C72B0', 'pinn':'#DD8452', 'ctx':'#55A868'}
    eps = range(len(smooth(sw1)))

    # 학습 곡선
    ax1 = axes[0]
    ax1.plot(smooth(sw1), color=colors['sac'],
            linewidth=2, label='SAC only')
    ax1.plot(smooth(sw2), color=colors['pinn'],
            linewidth=2, label='PINN + SAC')
    ax1.plot(smooth(sw3), color=colors['ctx'],
            linewidth=2, label='PINN + Context + SAC',
            linestyle='--')
    ax1.axhline(y=0.5, color='gray', linestyle=':',
               alpha=0.5, label='50% threshold')
    ax1.set_title('(a) Switch Success Rate (180°)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Switch Success Rate')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 최종 성공률 막대
    ax2 = axes[1]
    methods = ['SAC\n(finetune)', 'PINN+SAC\n(finetune)',
               'PINN+CTX\n(finetune)']
    rates   = [r1, r2, r3]
    bars = ax2.bar(methods, rates,
                  color=[colors['sac'], colors['pinn'], colors['ctx']],
                  alpha=0.85, edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,
                f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax2.axhline(y=59.3, color='red', linestyle='--',
               alpha=0.7, label='Previous best (59.3%)')
    ax2.set_title('(b) Final Success Rate at 180°')
    ax2.set_ylabel('Switch Success Rate (%)')
    ax2.set_ylim(0, 110)
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('pinn_180_result.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("\n그래프 저장: pinn_180_result.png")

    print("\n===== 최종 요약 =====")
    print(f"SAC only:          {r1:.1f}%")
    print(f"PINN + SAC:        {r2:.1f}%")
    print(f"PINN + CTX + SAC:  {r3:.1f}%")
    print(f"이전 최고 성능:     59.3% (curriculum_marking)")
