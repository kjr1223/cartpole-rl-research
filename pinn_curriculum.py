import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
IMAGINARY_K   = 10   # 3 → 10으로 증가 (핵심 개선!)
CONTEXT_BONUS = 2.0
SWITCH_BONUS  = 5.0

# Curriculum 단계별 설정
# (시작각도, 에피소드 수)
CURRICULUM_STAGES = [
    (45,  200),
    (90,  200),
    (135, 300),
    (150, 300),  # 새로 추가!
    (165, 300),  # 새로 추가!
    (180, 400),
]

# ================================
# 환경: 시작 각도 조절 가능
# ================================
class SwitchingEnv(gym.Env):
    def __init__(self, start_angle_deg=180):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle   = np.radians(20)
        self.start_angle    = np.radians(start_angle_deg)
        self.start_angle_deg = start_angle_deg
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 시작 각도 근처에서 랜덤 시작
        self.state = np.array([
            0.0, 0.0,
            self.start_angle + np.random.uniform(-0.1, 0.1),
            0.0], dtype=np.float32)
        self.mode        = "swingup"
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

        # 전환 감지
        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode     = "balance"
            just_switched = True

        # 에너지 기반 reward (개선!)
        E_goal = self.mass_pole * self.gravity * self.length * 2
        E_now  = self._energy()

        if self.mode == "swingup":
            reward = (-0.5*abs(E_goal-E_now)
                     + np.cos(theta)
                     + 0.1*abs(theta_dot)
                     - 0.1*abs(x))
        else:
            reward = (np.cos(theta)
                     - 0.1*abs(x)
                     - 0.1*theta_dot**2
                     + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), reward, terminated, False, {
            "just_switched": just_switched}


# ================================
# Context DB 로드
# ================================
def load_context_states():
    try:
        conn   = sqlite3.connect(
            "/home/jrkim/cartpole_project/marking_data.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cart_pos,cart_vel,pole_ang,pole_vel "
            "FROM switching_markings")
        rows = cursor.fetchall()
        conn.close()
        if rows:
            print(f"  Context DB: {len(rows)}개 로드")
            return np.array(rows, dtype=np.float32)
    except Exception as e:
        print(f"  Context DB 없음: {e}")
    return None

def context_reward(state, context_states, threshold=0.5):
    if context_states is None: return 0.0
    dists = np.linalg.norm(context_states - state, axis=1)
    return CONTEXT_BONUS if dists.min() < threshold else 0.0


# ================================
# SAC Agent (기존 pinn_rl.py와 동일)
# ================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        return map(lambda x: torch.FloatTensor(np.array(x)), zip(*batch))
    def __len__(self): return len(self.buf)

class SACNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

class SACAgent:
    def __init__(self, state_dim=4, action_dim=1, hidden=256):
        self.actor   = SACNet(state_dim, action_dim*2, hidden)
        self.critic1 = SACNet(state_dim+action_dim, 1, hidden)
        self.critic2 = SACNet(state_dim+action_dim, 1, hidden)
        self.tc1     = SACNet(state_dim+action_dim, 1, hidden)
        self.tc2     = SACNet(state_dim+action_dim, 1, hidden)
        self.tc1.load_state_dict(self.critic1.state_dict())
        self.tc2.load_state_dict(self.critic2.state_dict())
        self.a_opt  = optim.Adam(self.actor.parameters(),   lr=LR)
        self.c1_opt = optim.Adam(self.critic1.parameters(), lr=LR)
        self.c2_opt = optim.Adam(self.critic2.parameters(), lr=LR)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.al_opt    = optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -action_dim
        self.tau = 0.005

    def select_action(self, state):
        with torch.no_grad():
            s  = torch.FloatTensor(state).unsqueeze(0)
            out = self.actor(s)
            mu, log_std = out.chunk(2, dim=-1)
            std    = log_std.clamp(-20, 2).exp()
            action = torch.tanh(mu + std * torch.randn_like(std))
        return action.squeeze().numpy()

    def update(self, buffer):
        if len(buffer) < BATCH_SIZE: return
        s,a,r,ns,d = buffer.sample(BATCH_SIZE)
        a = a.unsqueeze(1) if a.dim() == 1 else a
        r = r.unsqueeze(1); d = d.unsqueeze(1)

        with torch.no_grad():
            out    = self.actor(ns)
            mu,lst = out.chunk(2,dim=-1)
            std    = lst.clamp(-20,2).exp()
            na     = torch.tanh(mu + std*torch.randn_like(std))
            nlp    = (-0.5*((na-mu)/std)**2 - lst
                      - 0.5*np.log(2*np.pi)
                      - torch.log(1-na**2+1e-6)).sum(-1,keepdim=True)
            tq  = torch.min(
                self.tc1(torch.cat([ns,na],-1)),
                self.tc2(torch.cat([ns,na],-1)))
            tv  = r + GAMMA*(1-d)*(tq - self.log_alpha.exp()*nlp)

        for opt, critic in [(self.c1_opt,self.critic1),
                            (self.c2_opt,self.critic2)]:
            loss = nn.MSELoss()(critic(torch.cat([s,a],-1)), tv)
            opt.zero_grad(); loss.backward(); opt.step()

        out2   = self.actor(s)
        mu2,ls2 = out2.chunk(2,dim=-1)
        std2   = ls2.clamp(-20,2).exp()
        na2    = torch.tanh(mu2 + std2*torch.randn_like(std2))
        lp2    = (-0.5*((na2-mu2)/std2)**2 - ls2
                  - 0.5*np.log(2*np.pi)
                  - torch.log(1-na2**2+1e-6)).sum(-1,keepdim=True)
        q2 = torch.min(
            self.critic1(torch.cat([s,na2],-1)),
            self.critic2(torch.cat([s,na2],-1)))
        al = (self.log_alpha.exp()*lp2 - q2).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step()

        al2 = -(self.log_alpha*(lp2+self.target_entropy).detach()).mean()
        self.al_opt.zero_grad(); al2.backward(); self.al_opt.step()

        for p,tp in zip(self.critic1.parameters(),self.tc1.parameters()):
            tp.data.copy_(self.tau*p+(1-self.tau)*tp)
        for p,tp in zip(self.critic2.parameters(),self.tc2.parameters()):
            tp.data.copy_(self.tau*p+(1-self.tau)*tp)


# ================================
# Curriculum 학습 함수
# 단계별로 시작 각도 높여가며 학습
# ================================
def train_curriculum(use_pinn=True, use_context=False, label=""):
    import os

    context_states = load_context_states() if use_context else None

    # PINN 로드
    pinn = None
    if use_pinn:
        try:
            pinn = PINNTrainer()
            pinn.model.load_state_dict(
                torch.load("/home/jrkim/cartpole_project/pinn_model.pth",
                          weights_only=True))
            pinn.model.eval()
            print(f"  PINN 모델 로드 완료!")
        except Exception as e:
            print(f"  PINN 로드 실패: {e}")
            pinn = None

    # Agent와 Buffer는 단계 간 공유 (이전 지식 유지!)
    agent  = SACAgent()
    buffer = ReplayBuffer(BUFFER_SIZE)

    all_switched  = []  # 전체 전환 성공 기록
    stage_results = {}  # 단계별 결과

    print(f"\n{'='*50}")
    print(f"  {label} Curriculum Learning 시작")
    print(f"  PINN: {use_pinn} | Context: {use_context}")
    print(f"  IMAGINARY_K: {IMAGINARY_K}")
    print(f"{'='*50}")

    for stage_idx, (angle_deg, episodes) in enumerate(CURRICULUM_STAGES):
        # 저장된 단계 있으면 건너뜀
        fname = f"/home/jrkim/cartpole_project/pinn_curr_{angle_deg}_{label}.pth"
        if os.path.exists(fname):
            agent.actor.load_state_dict(
                torch.load(fname, weights_only=True))
            print(f"\n[{stage_idx+1}단계 {angle_deg}도] 불러옴 → 건너뜀")
            all_switched.extend([0]*episodes)
            stage_results[angle_deg] = 0.0
            continue

        env = SwitchingEnv(start_angle_deg=angle_deg)
        switched_list = []

        print(f"\n[{stage_idx+1}단계] 시작각도={angle_deg}도 | {episodes}에피소드")

        for ep in range(episodes):
            obs, _     = env.reset()
            ep_reward  = 0
            switched   = False

            for step in range(MAX_STEPS):
                action = agent.select_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                just_switched = info.get("just_switched", False)

                if just_switched:
                    switched = True
                    reward  += SWITCH_BONUS

                if use_context and context_states is not None:
                    reward += context_reward(next_obs, context_states)

                buffer.push(obs, np.array([float(action)]) if not isinstance(action, np.ndarray) else action.reshape(1), reward, next_obs, float(done))

                # PINN 가상 경험 생성 (핵심!)
                if use_pinn and pinn is not None and len(buffer) > 10:
                    for _ in range(IMAGINARY_K):
                        idx    = random.randint(0, len(buffer.buf)-1)
                        s_img  = buffer.buf[idx][0]
                        a_img  = agent.select_action(s_img)

                        ns_img = pinn.predict(s_img, float(a_img))

                        # 개선된 가상 경험 reward
                        _, _, th_img, td_img = ns_img
                        E_goal = 0.1*9.8*0.5*2
                        pe = 0.1*9.8*0.5*(1-np.cos(th_img))
                        ke = 0.5*0.1*(0.5*td_img)**2
                        E_now = pe + ke
                        r_img = (-0.5*abs(E_goal-E_now)
                                + np.cos(th_img)
                                + 0.1*abs(td_img))

                        if use_context and context_states is not None:
                            r_img += context_reward(ns_img, context_states)

                        buffer.push(s_img, np.array([float(a_img)]), r_img, ns_img, 0.0)

                agent.update(buffer)
                ep_reward += reward
                obs = next_obs
                if done: break

            switched_list.append(1 if switched else 0)
            all_switched.append(1 if switched else 0)

            if (ep+1) % 100 == 0:
                rate = np.mean(switched_list[-100:])*100
                print(f"  ep={ep+1} | 전환성공률={rate:.1f}%/100")

        env.close()

        # 단계별 모델 저장
        torch.save(agent.actor.state_dict(), fname)
        rate = np.mean(switched_list)*100
        stage_results[angle_deg] = rate
        print(f"  → [{angle_deg}도] 최종 성공률: {rate:.1f}% | 저장: {fname}")

    # 최종 모델 저장
    final_fname = f"/home/jrkim/cartpole_project/pinn_curriculum_{label}.pth"
    torch.save(agent.actor.state_dict(), final_fname)
    print(f"\n최종 저장: {final_fname}")

    return all_switched, stage_results


# ================================
# 메인 실행
# ================================
if __name__ == "__main__":
    print("=" * 55)
    print("PINN + Curriculum Learning 비교 실험")
    print("단계: 45 → 90 → 135 → 150 → 165 → 180도")
    print(f"IMAGINARY_K: {IMAGINARY_K} (가상 경험 {IMAGINARY_K}개/스텝)")
    print("=" * 55)

    # 1. 기본 SAC + Curriculum
    print("\n[1] 기본 SAC + Curriculum")
    sw1, res1 = train_curriculum(
        use_pinn=False, use_context=False, label="sac")

    # 2. PINN + SAC + Curriculum
    print("\n[2] PINN + SAC + Curriculum")
    sw2, res2 = train_curriculum(
        use_pinn=True, use_context=False, label="pinn_sac")

    # 3. PINN + Context + SAC + Curriculum
    print("\n[3] PINN + Context + SAC + Curriculum")
    sw3, res3 = train_curriculum(
        use_pinn=True, use_context=True, label="pinn_ctx")

    # ================================
    # 결과 출력
    # ================================
    stages = [s[0] for s in CURRICULUM_STAGES]
    print("\n" + "="*55)
    print("단계별 전환 성공률 비교")
    print(f"{'각도':>6} | {'SAC':>8} | {'PINN+SAC':>10} | {'PINN+CTX':>10}")
    print("-"*45)
    for angle in stages:
        r1 = res1.get(angle, 0)
        r2 = res2.get(angle, 0)
        r3 = res3.get(angle, 0)
        print(f"{angle:>5}도 | {r1:>7.1f}% | {r2:>9.1f}% | {r3:>9.1f}%")

    # ================================
    # 그래프
    # ================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Curriculum PINN: Switch Success Rate\n'
        '(45→90→135→150→165→180°)',
        fontsize=13, fontweight='bold')

    colors = {'sac':'#4C72B0', 'pinn':'#DD8452', 'ctx':'#55A868'}

    # 막대 그래프
    ax1 = axes[0]
    x   = np.arange(len(stages))
    w   = 0.25
    b1  = ax1.bar(x-w,  [res1.get(a,0) for a in stages],
                 w, label='SAC',          color=colors['sac'],  alpha=0.85)
    b2  = ax1.bar(x,    [res2.get(a,0) for a in stages],
                 w, label='PINN+SAC',     color=colors['pinn'], alpha=0.85)
    b3  = ax1.bar(x+w,  [res3.get(a,0) for a in stages],
                 w, label='PINN+CTX+SAC', color=colors['ctx'],  alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax1.text(bar.get_x()+bar.get_width()/2., h+1,
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_title('(a) Switch Success Rate by Stage')
    ax1.set_xlabel('Start Angle (Curriculum Stage)')
    ax1.set_ylabel('Switch Success Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{a}°' for a in stages])
    ax1.set_ylim(0, 115)
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    # 선 그래프
    ax2 = axes[1]
    s_labels = [f'{a}°' for a in stages]
    ax2.plot(s_labels, [res1.get(a,0) for a in stages],
            'o-', color=colors['sac'],  linewidth=2.5,
            markersize=8, label='SAC')
    ax2.plot(s_labels, [res2.get(a,0) for a in stages],
            's-', color=colors['pinn'], linewidth=2.5,
            markersize=8, label='PINN+SAC')
    ax2.plot(s_labels, [res3.get(a,0) for a in stages],
            '^--', color=colors['ctx'], linewidth=2.5,
            markersize=8, label='PINN+CTX+SAC')
    ax2.axhline(y=50, color='gray', linestyle=':',
               alpha=0.5, label='50% threshold')
    ax2.set_title('(b) Switch Success Rate Trend')
    ax2.set_xlabel('Curriculum Stage')
    ax2.set_ylabel('Switch Success Rate (%)')
    ax2.set_ylim(0, 115)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = '/home/jrkim/cartpole_project/pinn_curriculum_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n그래프 저장: {out}")
