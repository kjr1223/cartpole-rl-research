import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame

# ================================
# 물리 모델 (가상 경험 생성용)
# 실제 환경과 동일한 라그랑지안 동역학
# ================================
class PhysicsModel:
    def __init__(self):
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 0.5
        self.dt        = 0.02
        self.force_mag = 10.0
        self.x_limit   = 2.4
        self.switch_angle = np.radians(20)

    def step(self, state, action):
        """
        물리 모델로 다음 state 예측
        실제 환경 없이 가상 경험 생성
        """
        x, x_dot, theta, theta_dot = state
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
        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # reward 계산
        E_goal = self.mass_pole * self.gravity * self.length * 2
        pe = self.mass_pole * self.gravity * self.length * (1 - np.cos(theta))
        ke = 0.5 * self.mass_pole * (self.length * theta_dot)**2
        E_now = pe + ke

        just_switched = abs(theta) < self.switch_angle
        if just_switched:
            reward = np.cos(theta) - 0.1*abs(x) - 0.1*theta_dot**2 + 3.0
        else:
            reward = (- 0.5*abs(E_goal-E_now)
                     + np.cos(theta)
                     + 0.1*abs(theta_dot)
                     - 0.1*abs(x))

        terminated = bool(abs(x) > self.x_limit)
        return next_state, reward, terminated, just_switched

    def generate_virtual_experiences(self, state, actor, n=5):
        """
        현재 state에서 물리 모델로
        가상 경험 n개 생성
        다양한 action으로 시뮬레이션
        """
        experiences = []
        for _ in range(n):
            # 랜덤 action으로 가상 경험 생성
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0)
                a, _ = actor.get_action(st, with_logprob=True)
            action = a.squeeze().item()

            next_state, reward, terminated, just_switched = self.step(state, action)
            experiences.append((state, action, reward, next_state, float(terminated)))

        return experiences


# ================================
# 환경
# ================================
class SwingupEnv(gym.Env):
    def __init__(self, start_angle_deg=180, render_mode=None):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle    = np.radians(20)
        self.start_angle     = np.radians(start_angle_deg)
        self.start_angle_deg = start_angle_deg
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode
        self.screen = self.clock = None
        self.np_random = np.random.default_rng()
        pygame.init()

    def reset(self):
        start_theta = self.start_angle + self.np_random.uniform(-0.1, 0.1)
        self.state = np.array([0.0, 0.0, start_theta, 0.0], dtype=np.float32)
        self.mode        = "swingup"
        self.switch_step = None
        self.total_steps = 0
        return self.state.copy()

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
            self.mode        = "balance"
            self.switch_step = self.total_steps
            just_switched    = True

        E_goal = self.mass_pole * self.gravity * self.length * 2
        E_now  = self._energy()

        if self.mode == "swingup":
            reward = (- 0.5*abs(E_goal-E_now)
                     + np.cos(theta)
                     + 0.1*abs(theta_dot)
                     - 0.1*abs(x))
        else:
            reward = (np.cos(theta)
                     - 0.1*abs(x)
                     - 0.1*theta_dot**2
                     + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        if self.render_mode == "human": self._render()
        return self.state.copy(), reward, terminated, just_switched

    def _render(self):
        W, H, scale = 600, 450, 600/(2*self.x_limit*2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption(
                f"Model-based RL (시작:{self.start_angle_deg}도)")
            self.clock = pygame.time.Clock()
        self.screen.fill((255,245,245) if self.mode=="swingup" else (245,255,245))
        x, _, theta, _ = self.state
        cx, cy = int(x*scale+W/2), int(H*0.55)
        for sign in [1,-1]:
            ex = cx+int(80*np.sin(self.switch_angle*sign))
            ey = cy-int(80*np.cos(self.switch_angle*sign))
            pygame.draw.line(self.screen,(200,200,0),(cx,cy),(ex,ey),2)
        pygame.draw.line(self.screen,(0,0,0),(0,cy+20),(W,cy+20),2)
        pygame.draw.rect(self.screen,
            (70,130,180) if self.mode=="swingup" else (50,180,50),
            (cx-30,cy-10,60,20))
        pl = int(self.length*2*scale)
        px,py = cx+int(pl*np.sin(theta)), cy-int(pl*np.cos(theta))
        pygame.draw.line(self.screen,(220,60,60),(cx,cy),(px,py),6)
        pygame.draw.circle(self.screen,(220,60,60),(px,py),8)
        gx,gy,gw = 80,40,440
        pygame.draw.rect(self.screen,(200,200,200),(gx,gy,gw,15))
        pygame.draw.rect(self.screen,(100,200,100),
                        (gx,gy,int(gw*(theta+np.pi)/(2*np.pi)),15))
        for angle,color in [(0,(255,0,0)),
                            (self.switch_angle,(255,165,0)),
                            (-self.switch_angle,(255,165,0))]:
            sx = gx+int(gw*(angle+np.pi)/(2*np.pi))
            pygame.draw.line(self.screen,color,(sx,gy-5),(sx,gy+20),2)
        font  = pygame.font.SysFont(None,28)
        font2 = pygame.font.SysFont(None,22)
        mode_text  = f"SWING-UP" if self.mode=="swingup" else "BALANCE (PPO)"
        mode_color = (0,0,200) if self.mode=="swingup" else (0,150,0)
        self.screen.blit(font.render(
            f"Angle:{np.degrees(theta):.1f}  |  {mode_text}",
            True,mode_color),(10,10))
        self.screen.blit(font2.render(
            f"step={self.total_steps} | "
            f"switch={'done!(step='+str(self.switch_step)+')' if self.switch_step else 'waiting'}",
            True,(80,80,80)),(10,H-30))
        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen: pygame.quit(); self.screen=None


# ================================
# 마킹 데이터
# ================================
def load_markings():
    conn   = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT cart_pos,cart_vel,pole_ang,pole_vel FROM switching_markings")
    rows = cursor.fetchall()
    conn.close()
    print(f"마킹 데이터 {len(rows)}개 불러옴")
    return np.array(rows)

def get_bonus(state, markings, threshold=0.5, bonus=3.0):
    distances = np.linalg.norm(markings - state, axis=1)
    return bonus if np.min(distances) < threshold else 0.0


# ================================
# SAC 신경망
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
# 카트폴 PPO
# ================================
class BalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2), nn.Softmax(dim=-1))
    def get_action(self,x):
        return (Categorical(self.net(x)).sample().float()*2-1)

def train_balance():
    import os
    actor = BalanceActor()
    if os.path.exists("balance_ppo.pth"):
        actor.load_state_dict(torch.load("balance_ppo.pth", weights_only=True))
        print("카트폴 PPO 불러옴!")
        return actor
    env   = gym.make("CartPole-v1")
    vnet  = nn.Sequential(
        nn.Linear(4,64),nn.Tanh(),
        nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    p_opt = torch.optim.Adam(actor.parameters(),lr=3e-4)
    v_opt = torch.optim.Adam(vnet.parameters(), lr=3e-4)
    print("카트폴 PPO 학습 중...")
    for ep in range(200):
        states,actions,rews,lps=[],[],[],[]
        obs,_=env.reset(); done=False
        while not done:
            st=torch.FloatTensor(obs)
            dist=Categorical(actor.net(st)); action=dist.sample()
            no,r,term,trunc,_=env.step(action.item())
            done=term or trunc
            states.append(obs); actions.append(action.item())
            rews.append(r); lps.append(dist.log_prob(action).item())
            obs=no
        st=torch.FloatTensor(np.array(states))
        at=torch.LongTensor(actions); lt=torch.FloatTensor(lps)
        R,rets=0,[]
        for r in reversed(rews): R=r+0.99*R; rets.insert(0,R)
        rt=torch.FloatTensor(rets)
        rt=(rt-rt.mean())/(rt.std()+1e-8)
        adv=rt-vnet(st).squeeze().detach()
        for _ in range(3):
            nlp=Categorical(actor.net(st)).log_prob(at)
            ratio=torch.exp(nlp-lt)
            pl=-torch.min(ratio*adv,torch.clamp(ratio,0.8,1.2)*adv).mean()
            vl=F.mse_loss(vnet(st).squeeze(),rt)
            p_opt.zero_grad(); pl.backward(); p_opt.step()
            v_opt.zero_grad(); vl.backward(); v_opt.step()
        if (ep+1)%50==0: print(f"  PPO ep={ep+1}/200 완료")
    env.close()
    torch.save(actor.state_dict(),"balance_ppo.pth")
    print("카트폴 PPO 완료!")
    return actor


# ================================
# Model-based RL 학습
# Dyna 스타일: 실제 경험 + 가상 경험
# ================================
def train_model_based(use_marking=False, episodes=1000,
                      virtual_per_real=5):
    """
    virtual_per_real: 실제 경험 1개당 가상 경험 몇 개 생성
    """
    import os
    markings = load_markings() if use_marking else None
    label    = "마킹" if use_marking else "기본"

    physics  = PhysicsModel()  # 물리 모델
    actor    = SACActor()

    # 저장된 모델 있으면 불러오기 (Curriculum 결과 활용!)
    curriculum_file = "curriculum_marking.pth" if use_marking \
                      else "curriculum_baseline.pth"
    if os.path.exists(curriculum_file):
        actor.load_state_dict(torch.load(curriculum_file, weights_only=True))
        print(f"Curriculum 모델 불러옴: {curriculum_file}")
        print("→ Curriculum 학습 결과에서 이어서 학습!")

    c1,c2   = SACCritic(),SACCritic()
    tc1,tc2 = SACCritic(),SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(),lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),  lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),  lr=3e-4)
    log_alpha = torch.zeros(1,requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha],lr=3e-4)

    # 실제 경험 버퍼 + 가상 경험 버퍼 분리
    real_buf    = []  # 실제 환경 경험
    virtual_buf = []  # 물리 모델 가상 경험
    gamma, tau = 0.99, 0.005

    print(f"\n{'='*40}")
    print(f"=== {label} Model-based RL ===")
    print(f"실제 경험 1개 → 가상 경험 {virtual_per_real}개 생성")
    print(f"{'='*40}")

    # 180도에서 바로 시작!
    env = SwingupEnv(start_angle_deg=180)
    all_switched = []
    switch_count = 0

    for ep in range(episodes):
        obs=env.reset(); done=False; switched=False; total_r=0

        while not done:
            with torch.no_grad():
                a,_=actor.get_action(
                    torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
            av=a.squeeze().item()
            no,r,term,just_switched=env.step(av)
            done=term

            # 마킹 보너스
            if use_marking and env.mode=="swingup" and abs(obs[2])<np.radians(40):
                r+=get_bonus(obs,markings)
            if just_switched:
                switched=True; r+=5.0

            # 실제 경험 저장
            real_buf.append((obs,av,r,no,float(done)))
            if len(real_buf)>10000: real_buf.pop(0)

            # 물리 모델로 가상 경험 생성 (Dyna 핵심!)
            virtual_exps = physics.generate_virtual_experiences(
                obs, actor, n=virtual_per_real)
            for exp in virtual_exps:
                # 마킹 보너스 가상 경험에도 적용
                s,a_,r_,ns,d = exp
                if use_marking and abs(s[2])<np.radians(40):
                    r_ += get_bonus(s, markings) * 0.5  # 가상 경험엔 절반만
                virtual_buf.append((s,a_,r_,ns,d))
            if len(virtual_buf)>50000: virtual_buf.pop(0)

            # 실제 + 가상 경험 합쳐서 학습
            total_buf = real_buf + virtual_buf
            if len(total_buf)>=64:
                idx=np.random.choice(len(total_buf),64,replace=False)
                b=[total_buf[i] for i in idx]
                s  =torch.FloatTensor(np.array([x[0] for x in b]))
                a_ =torch.FloatTensor(np.array([x[1] for x in b])).unsqueeze(1)
                r_ =torch.FloatTensor(np.array([x[2] for x in b])).unsqueeze(1)
                ns =torch.FloatTensor(np.array([x[3] for x in b]))
                d_ =torch.FloatTensor(np.array([x[4] for x in b])).unsqueeze(1)
                with torch.no_grad():
                    na,nlp=actor.get_action(ns,with_logprob=True)
                    tv=r_+gamma*(1-d_)*(
                        torch.min(tc1(ns,na),tc2(ns,na))-log_alpha.exp()*nlp)
                c1_opt.zero_grad(); F.mse_loss(c1(s,a_),tv).backward(); c1_opt.step()
                c2_opt.zero_grad(); F.mse_loss(c2(s,a_),tv).backward(); c2_opt.step()
                na2,nlp2=actor.get_action(s,with_logprob=True)
                q=torch.min(c1(s,na2),c2(s,na2))
                al=(log_alpha.exp()*nlp2-q).mean()
                a_opt.zero_grad(); al.backward(); a_opt.step()
                al2=-(log_alpha*(nlp2+-1.0).detach()).mean()
                al_opt.zero_grad(); al2.backward(); al_opt.step()
                for p,tp in zip(c1.parameters(),tc1.parameters()):
                    tp.data.copy_(tau*p+(1-tau)*tp)
                for p,tp in zip(c2.parameters(),tc2.parameters()):
                    tp.data.copy_(tau*p+(1-tau)*tp)
            obs=no; total_r+=r

        if switched: switch_count+=1
        all_switched.append(1 if switched else 0)

        if (ep+1)%100==0:
            rate=sum(all_switched[-100:])/100*100
            print(f"ep={ep+1} | 전환성공률={rate:.0f}%/100 | "
                  f"실제버퍼={len(real_buf)} | 가상버퍼={len(virtual_buf)}")
            switch_count=0

    env.close()
    fname = f"model_based_{label}.pth"
    torch.save(actor.state_dict(),fname)
    print(f"\n저장: {fname}")
    print(f"최종 성공률: {sum(all_switched[-100:])/100*100:.1f}%")
    return actor, all_switched


# ================================
# 시각화
# ================================
def visualize(sac_actor, balance_actor, label="Model-based RL", episodes=5):
    pygame.init()
    env = SwingupEnv(start_angle_deg=180, render_mode="human")
    print(f"\n=== {label} 시각화 ===")
    for ep in range(episodes):
        obs=env.reset(); done=False; total_r=0; running=True
        while not done and running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
            with torch.no_grad():
                st=torch.FloatTensor(obs).unsqueeze(0)
                a=sac_actor.get_action(st) if env.mode=="swingup" \
                  else balance_actor.get_action(st)
            obs,r,term,just_switched=env.step(a.squeeze().item())
            done=term; total_r+=r
            if just_switched:
                print(f"  전환 성공! step={env.switch_step} | "
                      f"각도={np.degrees(env.state[2]):.1f}도")
        print(f"ep={ep+1} | reward={total_r:.1f} | "
              f"전환={'성공✅' if env.switch_step else '실패❌'}")
    env.close()


# ================================
# 그래프
# ================================
def plot_results(base_sw, mark_sw):
    fig, ax = plt.subplots(figsize=(12,5))
    fig.suptitle('Model-based RL: Baseline vs Marking',
                fontsize=14, fontweight='bold')
    w   = 50
    eps = range(1, len(base_sw)+1)
    br  = [sum(base_sw[max(0,i-w):i+1])/min(i+1,w)*100 for i in range(len(base_sw))]
    mr  = [sum(mark_sw[max(0,i-w):i+1])/min(i+1,w)*100 for i in range(len(mark_sw))]
    ax.plot(eps, br, color='steelblue', linewidth=2, label='기본 Model-based RL')
    ax.plot(eps, mr, color='tomato',    linewidth=2, label='마킹 Model-based RL')
    ax.axhline(y=59.3, color='gray', linestyle='--', alpha=0.7,
               label='Curriculum SAC 마킹 (59.3%)')
    ax.set_title('Switch Success Rate % (last 50 ep) - 180도에서 시작')
    ax.set_xlabel('Episode'); ax.set_ylabel('Success Rate (%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_based_comparison.png', dpi=150, bbox_inches='tight')
    print("그래프 저장: model_based_comparison.png")


# ================================
# 메인 실행
# ================================
if __name__=="__main__":
    balance_actor = train_balance()

    # Model-based RL 학습
    # Curriculum 결과에서 이어서 + 물리 모델 가상 경험 추가
    base_actor, base_sw = train_model_based(
        use_marking=False, episodes=1000, virtual_per_real=5)
    mark_actor, mark_sw = train_model_based(
        use_marking=True,  episodes=1000, virtual_per_real=5)

    # 시각화
    visualize(base_actor, balance_actor, label="기본 Model-based RL")
    visualize(mark_actor, balance_actor, label="마킹 Model-based RL")

    # 그래프
    plot_results(base_sw, mark_sw)

    print("\n===== 최종 요약 =====")
    print(f"기본 Model-based RL 성공률: {sum(base_sw[-100:])/100*100:.1f}%")
    print(f"마킹 Model-based RL 성공률: {sum(mark_sw[-100:])/100*100:.1f}%")
    print(f"비교 - Curriculum SAC 마킹: 59.3%")
