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
# 환경: Residual RL 지원
# ================================
class ResidualEnv(gym.Env):
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
        self.base_actions   = []  # 물리 제어기 action 기록
        self.residual_actions = []  # SAC 보정값 기록
        return self.state.copy()

    def _energy(self):
        # 막대의 총 역학적 에너지 (위치에너지 + 운동에너지)
        _, _, theta, theta_dot = self.state
        pe = self.mass_pole * self.gravity * self.length * (1 - np.cos(theta))
        ke = 0.5 * self.mass_pole * (self.length * theta_dot)**2
        return pe + ke

    def physics_controller(self):
        """
        에너지 기반 스윙업 물리 제어기
        E_error = 목표에너지 - 현재에너지
        force = k * E_error * cos(theta) * theta_dot
        에너지가 부족하면 밀어서 채우고
        충분하면 멈추는 원리
        """
        _, _, theta, theta_dot = self.state
        E_goal = self.mass_pole * self.gravity * self.length * 2
        E_now  = self._energy()
        E_error = E_goal - E_now

        # 물리 기반 제어 입력 계산
        k = 2.0  # 제어 게인
        force = k * E_error * np.cos(theta) * theta_dot

        # -1 ~ +1 범위로 정규화
        base_action = np.clip(force / self.force_mag, -1.0, 1.0)
        return float(base_action)

    def step(self, residual_action):
        """
        residual_action: SAC가 출력한 보정값 (-1 ~ +1)
        최종 action = 물리 제어기 + SAC 보정값
        """
        # 물리 제어기 base action 계산
        base_action = self.physics_controller()

        # 최종 action = 물리 제어기 + SAC 보정 (가중치 조절)
        # base: 물리가 70% 담당
        # residual: SAC가 30% 보정
        final_action = np.clip(
            0.7 * base_action + 0.3 * float(residual_action),
            -1.0, 1.0)

        self.base_actions.append(base_action)
        self.residual_actions.append(float(residual_action))

        # 동역학 계산 (라그랑지안)
        x, x_dot, theta, theta_dot = self.state
        force = final_action * self.force_mag
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
            self.mode        = "balance"
            self.switch_step = self.total_steps
            just_switched    = True

        # Physics-informed reward
        E_goal = self.mass_pole * self.gravity * self.length * 2
        E_now  = self._energy()

        if self.mode == "swingup":
            reward = (- 0.5 * abs(E_goal - E_now)  # 에너지 오차 패널티
                     + np.cos(theta)                 # 위치 보상
                     + 0.1 * abs(theta_dot)          # 움직임 장려
                     - 0.1 * abs(x)                  # 카트 중앙 유지
                     - 0.05 * residual_action**2)    # 보정값 최소화 (물리 제어기 신뢰)
        else:
            reward = (np.cos(theta)
                     - 0.1 * abs(x)
                     - 0.1 * theta_dot**2
                     + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        if self.render_mode == "human": self._render(base_action, float(residual_action))
        return self.state.copy(), reward, terminated, just_switched

    def _render(self, base_action=0.0, residual_action=0.0):
        W, H, scale = 600, 500, 600/(2*self.x_limit*2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Residual RL (물리 + SAC 보정)")
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

        # 각도 게이지
        gx,gy,gw = 80,40,440
        pygame.draw.rect(self.screen,(200,200,200),(gx,gy,gw,15))
        pygame.draw.rect(self.screen,(100,200,100),
                        (gx,gy,int(gw*(theta+np.pi)/(2*np.pi)),15))
        for angle,color in [(0,(255,0,0)),
                            (self.switch_angle,(255,165,0)),
                            (-self.switch_angle,(255,165,0))]:
            sx = gx+int(gw*(angle+np.pi)/(2*np.pi))
            pygame.draw.line(self.screen,color,(sx,gy-5),(sx,gy+20),2)

        # 물리 제어기 게이지 (파란색)
        gy2 = 75
        pygame.draw.rect(self.screen,(200,200,200),(gx,gy2,gw,12))
        mid = gx+gw//2
        bw  = int(abs(base_action)*gw/2)
        col = (0,100,255) if base_action > 0 else (0,50,200)
        if base_action > 0: pygame.draw.rect(self.screen,col,(mid,gy2,bw,12))
        else:               pygame.draw.rect(self.screen,col,(mid-bw,gy2,bw,12))
        pygame.draw.line(self.screen,(0,0,0),(mid,gy2-3),(mid,gy2+15),2)

        # SAC 보정값 게이지 (빨간색)
        gy3 = 100
        pygame.draw.rect(self.screen,(200,200,200),(gx,gy3,gw,12))
        rw  = int(abs(residual_action)*gw/2)
        col2 = (255,100,0) if residual_action > 0 else (200,50,0)
        if residual_action > 0: pygame.draw.rect(self.screen,col2,(mid,gy3,rw,12))
        else:                   pygame.draw.rect(self.screen,col2,(mid-rw,gy3,rw,12))
        pygame.draw.line(self.screen,(0,0,0),(mid,gy3-3),(mid,gy3+15),2)

        font  = pygame.font.SysFont(None,28)
        font2 = pygame.font.SysFont(None,22)
        mode_text  = "SWING-UP (물리+SAC)" if self.mode=="swingup" else "BALANCE (PPO)"
        mode_color = (0,0,200) if self.mode=="swingup" else (0,150,0)
        self.screen.blit(font.render(
            f"Angle:{np.degrees(theta):.1f}  |  {mode_text}",
            True,mode_color),(10,10))
        self.screen.blit(font2.render(
            f"Physics(blue):{base_action:.3f}  SAC residual(red):{residual_action:.3f}",
            True,(80,80,80)),(gx,gy2-18))
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
# SAC 신경망 (보정값 출력)
# ================================
class ResidualActor(nn.Module):
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
        a    = torch.tanh(xt)  # -1 ~ +1 보정값
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

def train_balance(episodes=200):
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
    for ep in range(episodes):
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
# Residual RL 학습
# Curriculum Learning 적용
# ================================
def train_residual(use_marking=False, episodes_per_stage=500):
    import os
    markings = load_markings() if use_marking else None
    label    = "marking" if use_marking else "baseline"

    actor = ResidualActor()
    c1,c2   = SACCritic(),SACCritic()
    tc1,tc2 = SACCritic(),SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(),lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),  lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),  lr=3e-4)
    log_alpha = torch.zeros(1,requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha],lr=3e-4)
    buf = []
    gamma, tau = 0.99, 0.005

    name = "마킹 Residual RL" if use_marking else "기본 Residual RL"
    print(f"\n{'='*40}")
    print(f"=== {name} ===")
    print(f"{'='*40}")

    stages = [(45,500),(90,500),(135,700),(180,1000)]
    all_switched = []

    for stage_num,(angle,eps) in enumerate(stages):
        fname = f"residual_{angle}_{label}.pth"
        if os.path.exists(fname):
            actor.load_state_dict(torch.load(fname, weights_only=True))
            print(f"\n[{stage_num+1}단계] {fname} 불러옴 → 건너뜀")
            all_switched.extend([0]*eps)
            continue

        env = ResidualEnv(start_angle_deg=angle)
        print(f"\n[{stage_num+1}단계] 시작각도={angle}도 | {eps}에피소드")
        switched_list = []

        for ep in range(eps):
            obs=env.reset(); done=False; switched=False

            while not done:
                with torch.no_grad():
                    a,_=actor.get_action(
                        torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
                av=a.squeeze().item()
                no,r,term,just_switched=env.step(av)
                done=term

                if use_marking and env.mode=="swingup" and abs(obs[2])<np.radians(40):
                    r+=get_bonus(obs,markings)
                if just_switched:
                    switched=True; r+=5.0

                buf.append((obs,av,r,no,float(done)))
                if len(buf)>20000: buf.pop(0)

                if len(buf)>=64:
                    idx=np.random.choice(len(buf),64,replace=False)
                    b=[buf[i] for i in idx]
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
                obs=no

            switched_list.append(1 if switched else 0)
            if (ep+1)%100==0:
                rate=sum(switched_list[-100:])/100*100
                print(f"    ep={ep+1} | 전환성공률={rate:.0f}%/100")

        env.close()
        torch.save(actor.state_dict(),fname)
        rate=sum(switched_list)/len(switched_list)*100
        print(f"  → 저장: {fname} | 성공률={rate:.1f}%")
        all_switched.extend(switched_list)

    final = f"residual_{label}.pth"
    torch.save(actor.state_dict(),final)
    print(f"\n최종 저장: {final}")
    return actor, all_switched


# ================================
# 시각화
# ================================
def visualize(sac_actor, balance_actor, label="Residual RL", episodes=5):
    pygame.init()
    env = ResidualEnv(start_angle_deg=180, render_mode="human")
    print(f"\n=== {label} 시각화 ===")
    print("파란 게이지 = 물리 제어기")
    print("빨간 게이지 = SAC 보정값")

    for ep in range(episodes):
        obs=env.reset(); done=False; total_r=0; running=True
        while not done and running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
            with torch.no_grad():
                st=torch.FloatTensor(obs).unsqueeze(0)
                if env.mode=="swingup":
                    a=sac_actor.get_action(st)
                else:
                    a=balance_actor.get_action(st)
            obs,r,term,just_switched=env.step(a.squeeze().item())
            done=term; total_r+=r
            if just_switched:
                print(f"  전환 성공! step={env.switch_step} | "
                      f"각도={np.degrees(env.state[2]):.1f}도")
        print(f"ep={ep+1} | reward={total_r:.1f} | "
              f"전환={'성공✅' if env.switch_step else '실패❌'}")
    env.close()


# ================================
# 그래프 비교
# Curriculum SAC vs Residual RL
# ================================
def plot_comparison(curr_base_sw, curr_mark_sw, res_base_sw, res_mark_sw):
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle('Curriculum SAC vs Residual RL Comparison',
                fontsize=14, fontweight='bold')

    def moving_rate(data, w=50):
        return [sum(data[max(0,i-w):i+1])/min(i+1,w)*100
                for i in range(len(data))]

    # 기본 SAC 비교
    ax1 = axes[0]
    ax1.plot(moving_rate(curr_base_sw), color='steelblue',
            linewidth=2, label='Curriculum SAC (기본)')
    ax1.plot(moving_rate(res_base_sw),  color='tomato',
            linewidth=2, label='Residual RL (기본)')
    for x,lbl in [(500,'45→90'),(1000,'90→135'),(1700,'135→180')]:
        ax1.axvline(x=x,color='gray',linestyle='--',alpha=0.5)
        ax1.text(x+10,5,lbl,fontsize=8,color='gray')
    ax1.set_title('Baseline: Curriculum vs Residual RL')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Switch Success Rate (%)')
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # 마킹 SAC 비교
    ax2 = axes[1]
    ax2.plot(moving_rate(curr_mark_sw), color='steelblue',
            linewidth=2, label='Curriculum SAC (마킹)')
    ax2.plot(moving_rate(res_mark_sw),  color='tomato',
            linewidth=2, label='Residual RL (마킹)')
    for x,lbl in [(500,'45→90'),(1000,'90→135'),(1700,'135→180')]:
        ax2.axvline(x=x,color='gray',linestyle='--',alpha=0.5)
        ax2.text(x+10,5,lbl,fontsize=8,color='gray')
    ax2.set_title('Marking: Curriculum vs Residual RL')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Switch Success Rate (%)')
    ax2.legend(); ax2.grid(True,alpha=0.3)

    plt.tight_layout()
    plt.savefig('residual_comparison.png',dpi=150,bbox_inches='tight')
    print("그래프 저장: residual_comparison.png")


# ================================
# 메인 실행
# ================================
if __name__=="__main__":
    # 카트폴 PPO (저장된 거 있으면 재사용)
    balance_actor = train_balance()

    # Residual RL 학습
    base_actor, base_sw = train_residual(use_marking=False)
    mark_actor, mark_sw = train_residual(use_marking=True)

    # 시각화
    visualize(base_actor, balance_actor, label="기본 Residual RL", episodes=5)
    visualize(mark_actor, balance_actor, label="마킹 Residual RL", episodes=5)

    # Curriculum SAC 결과 불러와서 비교
    # (curriculum_sac.py 결과랑 비교)
    print("\n===== 최종 요약 =====")
    total_eps = 500+500+700+1000
    print(f"기본 Residual RL 전환성공: {sum(base_sw)}회 / {total_eps}에피소드")
    print(f"마킹 Residual RL 전환성공: {sum(mark_sw)}회 / {total_eps}에피소드")
    print(f"기본 Residual RL 4단계 성공률: {sum(base_sw[-1000:])/1000*100:.1f}%")
    print(f"마킹 Residual RL 4단계 성공률: {sum(mark_sw[-1000:])/1000*100:.1f}%")
