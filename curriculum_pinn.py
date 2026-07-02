import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame
from pinn_model import PINNTrainer

# ================================
# 환경: Curriculum Learning 지원
# ================================
class CurriculumEnv(gym.Env):
    def __init__(self, start_angle_deg=45, render_mode=None, use_impulse=False):
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
        self.use_impulse = use_impulse
        self.impulse_duration = 3
        self.impulse_force = 1.0
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
        if self.use_impulse and self.total_steps < self.impulse_duration:
            action = self.impulse_force
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole*self.length, np.cos(theta), np.sin(theta)
        tmp    = (force + ml*theta_dot**2*s) / mt
        th_acc = (self.gravity*s - c*tmp) / \
                 (self.length*(4/3 - self.mass_pole*c**2/mt))
        x_acc  = tmp - ml*th_acc*c/mt
        x         += self.dt*x_dot;    
        x_dot     += self.dt*x_acc
        theta     += self.dt*theta_dot; 
        theta_dot += self.dt*th_acc
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
            reward = (- 0.5 * abs(E_goal - E_now)
                     + np.cos(theta)
                     + 0.1 * abs(theta_dot)
                     - 0.1 * abs(x))
        else:
            reward = (np.cos(theta)
                     - 0.1 * abs(x)
                     - 0.1 * theta_dot**2
                     + (3.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)
        if self.render_mode == "human": self._render()
        return self.state.copy(), reward, terminated, just_switched

    def _render(self):
        W, H, scale = 600, 450, 600/(2*self.x_limit*2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption(
                f"Curriculum SAC (시작각도={self.start_angle_deg}도)")
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
        mode_text  = f"SWING-UP (start:{self.start_angle_deg}deg)" \
                     if self.mode=="swingup" else "BALANCE (PPO)"
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
# 온톨로지 분류기
# ================================
def classify_state(obs):
    """
    현재 상태를 Normal/Caution/Danger로 분류
    - Danger:  |theta| < 20도 → 전환 직전, 결정론적 규칙 적용
    - Danger:  |theta| > 135도 → 180도 근처, 각속도 방향 강제 고정
    - Caution: 20도 < |theta| < 90도 → 전환 준비 + 스윙업 중간, Context 보너스
    - Normal:  90도 < |theta| < 135도 → 스윙업 초반, SAC 자유 학습
    """
    import numpy as np
    theta = obs[2]
    angle_deg = abs(np.degrees(theta))
    if angle_deg < 20:
        return "Danger"
    elif angle_deg > 135:
        return "Danger"
    elif angle_deg < 90:
        return "Caution"
    else:
        return "Normal"

def deterministic_action(obs):
    """
    결정론적 규칙 적용
    - 180도 근처(|θ| > 135도): 각속도 방향 강제 고정 → 방향 결정 문제 해결
    - Danger(|θ| < 20도): 에너지 기반 제어
    """
    import numpy as np
    _, _, theta, theta_dot = obs
    mass_pole, gravity, length = 0.1, 9.8, 0.5
    angle_deg = abs(np.degrees(theta))

    if angle_deg > 135:
        # 180도 근처: 각속도 방향으로 강하게 밀어서 방향 고정
        return float(np.sign(theta_dot)) if theta_dot != 0 else 1.0

    # Danger 구간 (|θ| < 20도): 에너지 기반 제어
    E_goal = 2 * mass_pole * gravity * length
    E_now  = 0.5 * mass_pole * (length * theta_dot)**2              + mass_pole * gravity * length * (1 - np.cos(theta))
    action = np.sign(theta_dot * np.cos(theta) * (E_goal - E_now))
    return float(action) if action != 0 else 0.0

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

def train_balance(episodes=200):
    # balance_ppo.pth 있으면 불러오기
    actor = BalanceActor()
    if __import__('os').path.exists("balance_ppo.pth"):
        actor.load_state_dict(torch.load("balance_ppo.pth", weights_only=True))
        print("카트폴 PPO 불러옴: balance_ppo.pth")
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
    torch.save(actor.state_dict(), "balance_ppo.pth")
    print("카트폴 PPO 완료! 저장: balance_ppo.pth")
    return actor


# ================================
# SAC 단계별 학습
# 각 단계마다 모델 저장 → 중간에 끊겨도 이어서 가능
# ================================
def train_sac_stage(actor, c1, c2, tc1, tc2,
                    a_opt, c1_opt, c2_opt, log_alpha, al_opt,
                    buf, start_angle_deg, episodes,
                    use_marking=False, markings=None,
                    use_ontology=False, label="", pinn=None):

    env = CurriculumEnv(start_angle_deg=start_angle_deg)
    gamma, tau = 0.99, 0.005
    print(f"\n  [시작각도={start_angle_deg}도] {episodes}에피소드 학습...")

    all_switched = []

    for ep in range(episodes):
        obs=env.reset(); done=False; switched=False

        while not done:
            # 온톨로지 분류기 적용
            situation = classify_state(obs)
            angle_deg = abs(np.degrees(obs[2]))
            if use_ontology and (situation == "Danger" or angle_deg > 135):
                av = deterministic_action(obs)
            else:
                with torch.no_grad():
                    a,_=actor.get_action(
                        torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
                av=a.squeeze().item()
            no,r,term,just_switched=env.step(av)
            done=term

            # 온톨로지 기반 보너스 적용
            # Caution 구간 (20~40도): Context 보너스
            # Danger 구간 (0~20도): 결정론적 규칙 적용 (보너스 X)
            if use_marking and env.mode=="swingup":
                situation = classify_state(obs)
                if use_ontology and situation == "Caution":
                    r+=get_bonus(obs, markings)
                elif not use_ontology:
                    r+=get_bonus(obs, markings)

            if just_switched:
                switched=True; r+=5.0

            buf.append((obs,av,r,no,float(done)))
            if pinn is not None and len(buf) > 10:
                for _ in range(3):
                    with torch.no_grad():
                        a_img = actor.get_action(torch.FloatTensor(obs).unsqueeze(0)).squeeze().item()
                    ns_img = pinn.predict(obs, a_img)
                    r_img  = float(np.cos(ns_img[2]))
                    buf.append((obs, a_img, r_img, ns_img, 0.0))
                    if len(buf) > 20000: buf.pop(0)
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

        all_switched.append(1 if switched else 0)

        if (ep+1)%100==0:
            rate = sum(all_switched[-100:])/100*100
            print(f"    ep={ep+1} | 전환성공률={rate:.0f}%/100")

    env.close()

    # 단계별 모델 저장 (중간 저장!)
    fname = f"stage_{start_angle_deg}_{label}.pth"
    torch.save(actor.state_dict(), fname)
    print(f"  → 저장: {fname}")

    success_rate = sum(all_switched)/len(all_switched)*100
    print(f"  → 최종 전환성공률: {success_rate:.1f}%")
    return success_rate, all_switched


# ================================
# Curriculum Learning 전체 실행
# ================================
def run_curriculum(use_marking=False, use_pinn=True, use_ontology=False, label_suffix=""):
    import os
    markings = load_markings() if use_marking else None
    label    = ("marking" if use_marking else "baseline") + label_suffix

    # PINN 로드
    pinn = PINNTrainer() if use_pinn else None
    if pinn is not None:
        try:
            pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
            print("PINN 모델 로드 완료")
        except:
            print("PINN 모델 없음 → PINN 없이 실행")
            pinn = None

    print(f"\n{'='*40}")
    print(f"=== {'마킹 SAC' if use_marking else '기본 SAC'} Curriculum Learning ===")
    print(f"{'='*40}")

    # 신경망 초기화
    actor = SACActor()
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

    # 단계 설정
    stages = [
        (45,  500),
        (90,  1000),
        (135, 700),
        (180, 1000),
    ]

    all_switched_total = []

    for stage_num, (angle, eps) in enumerate(stages):
        # 이미 저장된 단계 있으면 불러오기
        fname = f"stage_{angle}_{label}.pth"
        if os.path.exists(fname):
            actor.load_state_dict(torch.load(fname, weights_only=True))
            print(f"\n[{stage_num+1}단계] {fname} 불러옴 → 건너뜀")
            # 건너뛴 단계 switched 0으로 채움
            all_switched_total.extend([0]*eps)
            continue

        print(f"\n[{stage_num+1}단계] 시작각도={angle}도 | {eps}에피소드")
        rate, switched = train_sac_stage(
            actor, c1, c2, tc1, tc2,
            a_opt, c1_opt, c2_opt, log_alpha, al_opt,
            buf, angle, eps,
            use_marking=use_marking, markings=markings,
            use_ontology=use_ontology, label=label, pinn=pinn)
        all_switched_total.extend(switched)
        print(f"  [{stage_num+1}단계 완료] 성공률={rate:.1f}%")

    # 최종 모델 저장
    final_fname = f"curriculum_{label}.pth"
    torch.save(actor.state_dict(), final_fname)
    print(f"\n최종 저장: {final_fname}")
    return actor, all_switched_total


# ================================
# 시각화
# ================================
def visualize(sac_actor, balance_actor, label="SAC", episodes=5):
    pygame.init()
    env = CurriculumEnv(start_angle_deg=180, render_mode="human")
    print(f"\n=== {label} 시각화 (180도에서 시작) ===")
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
def plot_comparison(base_sw, mark_sw):
    fig, ax = plt.subplots(figsize=(12,5))
    fig.suptitle('Curriculum Learning: Baseline vs Marking SAC',
                fontsize=14, fontweight='bold')
    w   = 50
    eps = range(1, len(base_sw)+1)
    br  = [sum(base_sw[max(0,i-w):i+1])/min(i+1,w)*100 for i in range(len(base_sw))]
    mr  = [sum(mark_sw[max(0,i-w):i+1])/min(i+1,w)*100 for i in range(len(mark_sw))]
    ax.plot(eps, br, color='steelblue', linewidth=2, label='Baseline SAC')
    ax.plot(eps, mr, color='tomato',    linewidth=2, label='Marking SAC')
    for x, lbl in [(500,'45→90'), (1000,'90→135'), (1700,'135→180')]:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.7)
        ax.text(x+10, 5, lbl, fontsize=9, color='gray')
    ax.set_title('Switch Success Rate % (last 50 ep)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Success Rate (%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('curriculum_comparison.png', dpi=150, bbox_inches='tight')
    print("그래프 저장: curriculum_comparison.png")


# ================================
# 메인 실행
# ================================
if __name__=="__main__2":
    balance_actor = train_balance(episodes=200)
    base_actor, base_sw = run_curriculum(use_marking=False, use_pinn=False, use_ontology=False, label_suffix="_base")
    mark_actor, mark_sw = run_curriculum(use_marking=True)
    visualize(base_actor, balance_actor, label="기본 SAC", episodes=5)
    visualize(mark_actor, balance_actor, label="마킹 SAC", episodes=5)
    plot_comparison(base_sw, mark_sw)

    total_eps = 500+500+700+1000
    print(f"\n===== 최종 요약 =====")
    print(f"기본 SAC 전체 전환성공: {sum(base_sw)}회 / {total_eps}에피소드")
    print(f"마킹 SAC 전체 전환성공: {sum(mark_sw)}회 / {total_eps}에피소드")
    print(f"기본 SAC 4단계 성공률: {sum(base_sw[-1000:])/1000*100:.1f}%")
    print(f"마킹 SAC 4단계 성공률: {sum(mark_sw[-1000:])/1000*100:.1f}%")

# ================================
# PINN 비교 실행 (기존 메인 대체)
# ================================
def run_pinn_comparison(seed=42):
    set_seed(seed)
    print(f"\n시드: {seed}")
    import shutil
    # 기존 stage 파일 백업 (실험 구분용)
    import glob
    for f in glob.glob("stage_*_baseline.pth"): shutil.move(f, f+".bak")
    for f in glob.glob("stage_*_marking.pth"): shutil.move(f, f+".bak")
    print("\n" + "="*50)
    print("온톨로지 기반 Curriculum SAC 비교 실험")
    print("="*50)

    # 1. 기본 SAC (PINN X, Context X)
    print("\n[1] 기본 SAC")
    base_actor, base_sw = run_curriculum(use_marking=False, use_pinn=False, use_ontology=False, label_suffix="_base")

    # 2. PINN + SAC (Context X)
    # label을 pinn_base로 바꿔서 저장 파일 구분
    print("\n[2] PINN + SAC")
    onto_actor, onto_sw = run_curriculum(use_marking=False, use_pinn=False, use_ontology=True, label_suffix="_onto")

    # 3. PINN + Context + SAC
    print("\n[3] PINN + Context + SAC")
    full_actor, full_sw = run_curriculum(use_marking=True, use_pinn=True, use_ontology=True, label_suffix="_pinn_ctx")

    # 결과 시각화
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 50
    def moving_avg(sw):
        return [sum(sw[max(0,i-w):i+1])/min(i+1,w)*100
                for i in range(len(sw))]

    eps = range(1, len(base_sw)+1)
    ax.plot(eps, moving_avg(base_sw), color='gray',  linewidth=2, label='SAC (기본)')
    ax.plot(eps, moving_avg(onto_sw), color='blue',  linewidth=2, label='SAC + 온톨로지')
    ax.plot(eps, moving_avg(full_sw), color='red',   linewidth=2, label='PINN + Context + SAC + 온톨로지')

    for x, lbl in [(500,'45→90'), (1000,'90→135'), (1700,'135→180')]:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        ax.text(x+10, 5, lbl, fontsize=9, color='gray')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Switch Success Rate (%)')
    ax.set_title('온톨로지 기반 Curriculum SAC 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/jrkim/cartpole_project/pinn_curriculum_comparison.png',
                dpi=150)
    print("\n그래프 저장: pinn_curriculum_comparison.png")

    total_eps = 500+500+700+1000
    print("\n===== 최종 요약 =====")
    print(f"{'':30s} {'4단계 성공률':>12s} {'전체 전환':>10s}")
    print(f"{'SAC 기본':30s} "
          f"{sum(base_sw[-1000:])/1000*100:11.1f}% "
          f"{sum(base_sw):>10d}회")
    print(f"{'SAC + 온톨로지':30s} "
          f"{sum(onto_sw[-1000:])/1000*100:11.1f}% "
          f"{sum(onto_sw):>10d}회")
    print(f"{'PINN + Context + SAC + 온톨로지':30s} "
          f"{sum(full_sw[-1000:])/1000*100:11.1f}% "
          f"{sum(full_sw):>10d}회")

# ================================
# 시작 각도 조건별 실험
# ================================
# 교수님 피드백: "고정 시드 표현 대신 시작 각도로 표현"
#
# 각 조건의 물리적 의미:
#   180도: 완전히 뒤집힌 상태 (에너지 0, 가장 어려움)
#   135도: 중간 에너지 상태 (중간 난이도)
#     0도: 거의 세워진 상태 (에너지 충분, 상대적으로 쉬움)
#
# 실험 방식: 각 시작 각도를 4단계 커리큘럼의 마지막 단계로 고정하여
#           동일 조건(PINN + Context + 온톨로지) 하에 독립 실험
# ================================

def run_single_angle_experiment(fixed_angle_deg, episodes=1000, seed=42):
    """
    지정한 시작 각도에서만 집중 학습하는 단일 실험.
    커리큘럼 전체를 돌리는 것이 아니라,
    해당 각도 조건에서 성능을 집중 측정하는 것이 목적.

    Args:
        fixed_angle_deg: 시작 각도 (180 / 135 / 0 중 하나)
        episodes:        학습 에피소드 수 (기본 1000)
        seed:            재현성용 시드
    """
    import os
    set_seed(seed)

    print(f"\n{'='*50}")
    print(f"시작 각도: {fixed_angle_deg}도 | {episodes}에피소드 | seed={seed}")
    print(f"{'='*50}")

    label = f"angle{fixed_angle_deg}"

    # 마킹 & PINN 로드
    markings = load_markings()
    pinn = PINNTrainer()
    try:
        pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
        print("PINN 모델 로드 완료")
    except Exception:
        print("PINN 모델 없음 → PINN 없이 실행")
        pinn = None

    # 신경망 초기화
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

    # 저장 파일 있으면 불러오기
    fname = f"stage_{fixed_angle_deg}_{label}.pth"
    if os.path.exists(fname):
        actor.load_state_dict(torch.load(fname, weights_only=True))
        print(f"기존 모델 불러옴: {fname} → 이어서 학습")

    rate, switched = train_sac_stage(
        actor, c1, c2, tc1, tc2,
        a_opt, c1_opt, c2_opt, log_alpha, al_opt,
        buf, fixed_angle_deg, episodes,
        use_marking=True, markings=markings,
        use_ontology=True, label=label, pinn=pinn)

    print(f"\n[시작각도={fixed_angle_deg}도] 최종 전환성공률: {rate:.1f}%")
    return actor, switched


def run_init_angle_comparison(episodes=1000, seed=42):
    """
    3가지 시작 각도 조건을 순서대로 실험하고 비교 그래프 출력.

    조건:
      A. 180도 시작 - 완전히 뒤집힌 상태 (에너지 0, 가장 어려움)
      B. 135도 시작 - 중간 에너지 상태
      C.   0도 시작 - 거의 세워진 상태 (에너지 충분)
    """
    conditions = [
        (180, "A: 180도 시작\n(완전 역위치, 에너지=0)"),
        (135, "B: 135도 시작\n(중간 에너지)"),
        (0,   "C: 0도 근처 시작\n(거의 직립, 에너지 충분)"),
    ]

    results = {}  # {angle: switched_list}

    for angle, desc in conditions:
        print(f"\n\n{'#'*60}")
        print(f"# 조건: {desc.replace(chr(10), ' | ')}")
        print(f"{'#'*60}")
        _, switched = run_single_angle_experiment(angle, episodes=episodes, seed=seed)
        results[angle] = switched

    # ── 비교 그래프 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f'시작 각도별 전환 성공률 비교 (PINN + Context DB + 온톨로지, seed={seed})',
        fontsize=13, fontweight='bold')

    colors = {180: 'tomato', 135: 'steelblue', 0: 'seagreen'}
    titles = {
        180: 'A: 180도 시작\n(완전 역위치, 에너지=0)',
        135: 'B: 135도 시작\n(중간 에너지)',
        0:   'C: 0도 근처 시작\n(거의 직립, 에너지 충분)',
    }

    w = 50
    for ax, (angle, _) in zip(axes, conditions):
        sw = results[angle]
        ma = [sum(sw[max(0,i-w):i+1]) / min(i+1, w) * 100
              for i in range(len(sw))]
        eps_range = range(1, len(sw)+1)
        ax.plot(eps_range, ma, color=colors[angle], linewidth=2)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% 기준선')
        final_rate = sum(sw[-100:]) / min(100, len(sw)) * 100
        ax.set_title(f"{titles[angle]}\n최종 성공률: {final_rate:.1f}%", fontsize=10)
        ax.set_xlabel('Episode')
        ax.set_ylabel('전환 성공률 (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = '/home/jrkim/cartpole_project/init_angle_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프 저장: {out_path}")

    # ── 최종 요약 출력 ────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'조건':<30} {'최종 100ep 성공률':>15} {'전체 전환':>10}")
    print("-"*55)
    for angle, desc in conditions:
        sw = results[angle]
        final = sum(sw[-100:]) / min(100, len(sw)) * 100
        total = sum(sw)
        label_str = desc.replace('\n', ' ')
        print(f"{label_str:<30} {final:>14.1f}% {total:>10d}회")
    print("="*55)

    return results


# ================================
# 메인 진입점
# ================================
if __name__ == "__main__":
    import sys

    # 사용법:
    #   python3 curriculum_pinn.py                  → 기존 3-way 비교 실험 (seed=42)
    #   python3 curriculum_pinn.py --angle          → 시작 각도별 비교 실험 (seed=42)
    #   python3 curriculum_pinn.py --angle 42       → 시작 각도별 비교 실험 (seed=42)
    #   python3 curriculum_pinn.py 42               → 기존 3-way 비교 실험 (seed=42)

    if "--angle" in sys.argv:
        # 시작 각도 조건별 실험
        idx = sys.argv.index("--angle")
        seed = int(sys.argv[idx+1]) if idx+1 < len(sys.argv) and sys.argv[idx+1].isdigit() else 42
        run_init_angle_comparison(episodes=1000, seed=seed)
    else:
        # 기존 3-way 비교 실험 (SAC / SAC+온톨로지 / PINN+CTX+SAC+온톨로지)
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
        run_pinn_comparison(seed=seed)
