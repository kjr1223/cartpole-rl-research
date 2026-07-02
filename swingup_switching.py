import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
import pygame

# ================================
# 환경: 스윙업 → 카트폴 전환 구조
# ================================
class SwitchingEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle = np.radians(30)  # 전환 기준: ±30도
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode
        self.screen = self.clock = None
        self.np_random = np.random.default_rng()
        pygame.init()

    def reset(self, seed=None, options=None):
        if seed: self.np_random = np.random.default_rng(seed)
        self.state = np.array([0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
        self.mode, self.switch_step, self.total_steps = "swingup", None, 0
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole * self.length, np.cos(theta), np.sin(theta)
        tmp = (force + ml * theta_dot**2 * s) / mt
        th_acc = (self.gravity * s - c * tmp) / (self.length * (4/3 - self.mass_pole * c**2 / mt))
        x_acc = tmp - ml * th_acc * c / mt
        x += self.dt * x_dot; x_dot += self.dt * x_acc
        theta += self.dt * theta_dot; theta_dot += self.dt * th_acc
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.total_steps += 1

        # 전환 감지
        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode, self.switch_step, just_switched = "balance", self.total_steps, True
            print(f"  🔄 전환! step={self.total_steps} | 각도={np.degrees(theta):.1f}도")

        # 모드별 reward
        if self.mode == "swingup":
            reward = (np.cos(theta) + 0.1 * self.length * (1 + np.cos(theta))
                     + (0.5 if abs(theta) < np.pi/2 else 0) - 0.1*abs(x) - 0.001*theta_dot**2)
        else:
            reward = np.cos(theta) - 0.1*abs(x) - 0.1*theta_dot**2 + (1.0 if just_switched else 0)

        if self.render_mode == "human": self._render()
        return self.state.copy(), reward, bool(abs(x) > self.x_limit), False, {"just_switched": just_switched}

    def _render(self):
        W, H, scale = 600, 450, 600 / (2 * self.x_limit * 2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Swing-up → CartPole 전환")
            self.clock = pygame.time.Clock()

        # 모드별 배경색
        self.screen.fill((255,245,245) if self.mode == "swingup" else (245,255,245))
        x, _, theta, _ = self.state
        cx, cy = int(x * scale + W/2), int(H * 0.55)

        # 전환 각도 표시선
        for sign in [1, -1]:
            ex = cx + int(80 * np.sin(self.switch_angle * sign))
            ey = cy - int(80 * np.cos(self.switch_angle * sign))
            pygame.draw.line(self.screen, (200,200,0), (cx,cy), (ex,ey), 1)

        pygame.draw.line(self.screen, (0,0,0), (0,cy+20), (W,cy+20), 2)
        pygame.draw.rect(self.screen,
            (70,130,180) if self.mode=="swingup" else (50,180,50), (cx-30,cy-10,60,20))
        pl = int(self.length * 2 * scale)
        px, py = cx + int(pl*np.sin(theta)), cy - int(pl*np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60), (cx,cy), (px,py), 6)
        pygame.draw.circle(self.screen, (220,60,60), (px,py), 8)

        # 게이지
        gx, gy, gw = 100, 40, 400
        pygame.draw.rect(self.screen, (200,200,200), (gx,gy,gw,15))
        pygame.draw.rect(self.screen, (100,200,100), (gx,gy,int(gw*(theta+np.pi)/(2*np.pi)),15))
        for angle in [0, self.switch_angle, -self.switch_angle]:
            color = (255,0,0) if angle==0 else (255,165,0)
            sx = gx + int(gw * (angle + np.pi) / (2*np.pi))
            pygame.draw.line(self.screen, color, (sx,gy-5), (sx,gy+20), 2)

        font = pygame.font.SysFont(None, 28)
        mode_text  = "🔵 SWING-UP" if self.mode=="swingup" else "🟢 BALANCE"
        mode_color = (0,0,200) if self.mode=="swingup" else (0,150,0)
        self.screen.blit(font.render(
            f"각도: {np.degrees(theta):.1f}도  |  {mode_text}", True, mode_color), (10,10))
        self.screen.blit(pygame.font.SysFont(None,22).render(
            f"step={self.total_steps} | 전환={'완료(step='+str(self.switch_step)+')' if self.switch_step else '대기중'}",
            True, (80,80,80)), (10, H-30))
        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen: pygame.quit(); self.screen = None


# ================================
# 신경망: 스윙업용 SAC Actor
# ================================
class SwingupActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.mu  = nn.Linear(64, 1)
        self.log_std = nn.Linear(64, 1)

    def get_action(self, x, with_logprob=False):
        f = self.net(x)
        dist = Normal(self.mu(f), self.log_std(f).clamp(-20,2).exp())
        xt = dist.rsample()
        a  = torch.tanh(xt)
        if not with_logprob: return a
        lp = (dist.log_prob(xt) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp


# ================================
# 신경망: 카트폴용 PPO Actor
# ================================
class BalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2), nn.Softmax(dim=-1))

    def get_action(self, x):
        a = Categorical(self.net(x)).sample()
        return (a.float() * 2 - 1)  # 0→-1, 1→+1 변환


# ================================
# SAC Critic
# ================================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, s, a): return self.net(torch.cat([s,a], dim=-1))


# ================================
# 스윙업 SAC 학습
# 목표: 막대를 ±30도 이내로 올리기
# ================================
def train_swingup(episodes=500):
    env = SwitchingEnv()
    actor = SwingupActor()
    c1, c2, tc1, tc2 = Critic(), Critic(), Critic(), Critic()
    tc1.load_state_dict(c1.state_dict()); tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),   lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),   lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    al_opt = torch.optim.Adam([log_alpha], lr=3e-4)
    buf, gamma, tau = [], 0.99, 0.005

    print("=== 스윙업 SAC 학습 ===")
    switch_count = 0
    for ep in range(episodes):
        obs, _ = env.reset(); done = False; total_r = 0; switched = False
        while not done:
            with torch.no_grad():
                a, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
            av = a.squeeze().item()
            no, r, term, trunc, info = env.step(av)
            done = term or trunc
            if info["just_switched"]: switched = True; done = True
            buf.append((obs, av, r, no, float(done)))
            if len(buf) > 10000: buf.pop(0)
            if len(buf) >= 64:
                idx = np.random.choice(len(buf), 64, replace=False)
                b   = [buf[i] for i in idx]
                s  = torch.FloatTensor(np.array([x[0] for x in b]))
                a_ = torch.FloatTensor(np.array([x[1] for x in b])).unsqueeze(1)
                r_ = torch.FloatTensor(np.array([x[2] for x in b])).unsqueeze(1)
                ns = torch.FloatTensor(np.array([x[3] for x in b]))
                d_ = torch.FloatTensor(np.array([x[4] for x in b])).unsqueeze(1)
                with torch.no_grad():
                    na, nlp = actor.get_action(ns, with_logprob=True)
                    tv = r_ + gamma*(1-d_)*(torch.min(tc1(ns,na),tc2(ns,na)) - log_alpha.exp()*nlp)
                c1_opt.zero_grad(); F.mse_loss(c1(s,a_),tv).backward(); c1_opt.step()
                c2_opt.zero_grad(); F.mse_loss(c2(s,a_),tv).backward(); c2_opt.step()
                na2, nlp2 = actor.get_action(s, with_logprob=True)
                al = (log_alpha.exp()*nlp2 - torch.min(c1(s,na2),c2(s,na2))).mean()
                a_opt.zero_grad(); al.backward(); a_opt.step()
                al2 = -(log_alpha*(nlp2+-1.0).detach()).mean()
                al_opt.zero_grad(); al2.backward(); al_opt.step()
                for p,tp in zip(c1.parameters(),tc1.parameters()): tp.data.copy_(tau*p+(1-tau)*tp)
                for p,tp in zip(c2.parameters(),tc2.parameters()): tp.data.copy_(tau*p+(1-tau)*tp)
            obs = no; total_r += r
        if switched: switch_count += 1
        if (ep+1) % 50 == 0:
            print(f"ep={ep+1} | reward={total_r:.1f} | 전환성공={switch_count}회")
            switch_count = 0
    torch.save(actor.state_dict(), "swingup_sac_actor.pth")
    print("저장완료: swingup_sac_actor.pth")
    return actor


# ================================
# 카트폴 PPO 학습
# 목표: 막대가 위에 있을 때 버티기
# ================================
def train_balance(episodes=300):
    env = gym.make("CartPole-v1")
    actor = BalanceActor()
    vnet  = nn.Sequential(nn.Linear(4,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    p_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(vnet.parameters(),  lr=3e-4)

    print("\n=== 카트폴 PPO 학습 ===")
    rewards = []
    for ep in range(episodes):
        states, actions, rews, lps = [], [], [], []
        obs, _ = env.reset(); done = False; total_r = 0
        while not done:
            st = torch.FloatTensor(obs)
            probs = actor.net(st)
            dist  = Categorical(probs); action = dist.sample()
            no, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            states.append(obs); actions.append(action.item())
            rews.append(r); lps.append(dist.log_prob(action).item())
            obs = no; total_r += r
        rewards.append(total_r)
        st = torch.FloatTensor(np.array(states))
        at = torch.LongTensor(actions); lt = torch.FloatTensor(lps)
        R, rets = 0, []
        for r in reversed(rews): R = r + 0.99*R; rets.insert(0, R)
        rt = torch.FloatTensor(rets)
        rt = (rt - rt.mean()) / (rt.std() + 1e-8)
        adv = rt - vnet(st).squeeze().detach()
        for _ in range(3):
            nlp = Categorical(actor.net(st)).log_prob(at)
            ratio = torch.exp(nlp - lt)
            pl = -torch.min(ratio*adv, torch.clamp(ratio,0.8,1.2)*adv).mean()
            vl = F.mse_loss(vnet(st).squeeze(), rt)
            p_opt.zero_grad(); pl.backward(); p_opt.step()
            v_opt.zero_grad(); vl.backward(); v_opt.step()
        if (ep+1) % 50 == 0:
            print(f"ep={ep+1} | reward={total_r:.1f} | 최근10평균={np.mean(rewards[-10:]):.1f}")
    torch.save(actor.state_dict(), "balance_ppo_actor.pth")
    print("저장완료: balance_ppo_actor.pth")
    env.close(); return actor


# ================================
# 전환 구조 시각화
# ================================
def visualize(swingup, balance, episodes=3):
    env = SwitchingEnv(render_mode="human")
    print("\n=== 시각화 ===")
    print("파란배경=스윙업 | 초록배경=카트폴 | 주황선=전환기준(±30도)")
    for ep in range(episodes):
        obs, _ = env.reset(); done = False; total_r = 0; running = True
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0)
                a  = swingup.get_action(st) if env.mode=="swingup" else balance.get_action(st)
            obs, r, term, trunc, info = env.step(a.squeeze().item())
            done = term or trunc; total_r += r
            if info["just_switched"]: print(f"  → 카트폴 모드 전환!")
        print(f"ep={ep+1} | reward={total_r:.1f} | 전환={'성공' if env.switch_step else '실패'}")
    env.close()


# ================================
# 실행
# ================================
if __name__ == "__main__":
    swingup_actor = train_swingup(episodes=500)
    balance_actor = train_balance(episodes=300)
    visualize(swingup_actor, balance_actor, episodes=3)
