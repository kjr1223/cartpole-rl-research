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
# 환경: 스윙업 → 카트폴 전환 구조
# ================================
class SwitchingEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle = np.radians(20)  # 전환 기준: ±20도
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode
        self.screen = self.clock = None
        self.np_random = np.random.default_rng()
        pygame.init()

    def reset(self):
        self.state = np.array([0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
        self.mode        = "swingup"
        self.switch_step = None
        self.total_steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole * self.length, np.cos(theta), np.sin(theta)
        tmp    = (force + ml * theta_dot**2 * s) / mt
        th_acc = (self.gravity * s - c * tmp) / \
                 (self.length * (4/3 - self.mass_pole * c**2 / mt))
        x_acc  = tmp - ml * th_acc * c / mt
        x         += self.dt * x_dot;    x_dot     += self.dt * x_acc
        theta     += self.dt * theta_dot; theta_dot += self.dt * th_acc
        theta      = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.total_steps += 1

        # 전환 감지
        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode        = "balance"
            self.switch_step = self.total_steps
            just_switched    = True

        # 모드별 reward
        if self.mode == "swingup":
            reward = (np.cos(theta)
                     + 0.1 * self.length * (1 + np.cos(theta))
                     + (0.5 if abs(theta) < np.pi/2 else 0.0)
                     - 0.1 * abs(x)
                     - 0.001 * theta_dot**2)
        else:
            reward = (np.cos(theta)
                     - 0.1 * abs(x)
                     - 0.1 * theta_dot**2
                     + (1.0 if just_switched else 0.0))

        terminated = bool(abs(x) > self.x_limit)

        if self.render_mode == "human":
            self._render()

        return self.state.copy(), reward, terminated, just_switched

    def _render(self):
        W, H, scale = 600, 450, 600 / (2 * self.x_limit * 2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("스윙업 → 카트폴 전환 (SAC)")
            self.clock = pygame.time.Clock()

        self.screen.fill((255,245,245) if self.mode=="swingup" else (245,255,245))
        x, _, theta, _ = self.state
        cx, cy = int(x * scale + W/2), int(H * 0.55)

        for sign in [1, -1]:
            ex = cx + int(80 * np.sin(self.switch_angle * sign))
            ey = cy - int(80 * np.cos(self.switch_angle * sign))
            pygame.draw.line(self.screen, (200,200,0), (cx,cy), (ex,ey), 2)

        pygame.draw.line(self.screen, (0,0,0), (0,cy+20), (W,cy+20), 2)
        pygame.draw.rect(self.screen,
            (70,130,180) if self.mode=="swingup" else (50,180,50),
            (cx-30, cy-10, 60, 20))
        pl = int(self.length * 2 * scale)
        px, py = cx+int(pl*np.sin(theta)), cy-int(pl*np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60), (cx,cy), (px,py), 6)
        pygame.draw.circle(self.screen, (220,60,60), (px,py), 8)

        gx, gy, gw = 80, 40, 440
        pygame.draw.rect(self.screen, (200,200,200), (gx,gy,gw,15))
        pygame.draw.rect(self.screen, (100,200,100),
                        (gx, gy, int(gw*(theta+np.pi)/(2*np.pi)), 15))
        for angle, color in [(0,(255,0,0)),
                              (self.switch_angle,(255,165,0)),
                              (-self.switch_angle,(255,165,0))]:
            sx = gx + int(gw*(angle+np.pi)/(2*np.pi))
            pygame.draw.line(self.screen, color, (sx,gy-5), (sx,gy+20), 2)

        font  = pygame.font.SysFont(None, 30)
        font2 = pygame.font.SysFont(None, 22)
        mode_text  = "🔵 SWING-UP (SAC)" if self.mode=="swingup" else "🟢 BALANCE (PPO)"
        mode_color = (0,0,200) if self.mode=="swingup" else (0,150,0)
        self.screen.blit(font.render(
            f"각도: {np.degrees(theta):.1f}도  |  {mode_text}",
            True, mode_color), (10,10))
        self.screen.blit(font2.render(
            f"step={self.total_steps} | "
            f"전환={'완료!(step='+str(self.switch_step)+')' if self.switch_step else '대기중'}",
            True, (80,80,80)), (10, H-30))
        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen: pygame.quit(); self.screen = None


# ================================
# 마킹 데이터 불러오기
# switching_markings 테이블에서 로드
# ================================
def load_markings():
    conn   = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cart_pos, cart_vel, pole_ang, pole_vel
        FROM switching_markings
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"마킹 데이터 {len(rows)}개 불러옴")
    return np.array(rows)


# ================================
# 마킹 보너스 계산
# 현재 state가 마킹된 state랑 가까우면 보너스
# ================================
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
            nn.Linear(4,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.mu      = nn.Linear(64, 1)
        self.log_std = nn.Linear(64, 1)

    def get_action(self, x, with_logprob=False):
        f    = self.net(x)
        dist = Normal(self.mu(f), self.log_std(f).clamp(-20,2).exp())
        xt   = dist.rsample()
        a    = torch.tanh(xt)
        if not with_logprob: return a
        lp = (dist.log_prob(xt) - torch.log(1-a.pow(2)+1e-6)).sum(-1, keepdim=True)
        return a, lp


class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=-1))


# ================================
# 카트폴 PPO (balance 모드 자동제어)
# ================================
class BalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2), nn.Softmax(dim=-1))

    def get_action(self, x):
        a = Categorical(self.net(x)).sample()
        return (a.float() * 2 - 1)


def train_balance(episodes=200):
    env   = gym.make("CartPole-v1")
    actor = BalanceActor()
    vnet  = nn.Sequential(
        nn.Linear(4,64), nn.Tanh(),
        nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    p_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(vnet.parameters(),  lr=3e-4)
    print("카트폴 PPO 학습 중...")
    for ep in range(episodes):
        states, actions, rews, lps = [], [], [], []
        obs, _ = env.reset(); done = False
        while not done:
            st     = torch.FloatTensor(obs)
            dist   = Categorical(actor.net(st))
            action = dist.sample()
            no, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            states.append(obs); actions.append(action.item())
            rews.append(r); lps.append(dist.log_prob(action).item())
            obs = no
        st = torch.FloatTensor(np.array(states))
        at = torch.LongTensor(actions); lt = torch.FloatTensor(lps)
        R, rets = 0, []
        for r in reversed(rews): R = r + 0.99*R; rets.insert(0, R)
        rt  = torch.FloatTensor(rets)
        rt  = (rt - rt.mean()) / (rt.std() + 1e-8)
        adv = rt - vnet(st).squeeze().detach()
        for _ in range(3):
            nlp   = Categorical(actor.net(st)).log_prob(at)
            ratio = torch.exp(nlp - lt)
            pl = -torch.min(ratio*adv, torch.clamp(ratio,0.8,1.2)*adv).mean()
            vl = F.mse_loss(vnet(st).squeeze(), rt)
            p_opt.zero_grad(); pl.backward(); p_opt.step()
            v_opt.zero_grad(); vl.backward(); v_opt.step()
        if (ep+1) % 50 == 0: print(f"  PPO ep={ep+1}/200 완료")
    env.close()
    print("카트폴 PPO 학습 완료!")
    return actor


# ================================
# SAC 학습 (마킹 반영 여부 선택 가능)
# ================================
def train_sac(episodes=5000, use_marking=False):
    env      = SwitchingEnv()
    markings = load_markings() if use_marking else None
    actor    = SACActor()
    c1, c2   = SACCritic(), SACCritic()
    tc1, tc2 = SACCritic(), SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),   lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),   lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha], lr=3e-4)
    buf, gamma, tau = [], 0.99, 0.005

    label = "마킹 SAC" if use_marking else "기본 SAC"
    print(f"\n=== {label} 학습 시작 ({episodes} 에피소드) ===")

    all_rewards, switch_counts = [], []
    switch_count = 0

    for ep in range(episodes):
        obs = env.reset(); done = False; total_r = 0; switched = False

        while not done:
            with torch.no_grad():
                a, _ = actor.get_action(
                    torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
            av = a.squeeze().item()
            no, r, term, just_switched = env.step(av)
            done = term

            # 마킹 보너스 추가 (마킹 SAC만)
            # 조건: swingup 모드 + 각도 ±40도 이내일 때만
            # 전환 직전 순간에만 보너스를 줘야 올바르게 학습
            if use_marking and env.mode == "swingup" and abs(obs[2]) < np.radians(40):
                r += get_bonus(obs, markings)

            # 전환 성공 시 큰 보너스 + 에피소드 계속 진행
            if just_switched:
                switched = True
                r += 5.0  # 전환 성공 보너스

            buf.append((obs, av, r, no, float(done)))
            if len(buf) > 10000: buf.pop(0)

            if len(buf) >= 64:
                idx = np.random.choice(len(buf), 64, replace=False)
                b   = [buf[i] for i in idx]
                s   = torch.FloatTensor(np.array([x[0] for x in b]))
                a_  = torch.FloatTensor(np.array([x[1] for x in b])).unsqueeze(1)
                r_  = torch.FloatTensor(np.array([x[2] for x in b])).unsqueeze(1)
                ns  = torch.FloatTensor(np.array([x[3] for x in b]))
                d_  = torch.FloatTensor(np.array([x[4] for x in b])).unsqueeze(1)

                with torch.no_grad():
                    na, nlp = actor.get_action(ns, with_logprob=True)
                    tv = r_ + gamma*(1-d_)*(
                        torch.min(tc1(ns,na), tc2(ns,na)) - log_alpha.exp()*nlp)

                c1_opt.zero_grad(); F.mse_loss(c1(s,a_),tv).backward(); c1_opt.step()
                c2_opt.zero_grad(); F.mse_loss(c2(s,a_),tv).backward(); c2_opt.step()

                na2, nlp2 = actor.get_action(s, with_logprob=True)
                al = (log_alpha.exp()*nlp2 - torch.min(c1(s,na2),c2(s,na2))).mean()
                a_opt.zero_grad(); al.backward(); a_opt.step()

                al2 = -(log_alpha*(nlp2+-1.0).detach()).mean()
                al_opt.zero_grad(); al2.backward(); al_opt.step()

                for p,tp in zip(c1.parameters(),tc1.parameters()):
                    tp.data.copy_(tau*p+(1-tau)*tp)
                for p,tp in zip(c2.parameters(),tc2.parameters()):
                    tp.data.copy_(tau*p+(1-tau)*tp)

            obs = no; total_r += r

        if switched: switch_count += 1
        all_rewards.append(total_r)
        switch_counts.append(1 if switched else 0)

        if (ep+1) % 100 == 0:
            avg_r  = np.mean(all_rewards[-100:])
            avg_sw = sum(switch_counts[-100:])
            print(f"ep={ep+1} | reward={total_r:.1f} | "
                  f"최근100 평균={avg_r:.1f} | "
                  f"전환성공={avg_sw}회/100")
            switch_count = 0

    fname = "marking_sac.pth" if use_marking else "baseline_sac.pth"
    torch.save(actor.state_dict(), fname)
    print(f"저장완료: {fname}")
    return actor, all_rewards, switch_counts


# ================================
# 시각화: 학습된 SAC + PPO 전환 확인
# ================================
def visualize(sac_actor, balance_actor, label="SAC", episodes=3):
    env = SwitchingEnv(render_mode="human")
    print(f"\n=== {label} 시각화 ===")
    print("파란배경=스윙업(SAC) | 초록배경=카트폴(PPO)")

    for ep in range(episodes):
        obs = env.reset(); done = False; total_r = 0; running = True

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0)
                if env.mode == "swingup":
                    a = sac_actor.get_action(st)
                else:
                    a = balance_actor.get_action(st)

            obs, r, term, just_switched = env.step(a.squeeze().item())
            done = term; total_r += r

            if just_switched:
                print(f"  🔄 전환 성공! step={env.switch_step} | "
                      f"각도={np.degrees(env.state[2]):.1f}도")

        print(f"ep={ep+1} | reward={total_r:.1f} | "
              f"전환={'성공✅' if env.switch_step else '실패❌'}")

    env.close()


# ================================
# 그래프: 기본 SAC vs 마킹 SAC 비교
# ================================
def plot_comparison(base_rewards, base_switches, mark_rewards, mark_switches):
    def moving_avg(data, w=10):
        return [np.mean(data[max(0,i-w):i+1]) for i in range(len(data))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('기본 SAC vs 마킹 SAC 비교', fontsize=14, fontweight='bold')

    # 학습 곡선
    eps = range(1, len(base_rewards)+1)
    axes[0].plot(eps, moving_avg(base_rewards), color='steelblue',
                linewidth=2, label='기본 SAC')
    axes[0].plot(eps, moving_avg(mark_rewards), color='tomato',
                linewidth=2, label='마킹 SAC')
    axes[0].set_title('학습 곡선 (이동평균 10)')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 전환 성공률
    window = 50
    base_rate = [sum(base_switches[max(0,i-window):i+1]) /
                 min(i+1, window) * 100 for i in range(len(base_switches))]
    mark_rate = [sum(mark_switches[max(0,i-window):i+1]) /
                 min(i+1, window) * 100 for i in range(len(mark_switches))]
    axes[1].plot(eps, base_rate, color='steelblue', linewidth=2, label='기본 SAC')
    axes[1].plot(eps, mark_rate, color='tomato',    linewidth=2, label='마킹 SAC')
    axes[1].set_title('전환 성공률 % (최근 50 에피소드)')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('전환 성공률 (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('switching_comparison.png', dpi=150, bbox_inches='tight')
    print("\n그래프 저장: switching_comparison.png")


# ================================
# 메인 실행
# ================================
if __name__ == "__main__":
    # 1. 카트폴 PPO 학습
    balance_actor = train_balance(episodes=200)

    # 2. 기본 SAC 학습 (마킹 없음)
    base_actor, base_rewards, base_switches = train_sac(
        episodes=5000, use_marking=False)

    # 3. 마킹 SAC 학습 (마킹 반영)
    mark_actor, mark_rewards, mark_switches = train_sac(
        episodes=5000, use_marking=True)

    # 4. 기본 SAC 시각화
    visualize(base_actor, balance_actor, label="기본 SAC", episodes=3)

    # 5. 마킹 SAC 시각화
    visualize(mark_actor, balance_actor, label="마킹 SAC", episodes=3)

    # 6. 그래프 비교
    plot_comparison(base_rewards, base_switches, mark_rewards, mark_switches)

    # 최종 요약
    print("\n===== 최종 요약 =====")
    print(f"기본 SAC 전체 평균: {np.mean(base_rewards):.1f}")
    print(f"마킹 SAC 전체 평균: {np.mean(mark_rewards):.1f}")
    print(f"기본 SAC 전환 성공: {sum(base_switches)}회 / 1000 에피소드")
    print(f"마킹 SAC 전환 성공: {sum(mark_switches)}회 / 1000 에피소드")
