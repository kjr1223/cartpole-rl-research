import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import pygame

# ================================
# 환경: 스윙업 → 카트폴 전환 구조
# ================================
class SwitchingEnv(gym.Env):
    def __init__(self):
        self.gravity, self.mass_cart, self.mass_pole = 9.8, 1.0, 0.1
        self.length, self.dt, self.force_mag, self.x_limit = 0.5, 0.02, 10.0, 2.4
        self.switch_angle = np.radians(20)  # 전환 기준: ±30도
        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.screen = self.clock = None
        self.np_random = np.random.default_rng()
        pygame.init()

    def reset(self):
        # 막대가 아래서 시작
        self.state = np.array([0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
        self.mode        = "swingup"  # 초기 모드
        self.switch_step = None
        self.total_steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag

        # 라그랑지안 동역학 계산
        mt = self.mass_cart + self.mass_pole
        ml, c, s = self.mass_pole * self.length, np.cos(theta), np.sin(theta)
        tmp    = (force + ml * theta_dot**2 * s) / mt
        th_acc = (self.gravity * s - c * tmp) / \
                 (self.length * (4/3 - self.mass_pole * c**2 / mt))
        x_acc  = tmp - ml * th_acc * c / mt

        # 상태 업데이트
        x         += self.dt * x_dot;   x_dot     += self.dt * x_acc
        theta     += self.dt * theta_dot; theta_dot += self.dt * th_acc
        theta      = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.total_steps += 1

        # 전환 감지: ±30도 이내 진입하면 카트폴 모드로 전환
        just_switched = False
        if self.mode == "swingup" and abs(theta) < self.switch_angle:
            self.mode        = "balance"
            self.switch_step = self.total_steps
            just_switched    = True

        terminated = bool(abs(x) > self.x_limit)
        return self.state.copy(), terminated, just_switched

    def render(self, current_action=0.0):
        W, H, scale = 600, 450, 600 / (2 * self.x_limit * 2)
        if not self.screen:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("수동 조종 + 전환 구조 테스트")
            self.clock = pygame.time.Clock()

        # 모드별 배경색
        self.screen.fill((255,245,245) if self.mode=="swingup" else (245,255,245))

        x, _, theta, _ = self.state
        cx, cy = int(x * scale + W/2), int(H * 0.55)

        # 전환 기준선 (±30도)
        for sign in [1, -1]:
            ex = cx + int(80 * np.sin(self.switch_angle * sign))
            ey = cy - int(80 * np.cos(self.switch_angle * sign))
            pygame.draw.line(self.screen, (200,200,0), (cx,cy), (ex,ey), 2)

        # 바닥선
        pygame.draw.line(self.screen, (0,0,0), (0,cy+20), (W,cy+20), 2)

        # 카트 (모드별 색상)
        cart_color = (70,130,180) if self.mode=="swingup" else (50,180,50)
        pygame.draw.rect(self.screen, cart_color, (cx-30, cy-10, 60, 20))

        # 막대
        pl = int(self.length * 2 * scale)
        px, py = cx + int(pl*np.sin(theta)), cy - int(pl*np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60), (cx,cy), (px,py), 6)
        pygame.draw.circle(self.screen, (220,60,60), (px,py), 8)

        # 각도 게이지
        gx, gy, gw = 80, 40, 440
        pygame.draw.rect(self.screen, (200,200,200), (gx,gy,gw,15))
        pygame.draw.rect(self.screen, (100,200,100),
                        (gx, gy, int(gw*(theta+np.pi)/(2*np.pi)), 15))
        # 빨간선 = 위(0도), 주황선 = 전환기준(±30도)
        for angle, color in [(0,(255,0,0)),
                              (self.switch_angle,(255,165,0)),
                              (-self.switch_angle,(255,165,0))]:
            sx = gx + int(gw * (angle + np.pi) / (2*np.pi))
            pygame.draw.line(self.screen, color, (sx,gy-5), (sx,gy+20), 2)

        # 행동값 게이지
        ax_y = 75
        pygame.draw.rect(self.screen, (200,200,200), (gx, ax_y, gw, 12))
        mid  = gx + gw//2
        bw   = int(abs(current_action) * gw/2)
        col  = (255,100,0) if current_action > 0 else (0,100,255)
        if current_action > 0:
            pygame.draw.rect(self.screen, col, (mid, ax_y, bw, 12))
        else:
            pygame.draw.rect(self.screen, col, (mid-bw, ax_y, bw, 12))
        pygame.draw.line(self.screen, (0,0,0), (mid,ax_y-3), (mid,ax_y+15), 2)

        # 텍스트
        font  = pygame.font.SysFont(None, 30)
        font2 = pygame.font.SysFont(None, 22)
        mode_text  = "🔵 SWING-UP 모드 (← → 조종)" if self.mode=="swingup" \
                     else "🟢 BALANCE 모드 (자동 제어 중)"
        mode_color = (0,0,200) if self.mode=="swingup" else (0,150,0)

        self.screen.blit(font.render(
            f"각도: {np.degrees(theta):.1f}도  |  {mode_text}",
            True, mode_color), (10, 10))
        self.screen.blit(font2.render(
            "게이지: 빨간=위(0도) | 주황=전환기준(±30도)",
            True, (100,100,100)), (gx, gy-18))
        self.screen.blit(font2.render(
            f"행동값: {current_action:.2f}  |  step={self.total_steps}  |  "
            f"전환={'완료!(step='+str(self.switch_step)+')' if self.switch_step else '대기중'}",
            True, (80,80,80)), (10, H-30))
        self.screen.blit(font2.render(
            "← → 조종 | R: 리셋 | Q: 종료",
            True, (120,120,120)), (10, H-55))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen: pygame.quit(); self.screen = None


# ================================
# 카트폴 PPO (balance 모드용)
# 학습된 모델 불러와서 자동 제어
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
        return (a.float() * 2 - 1)  # 0→-1, 1→+1


# ================================
# balance 모드용 PPO 빠르게 학습
# ================================
def train_balance():
    env   = gym.make("CartPole-v1")
    actor = BalanceActor()
    vnet  = nn.Sequential(
        nn.Linear(4,64), nn.Tanh(),
        nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    p_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(vnet.parameters(),  lr=3e-4)

    print("카트폴 PPO 학습 중... (200 에피소드)")
    for ep in range(200):
        states, actions, rews, lps = [], [], [], []
        obs, _ = env.reset(); done = False
        while not done:
            st    = torch.FloatTensor(obs)
            dist  = Categorical(actor.net(st))
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
            vl = torch.nn.functional.mse_loss(vnet(st).squeeze(), rt)
            p_opt.zero_grad(); pl.backward(); p_opt.step()
            v_opt.zero_grad(); vl.backward(); v_opt.step()
        if (ep+1) % 50 == 0: print(f"  ep={ep+1}/200 완료")
    env.close()
    print("카트폴 PPO 학습 완료!")
    return actor


# ================================
# 메인: 수동 조종 + 전환 구조 테스트
# ================================
if __name__ == "__main__":
    # 카트폴 제어기 먼저 학습
    balance_actor = train_balance()

    env = SwitchingEnv()
    obs = env.reset()

    print("\n=== 수동 조종 + 전환 구조 테스트 ===")
    print("← → : 카트 조종 (스윙업 모드)")
    print("막대가 ±30도 이내 진입하면 자동으로 카트폴 모드 전환!")
    print("R: 리셋 | Q: 종료")
    print("=====================================\n")

    running = True
    current_action = 0.0

    while running:
        space_pressed = reset_pressed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                if event.key == pygame.K_r: reset_pressed = True

        if reset_pressed:
            obs = env.reset()
            print("  리셋!")
            continue

        # 모드에 따라 행동 결정
        if env.mode == "swingup":
            # 사람이 직접 조종
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:   current_action =  1.0
            elif keys[pygame.K_LEFT]:  current_action = -1.0
            else:                      current_action =  0.0
        else:
            # 카트폴 모드: PPO가 자동 제어
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0)
                current_action = balance_actor.get_action(st).squeeze().item()

        obs, terminated, just_switched = env.step(current_action)
        env.render(current_action)

        if just_switched:
            print(f"  🔄 전환 성공! step={env.switch_step} | "
                  f"각도={np.degrees(env.state[2]):.1f}도")
            print("  → 이제 PPO가 자동으로 막대를 잡아요!")

        if terminated:
            print("  카트 벗어남 → 리셋")
            obs = env.reset()

    env.close()
    print("종료!")
