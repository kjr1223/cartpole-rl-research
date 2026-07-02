import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
import pygame

# ================================
# 환경
# ================================
class CartPoleSwingUpContinuous(gym.Env):
    def __init__(self, render_mode=None):
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 0.5
        self.dt        = 0.02
        self.force_mag = 10.0
        self.x_limit   = 2.4

        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.render_mode    = render_mode
        self.screen         = None
        self.clock          = None
        self.state          = None
        self.current_action = 0.0
        self.np_random      = np.random.default_rng()
        pygame.init()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.state = np.array([
            0.0, 0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1),
            0.0
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        self.current_action = float(action)
        force = float(action) * self.force_mag

        total_mass = self.mass_cart + self.mass_pole
        ml    = self.mass_pole * self.length
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        temp      = (force + ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / \
                    (self.length * (4/3 - self.mass_pole * cos_t**2 / total_mass))
        x_acc     = temp - ml * theta_acc * cos_t / total_mass

        x         = x         + self.dt * x_dot
        x_dot     = x_dot     + self.dt * x_acc
        theta     = theta     + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        theta     = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        upright_reward = np.cos(theta)
        energy_bonus   = 0.1 * self.length * (1 + np.cos(theta))
        swing_bonus    = 0.5 if abs(theta) < np.pi / 2 else 0.0
        x_penalty      = -0.1 * abs(x)
        vel_penalty    = -0.001 * theta_dot**2
        reward = upright_reward + energy_bonus + swing_bonus + x_penalty + vel_penalty

        terminated = bool(abs(x) > self.x_limit)

        if self.render_mode == "human":
            self._render()

        return self.state.copy(), reward, terminated, False, {}

    def _render(self, agent_name="Agent"):
        screen_w, screen_h = 600, 450
        scale = screen_w / (2 * self.x_limit * 2)

        if self.screen is None:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption(f"Swing-up: {agent_name}")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        x, _, theta, _ = self.state
        cart_x = int(x * scale + screen_w / 2)
        cart_y = int(screen_h * 0.55)

        pygame.draw.line(self.screen, (0,0,0),
                        (0, cart_y+20), (screen_w, cart_y+20), 2)
        pygame.draw.rect(self.screen, (70,130,180),
                        (cart_x-30, cart_y-10, 60, 20))

        pole_len = int(self.length * 2 * scale)
        pole_x   = cart_x + int(pole_len * np.sin(theta))
        pole_y   = cart_y - int(pole_len * np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60),
                        (cart_x, cart_y), (pole_x, pole_y), 6)
        pygame.draw.circle(self.screen, (220,60,60), (pole_x, pole_y), 8)

        gauge_x, gauge_y, gauge_w = 100, 40, 400
        pygame.draw.rect(self.screen, (200,200,200),
                        (gauge_x, gauge_y, gauge_w, 15))
        ratio = (theta + np.pi) / (2 * np.pi)
        pygame.draw.rect(self.screen, (100,200,100),
                        (gauge_x, gauge_y, int(gauge_w * ratio), 15))
        pygame.draw.line(self.screen, (255,0,0),
                        (gauge_x + gauge_w//2, gauge_y-5),
                        (gauge_x + gauge_w//2, gauge_y+20), 2)

        action_gauge_y = 75
        pygame.draw.rect(self.screen, (200,200,200),
                        (gauge_x, action_gauge_y, gauge_w, 15))
        mid_x  = gauge_x + gauge_w // 2
        bar_w  = int(abs(self.current_action) * gauge_w / 2)
        a_color = (255,100,0) if self.current_action > 0 else (0,100,255)
        if self.current_action > 0:
            pygame.draw.rect(self.screen, a_color,
                           (mid_x, action_gauge_y, bar_w, 15))
        else:
            pygame.draw.rect(self.screen, a_color,
                           (mid_x - bar_w, action_gauge_y, bar_w, 15))
        pygame.draw.line(self.screen, (0,0,0),
                        (mid_x, action_gauge_y-5),
                        (mid_x, action_gauge_y+20), 2)

        font  = pygame.font.SysFont(None, 28)
        font2 = pygame.font.SysFont(None, 22)
        font3 = pygame.font.SysFont(None, 32)

        angle_deg = np.degrees(theta)
        is_up  = abs(theta) < 0.3
        status = "UP! 성공!" if is_up else "Swinging..."
        color  = (0,150,0) if is_up else (150,0,0)

        self.screen.blit(
            font.render(f"각도: {angle_deg:.1f}도  |  {status}", True, color),
            (10, 10))
        self.screen.blit(
            font2.render("각도 게이지 (빨간선=위)", True, (100,100,100)),
            (gauge_x, gauge_y - 18))
        self.screen.blit(
            font2.render(f"행동값: {self.current_action:.3f}", True, (100,100,100)),
            (gauge_x, action_gauge_y - 18))
        self.screen.blit(
            font3.render(agent_name, True, (50,50,50)),
            (10, screen_h - 40))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================================
# 저장된 PPO 구조 그대로
# ================================
class OriginalPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, x):
        probs  = self.forward(x)
        dist   = Categorical(probs)
        action = dist.sample()
        # 이산 행동을 연속값으로 변환 (0→-1.0, 1→+1.0)
        continuous_action = action.float() * 2 - 1
        return continuous_action


# ================================
# SAC 신경망
# get_action이 action 하나만 반환하도록 수정
# ================================
class SACActorContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mean_layer    = nn.Linear(64, 1)
        self.log_std_layer = nn.Linear(64, 1)

    def get_action(self, x):
        # action만 반환 (시각화용)
        features = self.network(x)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features).clamp(-20, 2)
        std      = log_std.exp()
        dist     = Normal(mean, std)
        x_t      = dist.rsample()
        action   = torch.tanh(x_t)
        return action

    def get_action_with_logprob(self, x):
        # 학습용: action + log_prob 같이 반환
        features = self.network(x)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features).clamp(-20, 2)
        std      = log_std.exp()
        dist     = Normal(mean, std)
        x_t      = dist.rsample()
        action   = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class SACCriticContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


# ================================
# 시각화 함수
# ================================
def visualize(model, agent_name, episodes=3):
    env = CartPoleSwingUpContinuous(render_mode="human")
    print(f"\n{agent_name} 시각화 시작! (창 닫으면 다음으로)")

    for ep in range(episodes):
        obs, _  = env.reset()
        done    = False
        total_r = 0
        step    = 0
        running = True

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = model.get_action(state_t)
            action_val = action.squeeze().item()

            obs, reward, terminated, truncated, _ = env.step(action_val)
            env._render(agent_name)
            done     = terminated or truncated
            total_r += reward
            step    += 1

        print(f"  에피소드 {ep+1}: {step}스텝 | reward={total_r:.1f}")

    env.close()


# ================================
# SAC 재학습 (get_action_with_logprob 사용)
# ================================
def train_sac_quick():
    env     = CartPoleSwingUpContinuous()
    actor   = SACActorContinuous()
    critic1 = SACCriticContinuous()
    critic2 = SACCriticContinuous()
    tc1     = SACCriticContinuous()
    tc2     = SACCriticContinuous()
    tc1.load_state_dict(critic1.state_dict())
    tc2.load_state_dict(critic2.state_dict())

    a_opt  = torch.optim.Adam(actor.parameters(),   lr=3e-4)
    c1_opt = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    c2_opt = torch.optim.Adam(critic2.parameters(), lr=3e-4)
    log_alpha      = torch.zeros(1, requires_grad=True)
    alpha_opt      = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -1.0
    buffer         = []
    gamma, tau     = 0.99, 0.005

    print("SAC 재학습 중... (300 에피소드)")
    for episode in range(300):
        obs, _ = env.reset()
        done   = False
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                # 학습 중엔 get_action_with_logprob 사용
                action, _ = actor.get_action_with_logprob(state_t)
            action_val = action.squeeze().item()
            next_obs, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            buffer.append((obs, action_val, reward, next_obs, float(done)))
            if len(buffer) > 10000:
                buffer.pop(0)

            if len(buffer) >= 64:
                idx   = np.random.choice(len(buffer), 64, replace=False)
                batch = [buffer[i] for i in idx]
                s  = torch.FloatTensor(np.array([b[0] for b in batch]))
                a  = torch.FloatTensor(np.array([b[1] for b in batch])).unsqueeze(1)
                r  = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
                ns = torch.FloatTensor(np.array([b[3] for b in batch]))
                d  = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1)

                with torch.no_grad():
                    na, nlp = actor.get_action_with_logprob(ns)
                    tq = torch.min(tc1(ns,na), tc2(ns,na))
                    tv = r + gamma*(1-d)*(tq - log_alpha.exp()*nlp)

                c1_opt.zero_grad()
                F.mse_loss(critic1(s,a), tv).backward()
                c1_opt.step()
                c2_opt.zero_grad()
                F.mse_loss(critic2(s,a), tv).backward()
                c2_opt.step()

                new_a, new_lp = actor.get_action_with_logprob(s)
                q  = torch.min(critic1(s,new_a), critic2(s,new_a))
                al = (log_alpha.exp()*new_lp - q).mean()
                a_opt.zero_grad(); al.backward(); a_opt.step()

                al2 = -(log_alpha*(new_lp+target_entropy).detach()).mean()
                alpha_opt.zero_grad(); al2.backward(); alpha_opt.step()

                for p, tp in zip(critic1.parameters(), tc1.parameters()):
                    tp.data.copy_(tau*p.data + (1-tau)*tp.data)
                for p, tp in zip(critic2.parameters(), tc2.parameters()):
                    tp.data.copy_(tau*p.data + (1-tau)*tp.data)

            obs = next_obs

        if (episode+1) % 100 == 0:
            print(f"  {episode+1}/300 완료")

    print("SAC 재학습 완료!")
    return actor


# ================================
# 실행
# ================================
print("=" * 40)
print("PPO 시각화")
print("=" * 40)
ppo_model = OriginalPolicyNetwork()
ppo_model.load_state_dict(
    torch.load("swingup_v2_baseline.pth", weights_only=True))
ppo_model.eval()
visualize(ppo_model, "PPO", episodes=3)

print("\n" + "=" * 40)
print("SAC 시각화")
print("=" * 40)
sac_model = train_sac_quick()
sac_model.eval()
visualize(sac_model, "SAC", episodes=3)

print("\n시각화 완료!")
