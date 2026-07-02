import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from scipy.integrate import solve_ivp

class CartPoleSwingUpEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.gravity   = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length    = 0.5
        self.dt        = 0.02
        self.force_mag  = 10.0
        self.x_limit    = 2.4

        high = np.array([self.x_limit, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode
        self.screen = None
        self.clock  = None
        self.state  = None

        # pygame 미리 초기화
        pygame.init()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            0.0,
            0.0,
            np.pi + self.np_random.uniform(-0.1, 0.1),
            0.0
        ], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        # RK45로 dt 구간 적분
        result = solve_ivp(
            fun=lambda t, s: self._derivatives(s, force),
            t_span=(0.0, self.dt),
            y0=[x, x_dot, theta, theta_dot],
            method='RK45',
        )
        x, x_dot, theta, theta_dot = result.y[:, -1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        upright_reward = np.cos(theta)
        x_penalty      = -0.1 * abs(x)
        vel_penalty    = -0.01 * theta_dot**2
        reward = upright_reward + x_penalty + vel_penalty

        terminated = bool(abs(x) > self.x_limit)
        truncated  = False

        if self.render_mode == "human":
            self._render()

        return self.state.copy(), reward, terminated, truncated, {}

    def _derivatives(self, s, force):
        """CartPole 연속 동역학 — RK45 적분기에서 호출"""
        _, x_dot, theta, theta_dot = s
        total_mass = self.mass_cart + self.mass_pole
        ml         = self.mass_pole * self.length
        cos_t      = np.cos(theta)
        sin_t      = np.sin(theta)

        temp      = (force + ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / \
                    (self.length * (4/3 - self.mass_pole * cos_t**2 / total_mass))
        x_acc     = temp - ml * theta_acc * cos_t / total_mass

        return [x_dot, x_acc, theta_dot, theta_acc]

    def _render(self):
        screen_w, screen_h = 600, 400
        scale = screen_w / (2 * self.x_limit * 2)

        if self.screen is None:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("CartPole Swing-up")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        x, _, theta, _ = self.state
        cart_x = int(x * scale + screen_w / 2)
        cart_y = int(screen_h * 0.6)

        pygame.draw.line(self.screen, (0,0,0), (0, cart_y+20), (screen_w, cart_y+20), 2)
        pygame.draw.rect(self.screen, (70,130,180), (cart_x-30, cart_y-10, 60, 20))

        pole_len = int(self.length * 2 * scale)
        pole_x = cart_x + int(pole_len * np.sin(theta))
        pole_y = cart_y - int(pole_len * np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60), (cart_x, cart_y), (pole_x, pole_y), 6)
        pygame.draw.circle(self.screen, (220,60,60), (pole_x, pole_y), 8)

        font = pygame.font.SysFont(None, 28)
        angle_deg = np.degrees(theta)
        is_up = abs(theta) < 0.3
        status = "UP! Stabilizing" if is_up else "Swinging..."
        color  = (0,150,0) if is_up else (150,0,0)
        text = font.render(f"Angle: {angle_deg:.1f} deg  |  {status}", True, color)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================================
# 테스트 실행
# ================================
if __name__ == "__main__":
    import time

    env = CartPoleSwingUpEnv(render_mode="human")
    obs, _ = env.reset()

    print("=== Swing-up 환경 테스트 ===")
    print("막대가 아래에서 시작하는 거 확인해보세요!")
    print("랜덤 행동으로 실행 중... (창 닫으면 종료)")

    running = True
    step = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if step % 50 == 0:
            print(f"step {step:3d} | 각도={np.degrees(obs[2]):.1f}도 | reward={reward:.3f}")

        if terminated:
            obs, _ = env.reset()

        step += 1

    env.close()
    print("완료!")
