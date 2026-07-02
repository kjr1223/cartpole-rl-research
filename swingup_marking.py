import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sqlite3
import time

# ================================
# Swing-up 환경
# ================================
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
        self.marked_flash = 0  # 마킹 시 화면 번쩍임 효과

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

        total_mass = self.mass_cart + self.mass_pole
        ml = self.mass_pole * self.length
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        temp = (force + ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / \
                    (self.length * (4/3 - self.mass_pole * cos_t**2 / total_mass))
        x_acc = temp - ml * theta_acc * cos_t / total_mass

        x         = x         + self.dt * x_dot
        x_dot     = x_dot     + self.dt * x_acc
        theta     = theta     + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        theta     = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        upright_reward = np.cos(theta)
        x_penalty      = -0.1 * abs(x)
        vel_penalty    = -0.01 * theta_dot**2
        reward = upright_reward + x_penalty + vel_penalty

        terminated = bool(abs(x) > self.x_limit)

        if self.render_mode == "human":
            self._render()

        return self.state.copy(), reward, terminated, False, {}

    def _render(self, mark_flash=False):
        screen_w, screen_h = 600, 400
        scale = screen_w / (2 * self.x_limit * 2)

        if self.screen is None:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("CartPole Swing-up (SPACE: 마킹 / Q: 종료)")
            self.clock = pygame.time.Clock()

        # 마킹 시 배경 노란색 번쩍임
        if self.marked_flash > 0:
            self.screen.fill((255, 255, 150))
            self.marked_flash -= 1
        else:
            self.screen.fill((255, 255, 255))

        x, _, theta, _ = self.state
        cart_x = int(x * scale + screen_w / 2)
        cart_y = int(screen_h * 0.6)

        # 바닥선
        pygame.draw.line(self.screen, (0,0,0),
                        (0, cart_y+20), (screen_w, cart_y+20), 2)

        # 카트
        pygame.draw.rect(self.screen, (70,130,180),
                        (cart_x-30, cart_y-10, 60, 20))

        # 막대
        pole_len = int(self.length * 2 * scale)
        pole_x = cart_x + int(pole_len * np.sin(theta))
        pole_y = cart_y - int(pole_len * np.cos(theta))
        pygame.draw.line(self.screen, (220,60,60),
                        (cart_x, cart_y), (pole_x, pole_y), 6)
        pygame.draw.circle(self.screen, (220,60,60), (pole_x, pole_y), 8)

        # 상태 텍스트
        font = pygame.font.SysFont(None, 28)
        angle_deg = np.degrees(theta)
        is_up = abs(theta) < 0.3
        status = "UP! Stabilizing" if is_up else "Swinging..."
        color  = (0,150,0) if is_up else (150,0,0)
        text = font.render(
            f"Angle: {angle_deg:.1f} deg  |  {status}", True, color)
        self.screen.blit(text, (10, 10))

        # 마킹 안내
        font2 = pygame.font.SysFont(None, 24)
        guide = font2.render(
            "SPACE: Mark this moment! | Q: Quit", True, (100,100,100))
        self.screen.blit(guide, (10, screen_h - 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================================
# DB 준비 (Swing-up 전용 테이블)
# ================================
conn = sqlite3.connect("marking_data.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS swingup_markings (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        step      INTEGER,
        cart_pos  REAL,
        cart_vel  REAL,
        pole_ang  REAL,
        pole_vel  REAL,
        marked_at TEXT
    )
""")
conn.commit()
print("DB 준비 완료: swingup_markings 테이블")

# ================================
# 메인 실행
# ================================
env = CartPoleSwingUpEnv(render_mode="human")
obs, _ = env.reset()

print("\n=== Swing-up 마킹 시작 ===")
print("막대가 위로 올라가려는 찰나에 스페이스바!")
print("각도가 -30 ~ 30도 사이로 들어오는 순간이 좋아요")
print("Q키: 종료")
print("==========================\n")

step = 0
session_marks = 0
running = True

while running:
    space_pressed = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_pressed = True
            if event.key == pygame.K_q:
                running = False

    # 마킹!
    if space_pressed:
        cart_pos, cart_vel, pole_ang, pole_vel = obs
        angle_deg = np.degrees(pole_ang)

        cursor.execute("""
            INSERT INTO swingup_markings
            (step, cart_pos, cart_vel, pole_ang, pole_vel, marked_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            step,
            float(cart_pos), float(cart_vel),
            float(pole_ang), float(pole_vel),
            time.strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        session_marks += 1
        env.marked_flash = 5  # 화면 번쩍임

        print(f"⭐ 마킹 {session_marks}번째! "
              f"각도={angle_deg:.1f}도 | "
              f"각속도={pole_vel:.2f} | "
              f"step={step}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    step += 1

    if terminated:
        obs, _ = env.reset()

conn.close()
env.close()

print(f"\n총 {session_marks}개 마킹 저장 완료!")
print("다음: swingup_marking 데이터로 PPO 학습!")
