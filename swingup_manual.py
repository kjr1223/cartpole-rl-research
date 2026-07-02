import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sqlite3
import time

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
        self.marked_flash = 0

        pygame.init()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            0.0, 0.0,
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

    def _render(self):
        screen_w, screen_h = 600, 400
        scale = screen_w / (2 * self.x_limit * 2)

        if self.screen is None:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("Swing-up 직접 조종")
            self.clock = pygame.time.Clock()

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

        # 각도 게이지
        gauge_x, gauge_y, gauge_w = 150, 45, 300
        pygame.draw.rect(self.screen, (200,200,200),
                        (gauge_x, gauge_y, gauge_w, 15))
        ratio = (theta + np.pi) / (2 * np.pi)
        pygame.draw.rect(self.screen, (100,200,100),
                        (gauge_x, gauge_y, int(gauge_w * ratio), 15))
        pygame.draw.line(self.screen, (255,0,0),
                        (gauge_x + gauge_w//2, gauge_y-5),
                        (gauge_x + gauge_w//2, gauge_y+20), 2)

        font3 = pygame.font.SysFont(None, 22)
        self.screen.blit(
            font3.render("아래(-180)", True, (100,100,100)),
            (gauge_x-10, gauge_y+18))
        self.screen.blit(
            font3.render("위(0)", True, (100,100,100)),
            (gauge_x + gauge_w//2 - 15, gauge_y+18))

        # 상태 텍스트
        font = pygame.font.SysFont(None, 30)
        angle_deg = np.degrees(theta)
        is_up = abs(theta) < 0.3
        status = "UP! 지금 마킹!" if is_up else "Swinging..."
        color  = (0,150,0) if is_up else (150,0,0)
        self.screen.blit(
            font.render(f"각도: {angle_deg:.1f}도  |  {status}", True, color),
            (10, 10))

        # 조종 안내
        font2 = pygame.font.SysFont(None, 24)
        self.screen.blit(
            font2.render("← → 조종  |  SPACE: 마킹  |  R: 리셋  |  Q: 종료",
                        True, (80,80,80)),
            (10, screen_h - 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================================
# DB 준비
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

# ================================
# 메인 실행
# ================================
env = CartPoleSwingUpEnv(render_mode="human")
obs, _ = env.reset()

print("=== Swing-up 직접 조종 + 마킹 ===")
print("← → 방향키로 카트 조종 (꾹 누르세요!)")
print("막대가 위로 올라오는 순간 SPACE로 마킹!")
print("R: 리셋 / Q: 종료")
print("==================================\n")

step = 0
session_marks = 0
running = True

# 입력 제어용 변수
key_cooldown = 0      # 키 입력 쿨다운
ACTION_DELAY = 3      # 몇 프레임마다 한 번 입력 반영 (클수록 느려짐)

while running:
    space_pressed = False
    reset_pressed = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_pressed = True
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                reset_pressed = True

    # 리셋
    if reset_pressed:
        obs, _ = env.reset()
        step = 0
        print("  리셋!")
        continue

    # 방향키: 쿨다운으로 속도 조절
    key_cooldown -= 1
    keys = pygame.key.get_pressed()

    if key_cooldown <= 0:
        if keys[pygame.K_RIGHT]:
            action = 1
            key_cooldown = ACTION_DELAY
        elif keys[pygame.K_LEFT]:
            action = 0
            key_cooldown = ACTION_DELAY
        else:
            action = env.action_space.sample()
    else:
        action = env.action_space.sample()

    # 마킹!
    if space_pressed:
        cart_pos, cart_vel, pole_ang, pole_vel = obs
        angle_deg = np.degrees(pole_ang)

        cursor.execute("""
            INSERT INTO swingup_markings
            (step, cart_pos, cart_vel, pole_ang, pole_vel, marked_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (step,
              float(cart_pos), float(cart_vel),
              float(pole_ang), float(pole_vel),
              time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        session_marks += 1
        env.marked_flash = 5

        print(f"⭐ 마킹 {session_marks}번째! "
              f"각도={angle_deg:.1f}도 | "
              f"각속도={pole_vel:.2f}")

    obs, reward, terminated, truncated, _ = env.step(action)
    step += 1

    if terminated:
        print("  (카트 벗어남, 자동 리셋)")
        obs, _ = env.reset()
        step = 0

conn.close()
env.close()
print(f"\n총 {session_marks}개 마킹 저장 완료!")
