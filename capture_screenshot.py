"""
capture_screenshot.py
======================
CartPole Swing-up pygame 장면 캡처 → PNG 저장
APISAT 논문용 스크린샷 (막대가 세워진 순간 + 아래 위치 두 장)
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'offscreen')   # 헤드리스 렌더링

import numpy as np
import pygame
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartpole_swingup import CartPoleSwingUpEnv

def capture_frame(env, filename):
    """현재 pygame 화면을 PNG로 저장"""
    surface = env.screen
    pygame.image.save(surface, filename)
    print(f"저장: {filename}")

def save_scene(env, theta_deg, filename, label=""):
    """원하는 각도로 상태 설정 후 렌더링 → 저장"""
    theta = np.radians(theta_deg)
    env.state = np.array([0.0, 0.0, theta, 0.0], dtype=np.float32)

    # 렌더 화면이 없으면 초기화
    if env.screen is None:
        screen_w, screen_h = 600, 400
        env.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("CartPole Swing-up")
        env.clock = pygame.time.Clock()

    # _render 호출
    env._render()

    # 레이블 추가 (왼쪽 상단)
    if label:
        font = pygame.font.SysFont(None, 32)
        surf = font.render(label, True, (0, 80, 160))
        env.screen.blit(surf, (10, 40))

    pygame.display.flip()
    pygame.image.save(env.screen, filename)
    print(f"저장: {filename}  (theta={theta_deg}°)")

def main():
    pygame.init()
    env = CartPoleSwingUpEnv(render_mode="human")
    env.reset()

    # 두 장면 캡처
    # 1) 막대 아래 (초기 상태, theta=180°)
    save_scene(env, theta_deg=180.0,
               filename='screenshot_down.png',
               label='Initial: Pole Down (180°)')

    # 2) 막대 세워진 상태 (theta=0°, 직립)
    save_scene(env, theta_deg=0.0,
               filename='screenshot_up.png',
               label='Stabilized: Pole Up (0°)')

    # 3) 스윙 중간 (theta=60°, 상승 중)
    save_scene(env, theta_deg=60.0,
               filename='screenshot_swing.png',
               label='Swing-up in Progress (60°)')

    env.close()

    # matplotlib으로 3장 합쳐서 논문용 패널 이미지 생성
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    scenes = [
        ('screenshot_down.png',  '(a) Initial State\nPole Down (θ=180°)'),
        ('screenshot_swing.png', '(b) Swing-up Phase\nθ=60°'),
        ('screenshot_up.png',    '(c) Stabilized State\nPole Up (θ=0°)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CartPole Swing-up Sequence (Marking SAC)',
                 fontsize=14, fontweight='bold')

    for ax, (fname, title) in zip(axes, scenes):
        img = mpimg.imread(fname)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    out = 'apisat_swingup_sequence.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"패널 저장: {out}")

if __name__ == '__main__':
    main()
