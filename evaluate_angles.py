"""
evaluate_angles.py
==================
CartPole 초기각도별 (0°, 135°, 180°) 성공률 평가 스크립트

저장된 모델 불러와서 각 초기각도 조건에서 성공률 측정
학습 재실행 없이 rollout만 수행

[성공 기준]
    스윙업 후 balance 모드로 전환 성공 여부
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from curriculum_pinn import CurriculumEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


# =============================================
# SACActor: curriculum_pinn.py와 동일한 구조
# =============================================
class SACActor(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        f       = self.net(state)
        mu      = self.mu(f)
        log_std = self.log_std(f).clamp(-20, 2)
        return mu, log_std

    def select_action(self, state):
        """결정론적 행동 선택 (평가용)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = self.forward(state_t)
            action = torch.tanh(mu)
        return float(action.cpu().numpy().flatten()[0])


def load_actor(path):
    """
    .pth 파일에서 actor 가중치 로드
    저장 방식에 따라 키 구조가 다를 수 있어서 유연하게 처리
    """
    actor = SACActor().to(device)
    ckpt  = torch.load(path, map_location=device)

    # 저장 방식 1: {"actor": state_dict, ...}
    if isinstance(ckpt, dict) and "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    # 저장 방식 2: state_dict 직접 저장
    else:
        actor.load_state_dict(ckpt)

    actor.eval()
    print(f"[Load] {path}")
    return actor


def evaluate_model(actor, start_angle_deg, n_episodes=100, max_steps=1000, use_impulse=False):
    """
    주어진 초기각도에서 n_episodes번 평가
    
    Returns:
        success_rate : 성공률 (0.0 ~ 1.0)
        avg_reward   : 평균 누적 보상
        avg_steps    : 평균 스텝 수
    """
    env = CurriculumEnv(start_angle_deg=start_angle_deg, use_impulse=use_impulse)

    success_count = 0
    total_rewards = []
    total_steps   = []

    for ep in range(n_episodes):
        state    = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        success   = False

        for step in range(max_steps):
            action = actor.select_action(state)
            state, reward, terminated, just_switched = env.step(action)

            ep_reward += reward
            ep_steps  += 1

            # 성공: balance 모드로 전환됨
            if just_switched:
                success = True

            if terminated:
                break

        if success:
            success_count += 1

        total_rewards.append(ep_reward)
        total_steps.append(ep_steps)

    env.close()

    success_rate = success_count / n_episodes
    avg_reward   = np.mean(total_rewards)
    avg_steps    = np.mean(total_steps)

    return success_rate, avg_reward, avg_steps


def run_evaluation():
    """
    3가지 초기각도 × 3가지 모델 평가 실행
    """

    # 평가 설정
    configs = [
        {
            "label"     : "0°  (에너지 최대, 이미 서 있는 상태)",
            "angle"     : 0,
            "model_path": "stage_0_angle0.pth",
            "use_impulse": False,
        },
        {
            "label"     : "135° (중간 에너지)",
            "angle"     : 135,
            "model_path": "stage_135_angle135.pth",
            "use_impulse": False,
        },
        {
            "label"     : "180° (에너지 0, 완전 거꾸로)",
            "angle"     : 180,
            "model_path": "stage_180_impulse_ep1000.pth",
            "use_impulse": True,\
	    "use_impulse" : True, 
        },
    ]

    N_EPISODES = 100   # 에피소드 수
    MAX_STEPS  = 1000  # 에피소드당 최대 스텝

    print(f"\n{'='*60}")
    print(f"CartPole 초기각도별 성공률 평가")
    print(f"에피소드 수: {N_EPISODES} / 최대 스텝: {MAX_STEPS}")
    print(f"{'='*60}\n")

    results = []

    for cfg in configs:
        print(f"--- {cfg['label']} ---")

        # 모델 로드
        try:
            actor = load_actor(cfg["model_path"])
        except Exception as e:
            print(f"  [ERROR] 모델 로드 실패: {e}\n")
            continue

        # 평가 실행
        success_rate, avg_reward, avg_steps = evaluate_model(
            actor,
            start_angle_deg = cfg["angle"],
            n_episodes      = N_EPISODES,
            max_steps       = MAX_STEPS,
            use_impulse     = cfg.get("use_impulse", False),
        )

        results.append({
            "angle"       : cfg["angle"],
            "label"       : cfg["label"],
            "success_rate": success_rate,
            "avg_reward"  : avg_reward,
            "avg_steps"   : avg_steps,
        })

        print(f"  Success Rate : {success_rate*100:.1f}%")
        print(f"  Avg Reward   : {avg_reward:.2f}")
        print(f"  Avg Steps    : {avg_steps:.1f}")
        print()

    # 결과 표 출력
    print(f"\n{'='*60}")
    print(f"{'Angle':>8} | {'Success Rate':>13} | {'Avg Reward':>11} | {'Avg Steps':>10}")
    print(f"{'-'*8}-+-{'-'*13}-+-{'-'*11}-+-{'-'*10}")
    for r in results:
        print(f"  {r['angle']:>4}°  | {r['success_rate']*100:>12.1f}% | "
              f"{r['avg_reward']:>11.2f} | {r['avg_steps']:>10.1f}")
    print(f"{'='*60}\n")

    # 시각화
    plot_results(results)

    return results


def plot_results(results):
    """결과 시각화"""
    if not results:
        return

    angles       = [r["angle"] for r in results]
    success_rates = [r["success_rate"] * 100 for r in results]
    avg_rewards  = [r["avg_reward"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("CartPole Evaluation by Initial Angle", fontsize=13)

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = [f"{a}°" for a in angles]

    # 성공률 막대 그래프
    bars = axes[0].bar(labels, success_rates, color=colors, width=0.5, edgecolor="white")
    axes[0].set_title("Success Rate (%)")
    axes[0].set_xlabel("Initial Angle")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha='center', va='bottom', fontsize=11)

    # 평균 보상 막대 그래프
    bars2 = axes[1].bar(labels, avg_rewards, color=colors, width=0.5, edgecolor="white")
    axes[1].set_title("Average Reward")
    axes[1].set_xlabel("Initial Angle")
    axes[1].set_ylabel("Avg Reward")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, avg_rewards):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.5 if val >= 0 else -2),
                     f"{val:.1f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150, bbox_inches='tight')
    print("[Save] evaluation_results.png")


if __name__ == "__main__":
    results = run_evaluation()
