"""
초기 각도 조건별 실험
- 0도:   종말 조건 근처 (직립에 가까운 상태)
- 135도: 중간 에너지 상태
- 180도: 에너지가 없는 상태 (불안정 평형점)
"""
import numpy as np
import torch
import shutil, glob, os
from curriculum_pinn import (
    SACActor, SACCritic, load_markings,
    train_sac_stage, set_seed
)
from pinn_model import PINNTrainer

def run_by_init_angle(init_angle_deg):
    print(f"\n{'='*50}")
    print(f"초기 시작 각도: {init_angle_deg}도")
    print(f"{'='*50}")

    # 해당 초기각도 이상 단계만 실행
    all_stages = [(45, 500), (90, 1000), (135, 700), (180, 1000)]
    stages_to_run = [s for s in all_stages if s[0] >= init_angle_deg]

    label = f"init{init_angle_deg}"

    markings = load_markings()

    pinn = PINNTrainer()
    try:
        pinn.load("/home/jrkim/cartpole_project/pinn_model.pth")
        print("PINN 모델 로드 완료")
    except:
        print("PINN 모델 없음 → PINN 없이 실행")
        pinn = None

    actor  = SACActor()
    c1, c2 = SACCritic(), SACCritic()
    tc1, tc2 = SACCritic(), SACCritic()
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    a_opt  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    c1_opt = torch.optim.Adam(c1.parameters(),   lr=3e-4)
    c2_opt = torch.optim.Adam(c2.parameters(),   lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    al_opt    = torch.optim.Adam([log_alpha],     lr=3e-4)
    buf = []

    all_switched = []

    for angle, eps in stages_to_run:
        fname = f"stage_{angle}_{label}.pth"
        if os.path.exists(fname):
            actor.load_state_dict(torch.load(fname, weights_only=True))
            print(f"\n[{angle}도 단계] {fname} 불러옴 → 건너뜀")
            all_switched.extend([0]*eps)
            continue

        print(f"\n[{angle}도 단계] {eps}에피소드 학습 시작...")
        rate, switched = train_sac_stage(
            actor, c1, c2, tc1, tc2,
            a_opt, c1_opt, c2_opt, log_alpha, al_opt,
            buf, angle, eps,
            use_marking=True, markings=markings,
            use_ontology=True, label=label, pinn=pinn)
        all_switched.extend(switched)
        print(f"  [{angle}도 단계 완료] 성공률={rate:.1f}%")

    final_fname = f"curriculum_{label}.pth"
    torch.save(actor.state_dict(), final_fname)
    print(f"\n최종 저장: {final_fname}")
    return all_switched


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = {}
    for init_angle in [0, 135, 180]:
        sw = run_by_init_angle(init_angle)
        last = sw[-1000:] if len(sw) >= 1000 else sw
        rate = sum(last)/len(last)*100
        results[init_angle] = (sw, rate)
        print(f"\n>>> 초기각도 {init_angle}도 → 최종 성공률: {rate:.1f}%")

    # 결과 요약
    print("\n===== 최종 요약 =====")
    for angle, (sw, rate) in results.items():
        label = {0:"종말 조건 근처(0도)", 135:"중간 에너지(135도)", 180:"에너지 없음(180도)"}[angle]
        print(f"{label:25s} → 최종 성공률: {rate:.1f}%")

    # 그래프
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {0:'green', 135:'blue', 180:'red'}
    labels = {0:'0도 (직립 근처)', 135:'135도 (중간 에너지)', 180:'180도 (에너지 없음)'}
    w = 50
    for angle, (sw, _) in results.items():
        ma = [sum(sw[max(0,i-w):i+1])/min(i+1,w)*100 for i in range(len(sw))]
        ax.plot(range(1, len(ma)+1), ma,
                color=colors[angle], linewidth=2, label=labels[angle])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Switch Success Rate (%)')
    ax.set_title('초기 각도 조건별 성공률 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('init_angle_comparison.png', dpi=150)
    print("\n그래프 저장: init_angle_comparison.png")
