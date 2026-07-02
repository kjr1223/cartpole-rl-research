"""
finetune_180.py
===============
stage_135_angle135.pth를 베이스로
impulse 환경에서 180도 파인튜닝

[방법]
    1. stage_135_angle135.pth 불러오기
    2. CurriculumEnv(start_angle_deg=180, use_impulse=True)로 학습
    3. 결과 저장: stage_180_impulse.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sqlite3

from curriculum_pinn import (
    CurriculumEnv, SACActor, SACCritic,
    classify_state, deterministic_action,
    load_markings, get_bonus
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# =============================================
# 파인튜닝 설정
# =============================================
BASE_MODEL   = "stage_180_impulse_ep400.pth"   # 베이스 모델
SAVE_PATH    = "stage_180_impulse.pth"    # 저장 경로
EPISODES     = 2000                         # 파인튜닝 에피소드 수
LR           = 1e-4                        # 학습률 (파인튜닝이라 작게)
GAMMA        = 0.99
TAU          = 0.005

# =============================================
# 네트워크 초기화 + 베이스 모델 로드
# =============================================
actor  = SACActor()
c1     = SACCritic()
c2     = SACCritic()
tc1    = SACCritic()
tc2    = SACCritic()

# 베이스 모델 로드
ckpt = torch.load(BASE_MODEL, map_location="cpu")
actor.load_state_dict(ckpt)
print(f"[Load] {BASE_MODEL}")

# Critic은 랜덤 초기화 (새 환경이라 Q값 재학습)
tc1.load_state_dict(c1.state_dict())
tc2.load_state_dict(c2.state_dict())

# 옵티마이저
a_opt   = torch.optim.Adam(actor.parameters(),  lr=LR)
c1_opt  = torch.optim.Adam(c1.parameters(),     lr=LR)
c2_opt  = torch.optim.Adam(c2.parameters(),     lr=LR)

log_alpha = torch.zeros(1, requires_grad=True)
al_opt    = torch.optim.Adam([log_alpha], lr=LR)

# =============================================
# 마킹 데이터 로드 (있으면)
# =============================================
try:
    markings = load_markings()
    use_marking = True
    print(f"[Marking] {len(markings)}개 로드")
except Exception as e:
    markings    = None
    use_marking = False
    print(f"[Marking] 없음 ({e})")

# =============================================
# 파인튜닝 루프
# =============================================
env = CurriculumEnv(start_angle_deg=180, use_impulse=True)
buf = []
all_switched = []

print(f"\n{'='*50}")
print(f"180° 파인튜닝 시작 (베이스: {BASE_MODEL})")
print(f"총 에피소드: {EPISODES}")
print(f"{'='*50}\n")

for ep in range(EPISODES):
    obs  = env.reset()
    done = False
    switched = False

    while not done:
        # 온톨로지 분류
        situation = classify_state(obs)
        angle_deg = abs(np.degrees(obs[2]))

        if situation == "Danger" or angle_deg > 135:
            av = deterministic_action(obs)
        else:
            with torch.no_grad():
                a, _ = actor.get_action(
                    torch.FloatTensor(obs).unsqueeze(0), with_logprob=True)
            av = a.squeeze().item()

        no, r, term, just_switched = env.step(av)
        done = term

        # 마킹 보너스
        if use_marking and env.mode == "swingup":
            situation = classify_state(obs)
            if situation == "Caution":
                r += get_bonus(obs, markings)

        if just_switched:
            switched = True
            r += 5.0

        buf.append((obs, av, r, no, float(done)))
        if len(buf) > 20000:
            buf.pop(0)

        # SAC 업데이트
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
                tv = r_ + GAMMA * (1 - d_) * (
                    torch.min(tc1(ns, na), tc2(ns, na)) - log_alpha.exp() * nlp)

            c1_opt.zero_grad(); F.mse_loss(c1(s, a_), tv).backward(); c1_opt.step()
            c2_opt.zero_grad(); F.mse_loss(c2(s, a_), tv).backward(); c2_opt.step()

            na2, nlp2 = actor.get_action(s, with_logprob=True)
            q = torch.min(c1(s, na2), c2(s, na2))
            al = (log_alpha.exp() * nlp2 - q).mean()
            a_opt.zero_grad(); al.backward(); a_opt.step()

            al2 = -(log_alpha * (nlp2 + -1.0).detach()).mean()
            al_opt.zero_grad(); al2.backward(); al_opt.step()

            for p, tp in zip(c1.parameters(), tc1.parameters()):
                tp.data.copy_(TAU * p + (1 - TAU) * tp)
            for p, tp in zip(c2.parameters(), tc2.parameters()):
                tp.data.copy_(TAU * p + (1 - TAU) * tp)

        obs = no

    all_switched.append(1 if switched else 0)

    if (ep + 1) % 50 == 0:
        rate = sum(all_switched[-100:]) / min(len(all_switched), 100) * 100
        print(f"  ep={ep+1:4d} | 전환성공률={rate:.0f}%")

    # 중간 저장 (100에피소드마다)
    if (ep + 1) % 100 == 0:
        torch.save(actor.state_dict(), f"stage_180_impulse_ep{ep+1}.pth")
        print(f"  [Save] stage_180_impulse_ep{ep+1}.pth")

env.close()

# 최종 저장
torch.save(actor.state_dict(), SAVE_PATH)
print(f"\n[Save] {SAVE_PATH}")
print("파인튜닝 완료!")
