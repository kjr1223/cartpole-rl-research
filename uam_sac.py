"""
uam_sac.py
==========
UAM VRS 회피를 위한 SAC (Soft Actor-Critic) 학습 에이전트

CartPole curriculum_pinn.py의 SAC 구조를 그대로 재사용,
환경만 VRSEnv로 교체

[구조]
    ReplayBuffer  : 경험 저장소
    SACActor      : 행동 결정 네트워크 (정책)
    SACCritic     : 가치 평가 네트워크 (Q함수)
    SACAgent      : 학습 총괄
"""

import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random

from uam_vrs_env import VRSEnv
from uam_pinn_model import UAMPINNTrainer


# =============================================
# Context DB (마킹 데이터) 관련 함수
# uam_marking_db.py(사람이 직접 마킹) + uam_marking_augment.py(perturbation 증강)로
# marking_data.db의 uam_switching_markings 테이블에 모아둔 데이터를 사용한다.
# CartPole curriculum_pinn.py의 load_markings()/get_bonus()와 동일한 아이디어.
# =============================================
def load_uam_markings():
    """
    uam_switching_markings 테이블에서 마킹된 상태(V_x,V_z,alt,T_norm)를 모두 불러온다.
    source가 manual이든 augmented든 구분 없이 다 가져온다 (둘 다 검증된 CAUTION 데이터).
    """
    conn = sqlite3.connect("marking_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT V_x, V_z, alt, T_norm FROM uam_switching_markings")
    rows = cursor.fetchall()
    conn.close()
    print(f"[Context DB] UAM 마킹 데이터 {len(rows)}개 불러옴")
    return np.array(rows, dtype=np.float32)


# 상태 변수마다 값의 범위가 크게 달라서(alt: 0~200 vs T_norm: 0~1) 그냥 유클리드
# 거리로 비교하면 alt 차이가 거리 계산을 거의 독점해버린다. 그래서 거리를 재기 전에
# 각 변수를 자기 범위로 나눠서 정규화한다 (대략적인 관측 공간 범위, observation_space와 동일).
STATE_SCALE = np.array([20.0, 10.0, 200.0, 1.0], dtype=np.float32)


def get_uam_bonus(state, markings, threshold=0.3, bonus=3.0):
    """
    현재 state가 마킹된 점들 중 가장 가까운 것과 (정규화된) 거리가
    threshold 이내면 보너스를 준다. CartPole의 get_bonus()와 동일한 아이디어.
    """
    if markings is None or len(markings) == 0:
        return 0.0
    norm_state    = np.asarray(state, dtype=np.float32) / STATE_SCALE
    norm_markings = markings / STATE_SCALE
    distances = np.linalg.norm(norm_markings - norm_state, axis=1)
    return bonus if np.min(distances) < threshold else 0.0


# =============================================
# 디바이스 설정 (GPU 있으면 GPU 사용)
# =============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


# =============================================
# ReplayBuffer: 경험 저장소
# (s, a, r, s', done) 튜플을 저장하고 랜덤 샘플링
# CartPole과 동일한 구조
# =============================================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================
# SACActor: 행동 결정 네트워크
# 상태 → 행동 분포 (평균, 표준편차) 출력
# CartPole과 동일: Linear(4→128)→ReLU→Linear(128→128)→ReLU
# =============================================
class SACActor(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()

        # 공유 레이어
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 평균 head
        self.mu      = nn.Linear(hidden, action_dim)
        # 표준편차 head (log scale)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        f = self.net(state)
        mu      = self.mu(f)
        log_std = self.log_std(f).clamp(-20, 2)  # 수치 안정성
        return mu, log_std

    def sample(self, state):
        """
        행동 샘플링 + 로그 확률 계산
        tanh squashing으로 [-1, 1] 범위로 압축
        """
        mu, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mu, std)
        x    = dist.rsample()               # reparameterization trick
        action = torch.tanh(x)              # [-1, 1] 범위로 압축

        # tanh squashing에 의한 로그 확률 보정
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# =============================================
# SACCritic: 가치 평가 네트워크 (Q함수)
# (상태, 행동) → Q값 출력
# 안정적인 학습을 위해 Q네트워크 2개 사용 (Double Q)
# =============================================
class SACCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()

        # Q1 네트워크
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Q2 네트워크 (Double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# =============================================
# SACAgent: 학습 총괄
# =============================================
class SACAgent:
    def __init__(
        self,
        state_dim   = 4,
        action_dim  = 1,
        hidden      = 128,
        lr          = 3e-4,       # 학습률
        gamma       = 0.99,       # 할인율
        tau         = 0.005,      # 타겟 네트워크 소프트 업데이트 비율
        alpha       = 0.2,        # 엔트로피 온도 (탐험 정도)
        batch_size  = 256,
        buffer_size = 100_000,
        auto_alpha  = True,       # 엔트로피 자동 조정
    ):
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        # 네트워크 초기화
        self.actor   = SACActor(state_dim, action_dim, hidden).to(device)
        self.critic  = SACCritic(state_dim, action_dim, hidden).to(device)

        # 타겟 크리틱 (소프트 업데이트용, 학습 안 함)
        self.critic_target = SACCritic(state_dim, action_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 옵티마이저
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # 엔트로피 자동 조정 (auto_alpha=True일 때)
        if self.auto_alpha:
            self.target_entropy = -action_dim          # 목표 엔트로피
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=device)
            # alpha 전용 학습률을 actor/critic보다 훨씬 작게 둔다 (lr의 1/10).
            # 이 환경은 보상 스케일이 수천 단위라 log_alpha가 조금만 빨리 움직여도
            # alpha(=exp(log_alpha))가 짧은 구간 안에 너무 커져버리는 걸 실제로 겪었음
            # (episode 60→70 사이에 0.11→5.8로 폭증). 천천히 움직이게 해서 완충한다.
            self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr * 0.1)

        # 리플레이 버퍼
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, deterministic=False):
        """
        행동 선택
        deterministic=True: 평가 시 사용 (탐험 없음)
        deterministic=False: 학습 시 사용 (탐험 있음)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(state_t)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state_t)
        return action.cpu().numpy().flatten()

    def update(self):
        """
        네트워크 업데이트 (배치 학습)
        버퍼에 충분한 데이터가 쌓이면 호출
        """
        if len(self.buffer) < self.batch_size:
            return None, None, None

        # 배치 샘플링
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # ----------------------
        # 1. Critic 업데이트
        # ----------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + self.gamma * (1 - done) * q_next

        q1, q2       = self.critic(state, action)
        critic_loss  = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----------------------
        # 2. Actor 업데이트
        # ----------------------
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new       = self.critic(state, new_action)
        q_new                = torch.min(q1_new, q2_new)
        actor_loss           = (self.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----------------------
        # 3. 엔트로피 자동 조정
        # ----------------------
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # log_alpha에 상한/하한이 없으면 alpha = exp(log_alpha)가 기하급수적으로
            # 폭주할 수 있다 (정책 엔트로피가 목표보다 낮은 상태가 계속되면 log_alpha가
            # 끝없이 커지는 악순환에 빠짐). SACActor의 log_std.clamp(-20,2)와 같은
            # 이유로 안전장치를 둔다.
            #
            # 처음엔 exp(-5)~exp(2) ≈ 0.0067~7.39로 잡았는데, 실제로 돌려보니
            # alpha가 상한(7.39)까지 차오르는 것만으로도 entropy 항이 Q(보상, 수천
            # 단위)를 압도해서 정책이 망가지고 보상이 -20000~-34000까지 무너졌다
            # (episode 70 근처에서 0.11→5.8로 급증, 이후 7.39에 고정).
            # 그래서 상한을 훨씬 낮춰서 exp(-5)~exp(-1) ≈ 0.0067~0.37로 제한한다.
            # (이전 검증에서 alpha가 0.11 이하일 때는 보상이 안정적이었음)
            self.log_alpha.data.clamp_(-5.0, -1.0)
            self.alpha = self.log_alpha.exp().item()

        # ----------------------
        # 4. 타겟 네트워크 소프트 업데이트
        # ----------------------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return (
            critic_loss.item(),
            actor_loss.item(),
            alpha_loss.item() if alpha_loss is not None else 0.0,
        )

    def save(self, path):
        torch.save({
            "actor"  : self.actor.state_dict(),
            "critic" : self.critic.state_dict(),
            "alpha"  : self.alpha,
        }, path)
        print(f"[Save] {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.alpha = ckpt["alpha"]
        print(f"[Load] {path}")


# =============================================
# 학습 루프
# =============================================
def train(
    curriculum = [
        # (alt_low, alt_high, max_steps, episodes)
        (20.0,  40.0,  3000, 200),
        (60.0,  90.0,  5000, 200),
        (100.0, 150.0, 8000, 600),
    ],
    eval_interval = 50,
    save_interval = 100,
    warmup_steps  = 1000,
    save_dir      = "./uam_checkpoints",
    use_pinn      = False,                        # True면 PINN으로 가상 transition을 만들어 버퍼에 추가
    pinn_path     = "./uam_pinn_model.pth",        # uam_pinn_model.py로 미리 학습해둔 PINN 가중치 경로
    n_imagined    = 3,                             # 실제 transition 1개당 PINN으로 만들 가상 transition 개수
    use_marking   = False,                         # True면 Context DB(마킹) 보너스 사용 (CAUTION 구역에서만)
    marking_threshold = 0.3,                       # get_uam_bonus의 정규화 거리 임계값
    marking_bonus     = 3.0,                       # 마킹 근처일 때 더해줄 보상 보너스
    fixed_alpha   = 0.05,                          # None이면 auto-alpha 사용(불안정 가능성 있었음).
                                                    # 값을 주면 alpha를 이 값으로 고정하고 자동조정을 끈다.
    buffer_size   = 50_000,                         # RAM 절약: 기본 100k → 50k
    log_file      = None,                          # 실시간 학습 로그 파일 경로 (None이면 stdout만)
):
    os.makedirs(save_dir, exist_ok=True)

    # 실시간 로그 파일 핸들 (None이면 stdout만 사용)
    _log_fh = open(log_file, 'w', buffering=1) if log_file else None

    def _log(msg):
        """stdout + 파일에 동시 기록 (buffering=1 → 줄 단위 즉시 flush)"""
        print(msg)
        if _log_fh:
            _log_fh.write(msg + '\n')

    env  = VRSEnv()

    # auto-alpha(엔트로피 자동조정)가 학습 중간에 alpha를 계속 키우다가 어느 시점에
    # 정책이 망가지는 폭주를 반복적으로 일으켜서(클램프 상한을 7.39→0.368로 낮춰도
    # 패턴이 똑같이 재현됨), 근본 원인을 더 깊게 진단하기 전까지는 고정 alpha를
    # 기본값으로 쓴다. fixed_alpha=None으로 호출하면 기존 auto-alpha 방식도 쓸 수 있다.
    if fixed_alpha is not None:
        agent = SACAgent(auto_alpha=False, alpha=fixed_alpha, buffer_size=buffer_size)
        print(f"[Alpha] 고정값 사용: alpha={fixed_alpha} (auto-alpha 비활성화)")
    else:
        agent = SACAgent(buffer_size=buffer_size)

    # ----------------------------------------
    # PINN 로드 (use_pinn=True일 때만)
    # PINN은 (state, action) → next_state를 예측하는 "가상 환경" 역할을 한다.
    # 실제 env.step() 없이도 가상의 경험을 만들어 버퍼에 채워서 데이터 효율을 높인다
    # (Dyna 스타일 모델 기반 강화학습, curriculum_pinn.py와 동일한 아이디어)
    # ----------------------------------------
    pinn = None
    if use_pinn:
        pinn = UAMPINNTrainer()
        try:
            pinn.load(pinn_path)
            print(f"[PINN] {pinn_path} 로드 완료 → 가상 transition 생성 활성화")
        except FileNotFoundError:
            print(f"[PINN] {pinn_path} 없음 → PINN 없이 학습 (먼저 uam_pinn_model.py를 실행해 학습해두세요)")
            pinn = None

    # ----------------------------------------
    # Context DB(마킹) 로드 (use_marking=True일 때만)
    # uam_marking_db.py(사람) + uam_marking_augment.py(증강)로 모아둔 CAUTION 구역
    # 마킹 데이터를 불러온다. DANGER/RECOVER는 온톨로지가 행동을 강제로 이미
    # 처리하고 있어서, 마킹 보너스는 CAUTION 구역에서만 적용한다.
    # ----------------------------------------
    markings = load_uam_markings() if use_marking else None

    # 기록용
    ep_rewards    = []
    eval_rewards  = []
    critic_losses = []
    actor_losses  = []

    total_steps = 0
    episode     = 0

    total_target_episodes = sum(c[3] for c in curriculum)

    _log(f"\n{'='*50}")
    _log(f"UAM VRS SAC 학습 시작")
    _log(f"총 스테이지: {len(curriculum)}개, 총 에피소드: {total_target_episodes}")
    _log(f"{'='*50}\n")

    for stage_idx, (alt_low, alt_high, max_steps, n_episodes) in enumerate(curriculum, start=1):
        env.reset_alt_low  = alt_low
        env.reset_alt_high = alt_high
        env.max_steps      = max_steps

        _log(f"\n[Curriculum Stage {stage_idx}] alt=({alt_low}, {alt_high})m, "
             f"max_steps={max_steps}, episodes={n_episodes}\n")

        for _ in range(n_episodes):
            episode += 1
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                # Warmup: 초반엔 랜덤 행동으로 버퍼 채우기
                if total_steps < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # ----------------------------------------
                # Context DB(마킹) 보너스
                # 행동을 적용하기 *전* obs가 CAUTION 구역에 있었고, 마킹된 점들과
                # (정규화 거리로) 가까우면 보상에 보너스를 더한다.
                # DANGER/RECOVER는 uam_vrs_env.py의 온톨로지가 행동을 강제로 이미
                # 처리하고 있어서 손대지 않고, CAUTION 구역만 보강한다.
                # ----------------------------------------
                if use_marking and markings is not None:
                    zone_before = env._ontology(obs[1], obs[3])
                    if zone_before in ("SAFE", "CAUTION"):  # SAFE 후반부(ratio≈0.4)도 포함 — 거리 조건이 2차 필터로 좁혀줌
                        reward += get_uam_bonus(obs, markings, marking_threshold, marking_bonus)

                # 버퍼에 경험 저장 (부트스트랩 차단은 진짜 종료(terminated)일 때만)
                agent.buffer.push(obs, action, reward, next_obs, float(terminated))

                # ----------------------------------------
                # PINN으로 가상 transition 추가 (Dyna 스타일)
                # 실제 환경을 한 번 더 진행시키는 대신, 같은 obs에서 정책이 고를 만한
                # 다른 행동(a_img)들을 PINN에 넣어 가상의 다음 상태(ns_img)를 예측하고,
                # 그 가상 상태에 대해 실제 보상 함수(env._compute_reward)를 그대로 적용해서
                # 버퍼에 추가한다. 실제 시뮬레이션 스텝 없이 데이터를 더 많이 확보하는 효과.
                # ----------------------------------------
                if pinn is not None:
                    # uam_pinn_model.py가 PINN을 학습시킬 때, DANGER/RECOVER 구역에서는
                    # 입력 행동을 항상 1.0(온톨로지 강제 override)으로 고정해서 데이터를
                    # 모았다. 그래서 PINN은 "DANGER/RECOVER 구역 + 행동=1.0 이외의 조합"을
                    # 학습 중 한 번도 본 적이 없다. 여기서 정책이 고른 임의의 행동을 그대로
                    # 넣으면 PINN이 한 번도 안 본 입력에 대해 예측하게 되어 결과가 엉터리로
                    # 나올 수 있다 (실제로 Critic 발산의 원인으로 확인됨). 그래서 PINN에
                    # 넣기 전에 실제 env.step()과 똑같은 온톨로지 override를 먼저 적용한다.
                    zone_obs = env._ontology(obs[1], obs[3])
                    for _ in range(n_imagined):
                        a_img_raw = agent.select_action(obs)          # 현재 정책으로 가상 행동 샘플링
                        if zone_obs in ("DANGER", "RECOVER"):
                            a_img = np.array([1.0], dtype=np.float32)  # PINN 학습 시와 동일한 override
                        else:
                            a_img = a_img_raw
                        ns_img = pinn.predict(obs, float(a_img[0]))   # PINN으로 가상 다음 상태 예측
                        # PINN 예측이 관측 공간 범위를 벗어날 수 있으니 클리핑
                        ns_img = np.clip(
                            ns_img, env.observation_space.low, env.observation_space.high
                        )

                        # 가상 상태에 대한 VRS 비율/온톨로지 구역을 계산해서
                        # 실제 환경과 똑같은 보상 함수로 보상/종료 여부를 구한다
                        vrs_ratio_img = env._vrs_ratio(ns_img[1], ns_img[3])
                        zone_img      = env._ontology(ns_img[1], ns_img[3])
                        r_img, term_img = env._compute_reward(
                            ns_img[1], ns_img[2], ns_img[3], vrs_ratio_img, zone_img
                        )

                        agent.buffer.push(obs, a_img, r_img, ns_img, float(term_img))

                obs        = next_obs
                ep_reward += reward
                total_steps += 1

                # 버퍼 충분하면 업데이트
                if total_steps >= warmup_steps:
                    c_loss, a_loss, _ = agent.update()
                    if c_loss is not None:
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            ep_rewards.append(ep_reward)

            # 진행상황 출력
            if episode % 10 == 0:
                avg = np.mean(ep_rewards[-10:])
                _log(f"Episode {episode:4d} | Avg Reward (10): {avg:8.2f} | "
                     f"Buffer: {len(agent.buffer):6d} | Alpha: {agent.alpha:.4f}")

            # 평가 (결정론적 정책으로)
            if episode % eval_interval == 0:
                eval_r = evaluate(agent, env, n_episodes=5)
                eval_rewards.append((episode, eval_r))
                _log(f"\n[Eval] Episode {episode} | Eval Reward: {eval_r:.2f}\n")

            # 체크포인트 저장
            if episode % save_interval == 0:
                agent.save(os.path.join(save_dir, f"uam_sac_ep{episode}.pth"))

    # 최종 모델 저장
    agent.save(os.path.join(save_dir, "uam_sac_final.pth"))

    # 학습 곡선 시각화
    plot_training(ep_rewards, eval_rewards, critic_losses, actor_losses)

    if _log_fh:
        _log_fh.close()

    env.close()
    return agent


def evaluate(agent, env, n_episodes=5):
    """결정론적 정책으로 평가"""
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / n_episodes


def plot_training(ep_rewards, eval_rewards, critic_losses, actor_losses):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("UAM VRS SAC Training Curves", fontsize=13)

    # 에피소드 보상
    axes[0, 0].plot(ep_rewards, alpha=0.4, color="#2980b9", label="Episode Reward")
    # 이동평균
    if len(ep_rewards) >= 10:
        ma = np.convolve(ep_rewards, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(ep_rewards)), ma, color="#e74c3c", label="MA(10)")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 평가 보상
    if eval_rewards:
        eps, rews = zip(*eval_rewards)
        axes[0, 1].plot(eps, rews, color="#2ecc71", marker='o')
    axes[0, 1].set_title("Eval Reward (Deterministic)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, alpha=0.3)

    # Critic Loss
    axes[1, 0].plot(critic_losses, alpha=0.6, color="#e67e22")
    axes[1, 0].set_title("Critic Loss")
    axes[1, 0].set_xlabel("Update Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Actor Loss
    axes[1, 1].plot(actor_losses, alpha=0.6, color="#9b59b6")
    axes[1, 1].set_title("Actor Loss")
    axes[1, 1].set_xlabel("Update Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "uam_training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Save] {out_path}")
    plt.close()   # headless 환경에서 plt.show()는 hang → close로 대체


# =============================================
# 실행
# =============================================
if __name__ == "__main__":
    import os as _os
    _base = _os.path.dirname(_os.path.abspath(__file__))
    agent = train(
        eval_interval  = 100,
        save_interval  = 100,
        warmup_steps   = 1000,
        save_dir       = _os.path.join(_base, "uam_checkpoints_v2"),  # 이전 run과 분리
        buffer_size    = 50_000,
        log_file       = _os.path.join(_base, "uam_training_v2.log"), # 실시간 로그
        use_pinn       = False,
        use_marking    = False,
    )
