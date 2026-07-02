"""
uam_pinn_model.py
==================
UAM 로터 동역학을 학습하는 PINN(Physics-Informed Neural Network) 모델

CartPole에서 검증한 pinn_model.py와 똑같은 구조를 그대로 재사용한다.
다른 점은 "물리 법칙"으로 라그랑지안 동역학(CartPole) 대신
uam_vrs_env.py의 모멘텀 이론 적분식(추력→가속도→속도→고도)을 쓴다는 것뿐이다.

PINN이 필요한 이유:
    일반 신경망은 (state, action) → next_state를 데이터만 보고 학습하지만,
    물리 법칙(여기서는 모멘텀 이론 운동방정식)을 손실함수에 같이 넣어주면
    데이터가 적어도 실제 동역학에 더 가깝게, 더 안정적으로 학습된다.
    학습이 끝나면 이 모델로 가상의 (s, a, r, s') 경험을 만들어서
    SAC 리플레이 버퍼에 추가할 수 있다 (Dyna 스타일 모델 기반 강화학습).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# UAM 물리 상수
# uam_vrs_env.py의 VRSEnv와 반드시 동일한 값을 써야 한다.
# (물리 잔차 손실이 비교 기준으로 쓰는 식이라 값이 다르면 잘못된 잔차가 나옴)
# ================================
MASS    = 10.0                    # UAM 질량 (kg)
GRAVITY = 9.81                    # 중력가속도 (m/s^2)
T_MAX   = 2.0 * MASS * GRAVITY    # 최대 추력 (N) = 호버링 추력의 2배
DT      = 1.0 / 400.0             # 적분 시간간격 (s), 400Hz


class UAMPINN(nn.Module):
    """
    PINN 기반 UAM 로터 동역학 모델

    입력: state(4) + action(1) = 5
        state  = [V_x, V_z, alt, T_norm]  (전진속도, 하강속도, 고도, 정규화 추력)
        action = dT                       (추력 변화량, [-1, 1])
    출력: next_state(4) = [V_x', V_z', alt', T_norm']

    신경망 구조: Linear(5→128) → Tanh → Linear(128→128) → Tanh → Linear(128→4)
    CartPole PINN과 동일하게 Tanh를 썼다 (ReLU보다 매끄러운 동역학 함수 근사에 적합).
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4)
        )

    def forward(self, state, action):
        # action이 (batch,) 형태(1차원)로 들어올 수도 있어서, (batch,1)로 맞춰줌
        if action.dim() == 1:
            action = action.unsqueeze(1)

        # state(4)와 action(1)을 이어붙여서 신경망 입력(5)을 만든다
        x = torch.cat([state, action], dim=1)

        # 신경망은 "다음 상태 자체"가 아니라 "변화량(delta)"을 예측한다.
        # next_state = state + delta 형태의 residual connection을 쓰면
        # 변화량이 보통 작기 때문에 학습이 더 쉽고 안정적이다.
        delta = self.net(x)
        next_state = state + delta
        return next_state


def physics_residual(state, action, next_state_pred):
    """
    모멘텀 이론 동역학 잔차 계산

    uam_vrs_env.py의 step()에 있는 적분식을 "물리 법칙"으로 그대로 가져와서,
    신경망이 예측한 next_state가 이 물리 법칙과 얼마나 다른지(잔차)를 계산한다.
    이 잔차를 손실함수에 더해주면 신경망이 물리적으로 말이 되는 방향으로 학습된다.

    실제 동역학 (uam_vrs_env.py step()과 동일):
        T_norm_new = clip(T_norm + dT * dt, 0, 1)   # 추력 업데이트
        T_actual   = T_norm_new * T_MAX             # 정규화 추력 → 실제 추력(N)
        a_z        = (T_actual - m*g) / m           # 수직 가속도 (양수=상승가속)
        V_z_new    = V_z - a_z * dt                 # 하강속도 업데이트
        alt_new    = alt - V_z * dt                 # 고도 업데이트 (주의: 갱신 *전* V_z 사용!)
        V_x_new    = V_x                            # 전진속도는 단순화를 위해 변화 없음

    인자:
        state           : (batch, 4) 텐서, [V_x, V_z, alt, T_norm]
        action          : (batch, 1) 텐서, dT
        next_state_pred : 신경망이 예측한 (batch, 4) next_state

    반환:
        residual      : 신경망 예측 - 물리 법칙 예측 (이 값이 작아지도록 학습시킴)
        next_physics  : 물리 법칙만으로 계산한 next_state (참고/검증용)
    """
    # state를 각 물리량으로 분리 (슬라이싱으로 (batch,1) 모양 유지)
    V_x    = state[:, 0:1]
    V_z    = state[:, 1:2]
    alt    = state[:, 2:3]
    T_norm = state[:, 3:4]
    dT     = action

    # 1. 추력 업데이트: 정규화 추력에 dT*dt만큼 더하고 [0,1]로 클리핑
    T_norm_new = torch.clamp(T_norm + dT * DT, 0.0, 1.0)
    T_actual   = T_norm_new * T_MAX

    # 2. 수직 가속도: a_z = (실제추력 - 중력) / 질량
    #    T_actual > mg 이면 a_z > 0 (상승 가속, 하강속도 감소)
    a_z = (T_actual - MASS * GRAVITY) / MASS

    # 3. 하강속도 업데이트 (V_z는 양수=하강이라서 상승가속이면 빼줌)
    V_z_new = V_z - a_z * DT

    # 4. 고도 업데이트: env.step()과 똑같이 "갱신 전" V_z를 사용한다.
    #    (V_z_new를 쓰면 env의 실제 적분 순서와 달라져서 잔차가 부정확해짐)
    alt_new = alt - V_z * DT

    # 5. 전진속도: 이 환경에서는 단순화를 위해 변화 없음
    V_x_new = V_x

    next_physics = torch.cat([V_x_new, V_z_new, alt_new, T_norm_new], dim=1)

    # 잔차 = 신경망 예측 - 물리 법칙 예측. 0에 가까울수록 물리적으로 정확한 예측.
    residual = next_state_pred - next_physics
    return residual, next_physics


class UAMPINNTrainer:
    """
    UAM PINN 학습을 총괄하는 클래스.
    "데이터 손실"(실제 관측된 next_state와의 차이)과
    "물리 손실"(물리 법칙과의 차이, physics_residual)을 더해서 함께 학습시킨다.
    """
    def __init__(self, lr=1e-3, lambda_physics=1.0):
        self.model = UAMPINN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # lambda_physics: 물리 손실에 곱하는 가중치.
        # 크게 하면 물리 법칙을 더 강하게 따르고, 작게 하면 데이터에 더 의존하게 됨.
        self.lambda_physics = lambda_physics
        self.losses = []   # 학습 곡선 확인용 손실 기록

    def compute_loss(self, states, actions, next_states):
        """
        한 배치(batch)에 대해 전체 손실(total_loss)과 세부 손실(data_loss, physics_loss)을 계산.
        numpy 배열을 받아서 내부에서 torch 텐서로 변환한다.
        """
        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)

        # 신경망으로 next_state 예측
        pred = self.model(states, actions)

        # 1. 데이터 손실: 신경망 예측이 실제로 관측된 next_state와 얼마나 다른지 (MSE)
        data_loss = nn.MSELoss()(pred, next_states)

        # 2. 물리 잔차 손실: 신경망 예측이 모멘텀 이론 동역학과 얼마나 다른지
        residual, _ = physics_residual(states, actions, pred)
        physics_loss = (residual ** 2).mean()

        # 전체 손실 = 데이터 손실 + 가중치 * 물리 손실
        total_loss = data_loss + self.lambda_physics * physics_loss
        return total_loss, data_loss, physics_loss

    def train_step(self, states, actions, next_states):
        """한 번의 배치 학습(forward → loss 계산 → backward → 파라미터 업데이트)"""
        self.optimizer.zero_grad()
        loss, dl, pl = self.compute_loss(states, actions, next_states)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item(), dl.item(), pl.item()

    def predict(self, state, action):
        """
        단일 (state, action) 하나에 대해 next_state를 예측.
        SAC 학습 루프에서 가상 transition을 만들 때 이 함수를 사용한다.
        """
        self.model.eval()        # 평가 모드 (dropout/batchnorm 없지만 관례상 명시)
        with torch.no_grad():    # 예측만 할 거라 그래디언트 계산 불필요 → 메모리/속도 절약
            s = torch.FloatTensor(state).unsqueeze(0)        # (4,) → (1,4)
            a = torch.FloatTensor([action]).reshape(1, 1)    # 스칼라 action → (1,1)
            pred = self.model(s, a)
        self.model.train()       # 다시 학습 모드로 복귀 (이후 추가 학습 가능하도록)
        return pred.squeeze(0).numpy()   # (1,4) → (4,) numpy 배열로 변환해서 반환

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


# =============================================
# 테스트: PINN이 UAM 로터 동역학을
# 제대로 학습하는지 확인
# =============================================
if __name__ == "__main__":
    from uam_vrs_env import VRSEnv

    print("=== UAM PINN 동역학 모델 테스트 ===")

    # ----------------------------------------
    # 1. 데이터 수집
    # VRSEnv를 랜덤 정책으로 굴려서 (state, action, next_state) 쌍을 모은다.
    # 이 데이터로 PINN을 "오프라인"으로 미리 학습시켜 두는 것.
    # ----------------------------------------
    print("1. VRSEnv에서 데이터 수집 중...")
    env = VRSEnv()
    states, actions, next_states = [], [], []

    for ep in range(200):
        obs, _ = env.reset()
        for _ in range(200):
            raw_action = np.random.uniform(-1.0, 1.0)

            # 주의: uam_vrs_env.py의 step()은 온톨로지가 DANGER/RECOVER 구역이면
            # 내가 넣은 action을 무시하고 dT=1.0으로 강제 덮어쓴다.
            # 그래서 실제 물리 업데이트에 쓰인 "진짜 dT"를 미리 계산해서 기록해야
            # PINN이 (state, action) → next_state를 올바르게 학습할 수 있다.
            # (raw_action을 그대로 기록하면 DANGER/RECOVER에서는 입력과 결과가 안 맞게 됨)
            V_x, V_z, alt, T_norm = obs
            zone = env._ontology(V_z, T_norm)
            applied_dT = 1.0 if zone in ("DANGER", "RECOVER") else raw_action

            states.append(obs.copy())
            actions.append([applied_dT])

            # 환경에는 raw_action을 그대로 넘긴다 (온톨로지 덮어쓰기는 step() 내부에서 처리됨)
            next_obs, reward, terminated, truncated, info = env.step(np.array([raw_action]))
            next_states.append(next_obs.copy())
            obs = next_obs

            if terminated or truncated:
                break
    env.close()

    states      = np.array(states)
    actions     = np.array(actions)
    next_states = np.array(next_states)
    print(f"   수집된 데이터: {len(states)}개")

    # ----------------------------------------
    # 2. PINN 학습
    # 수집한 데이터에서 매 epoch마다 무작위로 배치(256개)를 뽑아서 학습한다.
    # ----------------------------------------
    print("2. PINN 학습 중...")
    trainer = UAMPINNTrainer(lr=3e-4, lambda_physics=0.5)
    batch = 256
    n = len(states)

    for epoch in range(2000):
        idx = np.random.choice(n, batch)   # 전체 데이터 중 배치 크기만큼 무작위 샘플링
        loss, dl, pl = trainer.train_step(
            states[idx], actions[idx], next_states[idx]
        )
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:4d} | "
                  f"Total={loss:.5f} | "
                  f"Data={dl:.5f} | "
                  f"Physics={pl:.5f}")

    # ----------------------------------------
    # 3. 예측 정확도 확인
    # 학습이 끝난 PINN의 예측값을 실제 환경의 결과값과 직접 비교해본다.
    # ----------------------------------------
    print("3. 예측 정확도 확인...")
    test_state  = np.array([3.0, 1.0, 100.0, 0.5], dtype=np.float32)
    test_action = 0.2
    pred = trainer.predict(test_state, test_action)
    print(f"   입력 state: {test_state}")
    print(f"   예측 next : {pred}")

    # 같은 state에서 실제 환경을 한 스텝 진행시켜서 "정답" next_state를 얻는다
    test_env = VRSEnv()
    test_env.reset()
    test_env.state = test_state.copy()
    real_next, _, _, _, _ = test_env.step(np.array([test_action]))
    print(f"   실제 next : {real_next}")
    print(f"   오차      : {np.abs(pred - real_next)}")

    # 학습된 모델 가중치 저장 → 이후 uam_sac.py 학습 루프에서 load해서 사용
    trainer.save("/home/jrkim/cartpole_project/uam_pinn_model.pth")
    print("\n=== 완료! uam_pinn_model.pth 저장됨 ===")
