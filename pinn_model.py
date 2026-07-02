import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# CartPole PINN 동역학 모델
# 물리 법칙 + 신경망으로 동역학 학습
# ================================

# 물리 상수
GRAVITY    = 9.8
MASS_CART  = 1.0
MASS_POLE  = 0.1
LENGTH     = 0.5
DT         = 0.02
FORCE_MAG  = 10.0

class CartPolePINN(nn.Module):
    """
    PINN 기반 CartPole 동역학 모델
    입력: state(4) + action(1) = 5
    출력: next_state(4)
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
        if action.dim() == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        delta = self.net(x)
        # residual connection: 현재 state + 변화량
        next_state = state + delta
        return next_state


def physics_residual(state, action, next_state_pred):
    """
    라그랑지안 동역학 잔차 계산
    PINN의 핵심: 물리 법칙을 손실함수에 반영
    
    실제 동역학:
      th_acc = (g*sin - cos*tmp) / (L*(4/3 - m*cos^2/mt))
      x_acc  = tmp - m*L*th_acc*cos/mt
    
    잔차 = 신경망 예측 - 물리 법칙 예측
    """
    x       = state[:, 0:1]
    x_dot   = state[:, 1:2]
    theta   = state[:, 2:3]
    th_dot  = state[:, 3:4]
    force   = action * FORCE_MAG

    mt  = MASS_CART + MASS_POLE
    ml  = MASS_POLE * LENGTH
    c   = torch.cos(theta)
    s   = torch.sin(theta)

    tmp    = (force + ml * th_dot**2 * s) / mt
    th_acc = (GRAVITY*s - c*tmp) / \
             (LENGTH * (4/3 - MASS_POLE * c**2 / mt))
    x_acc  = tmp - ml * th_acc * c / mt

    # 물리 법칙으로 예측한 next_state
    next_x      = x     + DT * x_dot
    next_xdot   = x_dot + DT * x_acc
    next_theta  = theta + DT * th_dot
    next_thdot  = th_dot + DT * th_acc
    next_physics = torch.cat([next_x, next_xdot,
                               next_theta, next_thdot], dim=1)

    # 잔차: 신경망 예측 vs 물리 예측
    residual = next_state_pred - next_physics
    return residual, next_physics


class PINNTrainer:
    """
    PINN 학습 관리
    데이터 손실 + 물리 잔차 손실을 합쳐서 학습
    """
    def __init__(self, lr=1e-3, lambda_physics=1.0):
        self.model = CartPolePINN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_physics = lambda_physics  # 물리 손실 가중치
        self.losses = []

    def compute_loss(self, states, actions, next_states):
        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)

        # 신경망 예측
        pred = self.model(states, actions)

        # 1. 데이터 손실: 실제 next_state와의 차이
        data_loss = nn.MSELoss()(pred, next_states)

        # 2. 물리 잔차 손실: 라그랑지안 동역학과의 차이
        residual, _ = physics_residual(states, actions, pred)
        physics_loss = (residual**2).mean()

        # 전체 손실 = 데이터 손실 + 물리 손실
        total_loss = data_loss + self.lambda_physics * physics_loss
        return total_loss, data_loss, physics_loss

    def train_step(self, states, actions, next_states):
        self.optimizer.zero_grad()
        loss, dl, pl = self.compute_loss(states, actions, next_states)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item(), dl.item(), pl.item()

    def predict(self, state, action):
        """단일 state, action → next_state 예측"""
        self.model.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.FloatTensor([action])
            pred = self.model(s, a)
        self.model.train()
        return pred.squeeze(0).numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


# ================================
# 테스트: PINN이 CartPole 동역학을
# 제대로 학습하는지 확인
# ================================
if __name__ == "__main__":
    import gymnasium as gym

    print("=== PINN 동역학 모델 테스트 ===")

    # 1. 데이터 수집
    print("1. CartPole 환경에서 데이터 수집 중...")
    env = gym.make("CartPole-v1")
    states, actions, next_states = [], [], []

    for ep in range(100):
        obs, _ = env.reset()
        # 스윙업용 초기 각도 다양화
        obs[2] = np.random.uniform(-np.pi, np.pi)
        for _ in range(200):
            action = np.random.uniform(-1, 1)
            states.append(obs.copy())
            actions.append([action])
            # 물리 모델로 next_state 계산
            x, xd, th, thd = obs
            force = action * FORCE_MAG
            mt = MASS_CART + MASS_POLE
            ml = MASS_POLE * LENGTH
            c, s = np.cos(th), np.sin(th)
            tmp    = (force + ml*thd**2*s) / mt
            th_acc = (GRAVITY*s - c*tmp) / \
                     (LENGTH*(4/3 - MASS_POLE*c**2/mt))
            x_acc  = tmp - ml*th_acc*c/mt
            next_obs = np.array([
                x + DT*xd,
                xd + DT*x_acc,
                th + DT*thd,
                thd + DT*th_acc
            ], dtype=np.float32)
            next_states.append(next_obs)
            obs = next_obs
    env.close()

    states      = np.array(states)
    actions     = np.array(actions)
    next_states = np.array(next_states)
    print(f"   수집된 데이터: {len(states)}개")

    # 2. PINN 학습
    print("2. PINN 학습 중...")
    trainer = PINNTrainer(lr=3e-4, lambda_physics=0.5)
    batch = 256
    n = len(states)

    for epoch in range(2000):
        idx = np.random.choice(n, batch)
        loss, dl, pl = trainer.train_step(
            states[idx], actions[idx], next_states[idx]
        )
        if (epoch+1) % 100 == 0:
            print(f"   Epoch {epoch+1:4d} | "
                  f"Total={loss:.5f} | "
                  f"Data={dl:.5f} | "
                  f"Physics={pl:.5f}")

    # 3. 예측 정확도 확인
    print("3. 예측 정확도 확인...")
    test_state  = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)
    test_action = 0.5
    pred = trainer.predict(test_state, test_action)
    print(f"   입력 state:  {test_state}")
    print(f"   예측 next:   {pred}")

    # 물리 모델 직접 계산
    x, xd, th, thd = test_state
    force = test_action * FORCE_MAG
    mt = MASS_CART + MASS_POLE
    ml = MASS_POLE * LENGTH
    c, s = np.cos(th), np.sin(th)
    tmp    = (force + ml*thd**2*s) / mt
    th_acc = (GRAVITY*s - c*tmp) / \
             (LENGTH*(4/3 - MASS_POLE*c**2/mt))
    x_acc  = tmp - ml*th_acc*c/mt
    real = np.array([x+DT*xd, xd+DT*x_acc,
                     th+DT*thd, thd+DT*th_acc])
    print(f"   실제 next:   {real}")
    print(f"   오차:        {np.abs(pred-real)}")

    trainer.save("/home/jrkim/cartpole_project/pinn_model.pth")
    print("\n=== 완료! pinn_model.pth 저장됨 ===")
