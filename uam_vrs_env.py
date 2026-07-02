"""
uam_vrs_env.py
==============
UAM (Urban Air Mobility) VRS (Vortex Ring State) 회피 시뮬레이터

CartPole에서 검증한 Context DB + PINN + SAC + Ontology 프레임워크를
UAM 도메인에 적용하기 위한 Gymnasium 커스텀 환경

[상태 공간] 4개
    V_x    : 전진 속도 (m/s)
    V_z    : 하강 속도 (m/s, 양수=하강)
    alt    : 고도 (m)
    T_norm : 정규화 추력 [0, 1]

[행동 공간] 1개 (연속)
    dT : 추력 변화량 [-1, 1]

[온톨로지]
    SAFE    : V_z/v_h < 0.5          → SAC 자유 학습
    CAUTION : 0.5 ≤ V_z/v_h < 0.8   → Context 보너스, 경고
    DANGER  : 0.8 ≤ V_z/v_h < 2.0   → 추력 강제 증가
    RECOVER : V_z/v_h ≥ 2.0         → 감속 유도
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp


class VRSEnv(gym.Env):
    """
    UAM VRS 회피 강화학습 환경
    CartPole CurriculumEnv와 동일한 구조, 물리식만 UAM으로 교체
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # =====================
        # UAM 물리 파라미터
        # =====================
        self.mass       = 10.0          # UAM 질량 (kg)
        self.rho        = 1.225         # 공기 밀도 (kg/m^3), 해수면 기준
        self.rotor_R    = 0.5           # 로터 반경 (m)
        self.rotor_A    = np.pi * self.rotor_R ** 2  # 로터 디스크 면적 (m^2)
        self.g          = 9.81          # 중력가속도 (m/s^2)

        self.T_max      = 2.0 * self.mass * self.g   # 최대 추력 (N), 호버링의 2배
        self.T_min      = 0.0           # 최소 추력 (N)

        # 호버링 추력 = mg
        self.T_hover    = self.mass * self.g  # 98.1 N

        # 적분 시간간격: 400Hz
        self.dt         = 1.0 / 400.0   # 0.0025 s

        # 에피소드 최대 스텝 (400Hz x 10초 = 4000 스텝)
        self.max_steps  = 4000

        # 커리큘럼 단계에서 외부에서 직접 바꿔가며 사용
        self.reset_alt_low  = 100.0
        self.reset_alt_high = 150.0


        # 종료 조건
        self.alt_min    = 0.0           # 착륙 고도 (m)
        self.alt_max    = 200.0         # 최대 고도 (m)
        self.V_z_crash  = 10.0          # 이 속도 이상으로 착륙하면 추락 (m/s)

        # =====================
        # VRS 온톨로지 경계값
        # =====================
        self.VRS_CAUTION = 0.5          # V_z/v_h 이상이면 CAUTION
        self.VRS_DANGER  = 0.8          # V_z/v_h 이상이면 DANGER
        self.VRS_RECOVER = 2.0          # V_z/v_h 이상이면 RECOVER

        # =====================
        # Gym 공간 정의
        # =====================
        # 관측 공간: [V_x, V_z, alt, T_norm]
        obs_low  = np.array([-20.0, -10.0,   0.0, 0.0], dtype=np.float32)
        obs_high = np.array([ 20.0,  10.0, 200.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 행동 공간: dT [-1, 1] 연속
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # =====================
        # 내부 상태 변수
        # =====================
        self.state      = None          # [V_x, V_z, alt, T_norm]
        self.step_count = 0             # 현재 스텝 수
        self.render_mode = render_mode

        # 렌더링용 기록 버퍼
        self.history_alt       = []
        self.history_vrs_ratio = []
        self.history_ontology  = []

    # =============================================
    # reset(): 에피소드 초기 상태 설정
    # =============================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 초기 상태 랜덤 샘플링
        V_x    = self.np_random.uniform(0.0, 5.0)    # 전진 속도 (m/s)
        V_z    = self.np_random.uniform(0.0, 3.0)    # 하강 속도 (m/s), 살짝 하강 중
        alt    = self.np_random.uniform(self.reset_alt_low, self.reset_alt_high) # 고도 (m)
        T_norm = 0.5                                  # 추력 50%로 시작

        self.state = np.array([V_x, V_z, alt, T_norm], dtype=np.float32)
        self.step_count = 0

        # 기록 버퍼 초기화
        self.history_alt       = [alt]
        self.history_vrs_ratio = [self._vrs_ratio(V_z, T_norm)]
        self.history_ontology  = [self._ontology(V_z, T_norm)]

        return self.state, {}

    # =============================================
    # step(): 한 타임스텝 진행
    # =============================================
    def step(self, action):
        V_x, V_z, alt, T_norm = self.state

        # 행동 클리핑 ([-1, 1] 범위 보장)
        dT = float(np.clip(action, -1.0, 1.0).flat[0])

        # ----------------------
        # 1. 온톨로지 분류
        # ----------------------
        zone = self._ontology(V_z, T_norm)

        # ----------------------
        # 2. 결정론적 규칙 적용 (DANGER, RECOVER)
        # ----------------------
        if zone == "DANGER":
            # VRS 위험 → 추력 강제 증가
            dT = 1.0
        elif zone == "RECOVER":
            # 과속 하강 → 추력 강제 증가 + 전진속도 감속
            dT = 1.0

        # ----------------------
        # 3. RK45로 동역학 전체 통합 적분
        # T_norm도 상태로 포함해 추력 변화까지 정확하게 적분
        # ----------------------
        result = solve_ivp(
            fun=lambda t, s: self._derivatives(s, dT),
            t_span=(0.0, self.dt),
            y0=[V_x, V_z, alt, T_norm],
            method='RK45',
        )
        V_x_new, V_z_new, alt_new, T_norm_new = result.y[:, -1]

        # 상태 클리핑
        V_x_new    = float(np.clip(V_x_new,    -20.0, 20.0))
        V_z_new    = float(np.clip(V_z_new,    -10.0, 10.0))
        alt_new    = float(np.clip(alt_new,      0.0, 200.0))
        T_norm_new = float(np.clip(T_norm_new,   0.0,   1.0))

        self.state = np.array([V_x_new, V_z_new, alt_new, T_norm_new], dtype=np.float32)
        self.step_count += 1

        # ----------------------
        # 5. VRS 비율 계산
        # ----------------------
        vrs_ratio = self._vrs_ratio(V_z_new, T_norm_new)
        zone_new  = self._ontology(V_z_new, T_norm_new)

        # ----------------------
        # 6. 보상 계산
        # ----------------------
        reward, terminated = self._compute_reward(
            V_z_new, alt_new, T_norm_new, vrs_ratio, zone_new
        )

        # 최대 스텝 초과 시 종료
        truncated = self.step_count >= self.max_steps

        # 기록 저장
        self.history_alt.append(alt_new)
        self.history_vrs_ratio.append(vrs_ratio)
        self.history_ontology.append(zone_new)

        info = {
            "vrs_ratio" : vrs_ratio,
            "zone"      : zone_new,
            "step"      : self.step_count,
        }

        return self.state, reward, terminated, truncated, info

    # =============================================
    # _derivatives(): UAM VRS 연속 동역학
    # RK45 적분기에서 호출 — 상태벡터 [V_x, V_z, alt, T_norm]의 시간미분 반환
    # =============================================
    def _derivatives(self, s, dT):
        _, V_z, _, T_norm = s  # V_x 고정, alt는 dalt/dt = -V_z 이므로 직접 불필요
        # 적분 중간에도 T_norm을 물리적 범위로 제한
        T_norm_c = float(np.clip(T_norm, 0.0, 1.0))
        T_actual = T_norm_c * self.T_max
        # 수직 가속도: 양수=상승, V_z 양수=하강이므로 부호 반전
        a_z = (T_actual - self.mass * self.g) / self.mass
        return [
            0.0,   # dV_x/dt: 단순화 모델에서 전진속도 변화 없음
            -a_z,  # dV_z/dt: 상승가속이면 하강속도 감소
            -V_z,  # dalt/dt: 하강속도만큼 고도 감소
            dT,    # dT_norm/dt: 액션이 추력 변화율 결정
        ]

    # =============================================
    # _vrs_ratio(): VRS 위험도 비율 계산
    # V_z / v_h
    # v_h = sqrt(T / 2ρA)  ← 호버링 유도 속도
    # =============================================
    def _vrs_ratio(self, V_z, T_norm):
        T_actual = T_norm * self.T_max

        # 추력이 0이면 v_h 계산 불가 → 매우 큰 값 반환 (RECOVER 처리)
        if T_actual <= 0:
            return 999.0

        # 호버링 유도 속도 v_h
        v_h = np.sqrt(T_actual / (2.0 * self.rho * self.rotor_A))

        # VRS 비율 (V_z 음수=상승이면 ratio 음수 → SAFE)
        ratio = V_z / v_h

        return float(ratio)

    # =============================================
    # _ontology(): 상태 분류
    # CartPole의 classify_state()와 동일한 구조
    # =============================================
    def _ontology(self, V_z, T_norm):
        ratio = self._vrs_ratio(V_z, T_norm)

        if ratio < self.VRS_CAUTION:
            return "SAFE"
        elif ratio < self.VRS_DANGER:
            return "CAUTION"
        elif ratio < self.VRS_RECOVER:
            return "DANGER"
        else:
            return "RECOVER"

    # =============================================
    # _compute_reward(): 보상 계산
    # =============================================
    def _compute_reward(self, V_z, alt, T_norm, vrs_ratio, zone):
        reward     = 0.0
        terminated = False

        # ----------------------
        # 기본 보상: 매 스텝 생존
        # ----------------------
        reward += 1.0

        # ----------------------
        # 고도 유지 보상: 목표 고도(0m)에 가까울수록 보상
        # ----------------------
        reward -= 0.01 * alt   # 고도가 낮을수록 착륙에 가까움 → 보상

        # ----------------------
        # VRS 구역별 패널티
        # ----------------------
        if zone == "CAUTION":
            reward -= 5.0
        elif zone == "DANGER":
            reward -= 20.0
        elif zone == "RECOVER":
            reward -= 10.0     # DANGER보다 낮음 (VRS는 벗어났으므로)

        # ----------------------
        # 안전 착륙 성공: alt ≈ 0, V_z 작을 때
        # ----------------------
        if alt <= 1.0 and V_z <= 2.0:
            reward += 100.0
            terminated = True  # 성공적 착륙

        # ----------------------
        # 실패 조건
        # ----------------------
        # 1. 추락: 너무 빠른 속도로 착지
        if alt <= 0.0 and V_z > self.V_z_crash:
            reward -= 50.0
            terminated = True

        # 2. 고도 초과
        if alt >= self.alt_max:
            reward -= 10.0
            terminated = True

        return reward, terminated

    # =============================================
    # render(): 에피소드 시각화
    # =============================================
    def render(self):
        if len(self.history_alt) < 2:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("UAM VRS Simulator — Episode Trajectory", fontsize=13)

        t = np.arange(len(self.history_alt)) * self.dt

        # 온톨로지 구간별 컬러맵
        zone_colors = {
            "SAFE"   : "#2ecc71",
            "CAUTION": "#f39c12",
            "DANGER" : "#e74c3c",
            "RECOVER": "#9b59b6",
        }

        # ---- 상단: 고도 변화 ----
        ax1 = axes[0]
        ax1.plot(t, self.history_alt, color="#2980b9", linewidth=1.5)
        ax1.set_ylabel("Altitude (m)")
        ax1.set_title("Altitude")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.grid(True, alpha=0.3)

        # ---- 하단: VRS ratio + 온톨로지 구간 색 ----
        ax2 = axes[1]
        ratios = self.history_vrs_ratio
        zones  = self.history_ontology

        # 구간별 배경색
        for i in range(len(t) - 1):
            ax2.axvspan(t[i], t[i+1],
                        color=zone_colors.get(zones[i], "white"),
                        alpha=0.3)

        ax2.plot(t, ratios, color="#2c3e50", linewidth=1.5, label="$V_z/v_h$")
        ax2.axhline(self.VRS_CAUTION, color="#f39c12", linestyle="--",
                    linewidth=0.8, label=f"CAUTION ({self.VRS_CAUTION})")
        ax2.axhline(self.VRS_DANGER,  color="#e74c3c", linestyle="--",
                    linewidth=0.8, label=f"DANGER ({self.VRS_DANGER})")
        ax2.axhline(self.VRS_RECOVER, color="#9b59b6", linestyle="--",
                    linewidth=0.8, label=f"RECOVER ({self.VRS_RECOVER})")
        ax2.set_ylabel("$V_z / v_h$")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("VRS Ratio & Ontology Zone")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 범례 패치
        patches = [mpatches.Patch(color=c, alpha=0.5, label=z)
                   for z, c in zone_colors.items()]
        ax2.legend(handles=patches, loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.show()


# =============================================
# 기본 동작 확인 테스트
# =============================================
if __name__ == "__main__":
    print("=== VRSEnv 기본 동작 테스트 ===\n")

    env = VRSEnv(render_mode="human")
    obs, _ = env.reset(seed=42)

    print(f"초기 상태:")
    print(f"  V_x    = {obs[0]:.3f} m/s")
    print(f"  V_z    = {obs[1]:.3f} m/s")
    print(f"  alt    = {obs[2]:.3f} m")
    print(f"  T_norm = {obs[3]:.3f}")
    print(f"  VRS ratio = {env._vrs_ratio(obs[1], obs[3]):.3f}")
    print(f"  Zone      = {env._ontology(obs[1], obs[3])}")
    print()

    # 랜덤 정책으로 100스텝 실행
    total_reward = 0.0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            print(f"  step {i:3d} | alt={obs[2]:.1f}m | "
                  f"V_z={obs[1]:.2f} | ratio={info['vrs_ratio']:.3f} | "
                  f"zone={info['zone']:8s} | reward={reward:.2f}")

        if terminated or truncated:
            print(f"\n  에피소드 종료 (step {i})")
            break

    print(f"\n총 보상: {total_reward:.2f}")
    print("\n렌더링 중...")
    env.render()
    env.close()
