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
        self.alt_min    = 0.0           # 착륙 고도 (m). 이 고도 이하면 착지
        self.alt_max    = 200.0         # 최대 고도 (m). 이 고도 이상이면 너무 높이 올라간 것
        self.V_z_crash  = 10.0          # 이 속도 이상으로 착륙하면 추락 (m/s)

        # =====================
        # VRS 온톨로지 경계값
        # =====================
        self.VRS_CAUTION = 0.5          # V_z/v_h 이상이면 CAUTION
        self.VRS_DANGER  = 0.8          # V_z/v_h 이상이면 DANGER
        self.VRS_RECOVER = 2.0          # V_z/v_h 이상이면 RECOVER

        # =====================
        # 방법론별 플래그
        # =====================
        # zone 분류(_ontology)는 항상 실행 — 평가 로깅(VRS 진입률 등)에 필요
        # 플래그는 "개입(dT 강제)" 여부와 "VRS 패널티 포함" 여부만 제어
        self.use_ontology_override = True   # DANGER/RECOVER → dT=1.0 강제 (방법론2,3: True / 방법론0,1: False)
        self.use_vrs_penalty       = True   # VRS 이차 패널티 항 포함 여부 (방법론1,3: True / 방법론0,2: False)
        self.use_time_penalty      = True   # 시간 패널티(-0.01/step) 포함 여부 (방법론0-a: False / 나머지: True)

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
    def reset(self, seed=None, options=None, fixed=False):
        super().reset(seed=seed)

        if fixed:
            # 고정 시나리오: 방법론 0~3 비교 실험용
            # 동일 초기조건에서 시작해야 전략 차이만 보임
            V_x    = 0.0   # 전진 속도 고정 (수직 하강 시나리오)
            V_z    = 0.0   # 하강 속도 0 (호버링 상태에서 시작)
            alt    = 30.0  # 고도 30m (VRS 진입 평균 시작점, 20~40m 범위 대표값)
            T_norm = 0.5   # 추력 50%
        else:
            # 학습용: 랜덤 초기조건으로 다양한 상황 경험
            V_x    = self.np_random.uniform(0.0, 5.0)
            V_z    = self.np_random.uniform(0.0, 3.0)
            alt    = self.np_random.uniform(self.reset_alt_low, self.reset_alt_high)
            T_norm = 0.5

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
        # 2. 결정론적 규칙 적용 (DANGER, RECOVER) — use_ontology_override=True일 때만
        # zone 분류는 항상 위에서 실행 (로깅/평가용). 개입만 플래그로 제어.
        # ----------------------
        if self.use_ontology_override:
            if zone == "DANGER":
                dT = 1.0
            elif zone == "RECOVER":
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
        # 관측값 NaN/inf 안전 클리핑 (RK45 수치 오류 방어)
        self.state = self._safe_obs(self.state)
        self.step_count += 1

        # ----------------------
        # 4. Sanity check: NaN/inf 및 물리적 이상값 감지
        # ----------------------
        v_i_new, v_h_new = self._compute_vi(V_z_new, T_norm_new)
        state_ok = bool(np.all(np.isfinite(self.state)))
        vi_ok    = np.isfinite(v_i_new) and v_i_new >= 0.0
        vh_ok    = np.isfinite(v_h_new) and v_h_new >= 0.0

        if not (state_ok and vi_ok and vh_ok):
            print(f"[ANOMALY] ep_step={self.step_count} | "
                  f"state={self.state} | v_i={v_i_new:.4f} | v_h={v_h_new:.4f}")
            self.state = np.zeros(4, dtype=np.float32)
            return (self.state, -100.0, True, False,
                    {"anomaly": True, "v_i": v_i_new, "v_h": v_h_new,
                     "vrs_ratio": float('nan'), "zone": "ANOMALY",
                     "step": self.step_count})

        # ----------------------
        # 5. VRS 비율 계산
        # ----------------------
        vrs_ratio = V_z_new / v_h_new if v_h_new > 0.0 else 999.0
        zone_new  = self._ontology(V_z_new, T_norm_new)

        # ----------------------
        # 6. 보상 계산 (alt_prev 전달 → 하강 shaping 보상 계산용)
        # ----------------------
        reward, terminated, termination_reason = self._compute_reward(
            V_z_new, alt_new, T_norm_new, vrs_ratio, zone_new, alt_prev=alt
        )

        # 최대 스텝 초과 시 종료
        truncated = self.step_count >= self.max_steps
        if truncated and termination_reason is None:
            termination_reason = "timeout"

        # 기록 저장
        self.history_alt.append(alt_new)
        self.history_vrs_ratio.append(vrs_ratio)
        self.history_ontology.append(zone_new)

        info = {
            "vrs_ratio"          : vrs_ratio,
            "zone"               : zone_new,
            "step"               : self.step_count,
            "v_i"                : v_i_new,
            "v_h"                : v_h_new,
            "termination_reason" : termination_reason,
            "final_alt"          : float(alt_new),
            "final_vz"           : float(V_z_new),
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
    # _compute_vi(): 유도속도(v_i) 계산
    # x = V_c/v_h (상승 양수, 하강 음수) 구간별로 다른 공식 사용
    # =============================================
    def _compute_vi(self, V_z, T_norm):
        """
        [구간별 공식]
          x > 0      상승  : momentum theory   v_i = (-V_c + sqrt(V_c²+4v_h²)) / 2
          -2 ≤ x ≤ 0 VRS   : Glauert 다항식 근사 (아래)
          x < -2     windmill brake : momentum theory  v_i = (-V_c - sqrt(V_c²-4v_h²)) / 2

        [Glauert 다항식 계수 k0~k3]
          momentum theory 경계조건 (x=0 → y=1, x=-2 → y=1) 을 만족하도록
          구성한 근사 다항식이며, 특정 문헌의 정확한 curve-fit 계수를
          그대로 사용한 것은 아님. (논문 작성 시 출처 오인 방지)
          k0=1.0, k1=-0.5, k2=0.25, k3=0.25 (k4=0)

        반환: (v_i, v_h) — 단위 m/s
        """
        T_nom = float(np.clip(T_norm, 0.0, 1.0)) * self.T_max
        if T_nom <= 0.0:
            return 0.0, 0.0

        v_h = np.sqrt(T_nom / (2.0 * self.rho * self.rotor_A))
        V_c = -V_z          # 상승속도 (V_z 양수=하강이므로 부호 반전)
        x   = V_c / v_h    # 정규화 상승속도

        if x > 0.0:
            # 상승: momentum theory 상승 공식 (항상 실수 해)
            v_i = (-V_c + np.sqrt(V_c**2 + 4.0*v_h**2)) / 2.0

        elif x >= -2.0:
            # VRS 무효구간: Glauert 다항식 (계수 명시)
            # y = k0 + k1*x + k2*x^2 + k3*x^3 + k4*x^4
            # 경계조건: x=0 → y=1 (k0=1), x=-2 → y=1 (Leishman 검증)
            k0, k1, k2, k3, k4 = 1.0, -0.5, 0.25, 0.25, 0.0
            y   = k0 + k1*x + k2*x**2 + k3*x**3 + k4*x**4
            v_i = y * v_h

        else:
            # windmill brake state: momentum theory 역류 공식
            disc = V_c**2 - 4.0*v_h**2
            if disc < 0.0:
                # x가 -2 직하방(수치 오차) → 경계값으로 대체
                v_i = v_h
            else:
                v_i = (-V_c - np.sqrt(disc)) / 2.0

        # v_i는 물리적으로 항상 양수
        v_i = max(float(v_i), 0.0)
        return v_i, float(v_h)

    # =============================================
    # _vrs_ratio(): VRS 위험도 비율 계산
    # V_z / v_h  (구간 경계값 0.5 / 0.8 / 2.0 기준)
    # =============================================
    def _vrs_ratio(self, V_z, T_norm):
        _, v_h = self._compute_vi(V_z, T_norm)

        # 추력이 0이면 v_h=0 → 계산 불가 → RECOVER 처리
        if v_h <= 0.0:
            return 999.0

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
    # _compute_reward(): 보상 계산, 종료 사유를 결정하는 곳 
    # =============================================
    def _compute_reward(self, V_z, alt, T_norm, vrs_ratio, zone, alt_prev=None):
        reward             = 0.0
        terminated         = False
        termination_reason = None

        # ----------------------
        # 1. 하강 진행 보상
        # 생존 보상 제거 — 내려간 거리에 직접 비례해서 보상
        # alt_drop 양수 = 하강, 음수 = 상승 (상승엔 보상 없음)
        # ----------------------
        if alt_prev is not None:
            alt_drop = alt_prev - alt
            reward  += alt_drop * 1.0

        # ----------------------
        # 1-1. 시간 페널티
        # 매 스텝마다 작은 음수 보상 → 빨리 착지할수록 누적 손실 줄어듦
        # 6000스텝 timeout 시 약 -60 / 착지 보너스 +500 대비 충분히 작음
        # 추락(-100)이 timeout(-35~-60)보다 여전히 나쁘므로 추락 유도 위험 없음
        # ----------------------
        if self.use_time_penalty:
            reward -= 0.01

        # ----------------------
        # 2. VRS 패널티 (이차 함수) — use_vrs_penalty=True일 때만
        # 비율이 높아질수록 급격히 강해짐 → 깊은 VRS 진입 강력 억제
        # ratio=0.6: -0.3/스텝, ratio=1.0: -7.5/스텝, ratio=1.5: -30/스텝
        # ----------------------
        if self.use_vrs_penalty and vrs_ratio >= 0.5:
            reward -= (vrs_ratio - 0.5) ** 2 * 30.0

        # ----------------------
        # 3. 종료 조건 (elif로 우선순위 명시: 착지 > 추락 > 고도초과)
        # 현재 파라미터상 세 조건은 수학적으로 겹치지 않으나,
        # 미래 파라미터 변경 시 착지를 최우선으로 판정하기 위해 elif 사용
        # ----------------------
        if alt <= 1.0 and V_z <= 2.0:
            reward            += 500.0
            terminated         = True
            termination_reason = "landed"
            # 착륙 품질 tier 보너스 (부드러울수록 누적 가산)
            if V_z <= 1.5: reward += 100.0   # Tier 1
            if V_z <= 1.0: reward += 100.0   # Tier 2
            if V_z <= 0.5: reward += 100.0   # Tier 3

        elif alt <= 0.0 and V_z > self.V_z_crash:
            reward            -= 100.0
            terminated         = True
            termination_reason = "crash"

        elif alt >= self.alt_max:
            reward            -= 50.0
            terminated         = True
            termination_reason = "altitude_exceeded"

        # ----------------------
        # NaN/inf 안전장치
        # ----------------------
        if not np.isfinite(reward):
            print(f"[REWARD-NaN] step={self.step_count} | "
                  f"V_z={V_z:.3f} alt={alt:.3f} T_norm={T_norm:.3f} "
                  f"vrs_ratio={vrs_ratio} zone={zone} raw_reward={reward}")
            reward = -100.0

        reward = float(np.clip(reward, -200.0, 900.0))

        return reward, terminated, termination_reason

    # =============================================
    # _safe_obs(): 관측값 NaN/inf 안전 클리핑
    # step() 반환 전 observation을 검증할 때 사용
    # =============================================
    def _safe_obs(self, obs):
        """NaN/inf를 안전값(0)으로 대체하고 경고 출력"""
        if not np.all(np.isfinite(obs)):
            print(f"[OBS-NaN] step={self.step_count} | obs_raw={obs}")
            obs = np.where(np.isfinite(obs), obs, 0.0).astype(np.float32)
        # 관측 공간 경계로 클리핑
        low  = self.observation_space.low
        high = self.observation_space.high
        return np.clip(obs, low, high).astype(np.float32)

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
