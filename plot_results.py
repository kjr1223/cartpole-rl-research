"""
plot_results.py
===============
APISAT 논문용 CartPole 실험 결과 시각화
실제 학습 결과(.npy) 파일을 읽어 그래프 생성

[데이터 소스]
  real_cartpole_baseline.npy  : CartPole PPO 기본 (300에피소드 보상)
  real_cartpole_marking.npy   : CartPole PPO 마킹 (300에피소드 보상)
  real_swingup_baseline.npy   : Swing-up 기본 SAC (에피소드별 전환 성공 0/1)
  real_swingup_marking.npy    : Swing-up 마킹 SAC (에피소드별 전환 성공 0/1)
  real_residual_baseline.npy  : Residual RL 기본  (에피소드별 전환 성공 0/1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ══════════════════════════════════════════════════════════════════
# APISAT 논문 스타일 설정
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'lines.linewidth'  : 2.0,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════
def moving_avg(data, window=20):
    """이동평균 (앞쪽은 가용 데이터만 사용)"""
    result = np.zeros(len(data))
    for i in range(len(data)):
        s = max(0, i - window + 1)
        result[i] = np.mean(data[s:i+1])
    return result

def moving_std(data, window=20):
    result = np.zeros(len(data))
    for i in range(len(data)):
        s = max(0, i - window + 1)
        result[i] = np.std(data[s:i+1])
    return result

def shade_band(ax, x, mean, std, color, alpha=0.18):
    ax.fill_between(x, mean - std, mean + std,
                    color=color, alpha=alpha)

# ══════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터 파일 없음: {path}")
    return np.load(path).astype(float)

cp_base  = load('real_cartpole_baseline.npy')   # (300,)  episode reward
cp_mark  = load('real_cartpole_marking.npy')    # (300,)  episode reward
sw_base  = load('real_swingup_baseline.npy')    # (N,)    0/1 per episode
sw_mark  = load('real_swingup_marking.npy')     # (N,)    0/1 per episode
sw_res   = load('real_residual_baseline.npy')   # (N,)    0/1 per episode

N_sw = len(sw_base)   # swing-up 총 에피소드 수

print(f"CartPole 에피소드: {len(cp_base)} (baseline) / {len(cp_mark)} (marking)")
print(f"Swing-up 에피소드: {N_sw}")

# Swing-up 커리큘럼 단계 경계 자동 추정
# (에피소드 수를 4등분 — 실제 단계가 다르면 아래 STAGE_BOUNDS를 직접 수정)
STAGE_BOUNDS = [N_sw // 4, N_sw // 2, N_sw * 3 // 4]
STAGE_LABELS = ['45°→90°', '90°→135°', '135°→180°']

# ══════════════════════════════════════════════════════════════════
# Figure 1: 3-panel 메인 결과
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    'CartPole & Swing-up: Context DB + PINN + SAC Results',
    fontsize=14, fontweight='bold', y=1.02
)

COLOR_BASE  = '#2980b9'   # 파랑 — baseline
COLOR_MARK  = '#e74c3c'   # 빨강 — marking
COLOR_PINN  = '#8e44ad'   # 보라 — PINN / Residual RL

# ── (a) CartPole PPO: 에피소드 보상 ──────────────────────────────
ax = axes[0]
x1 = np.arange(1, len(cp_base) + 1)
W  = 20

base_m = moving_avg(cp_base, W)
mark_m = moving_avg(cp_mark, W)
base_s = moving_std(cp_base, W)
mark_s = moving_std(cp_mark, W)

shade_band(ax, x1, base_m, base_s, COLOR_BASE)
shade_band(ax, x1, mark_m, mark_s, COLOR_MARK)
ax.plot(x1, base_m, color=COLOR_BASE, label='Baseline PPO')
ax.plot(x1, mark_m, color=COLOR_MARK, label='Marking PPO')
ax.axhline(500, color='gray', linestyle='--', alpha=0.5, label='Max (500 steps)')

# 최종 평균값 주석
final_base = np.mean(cp_base[-50:])
final_mark = np.mean(cp_mark[-50:])
ax.annotate(f'{final_base:.1f}',
            xy=(len(cp_base), final_base),
            xytext=(len(cp_base) * 0.78, final_base - 60),
            arrowprops=dict(arrowstyle='->', color=COLOR_BASE, lw=1.2),
            color=COLOR_BASE, fontsize=10)
ax.annotate(f'{final_mark:.1f}',
            xy=(len(cp_mark), final_mark),
            xytext=(len(cp_mark) * 0.78, final_mark + 30),
            arrowprops=dict(arrowstyle='->', color=COLOR_MARK, lw=1.2),
            color=COLOR_MARK, fontsize=10)

ax.set_title('(a) CartPole: PPO Baseline vs Marking')
ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward (steps survived)')
ax.legend(loc='upper left')
ax.set_xlim(0, len(cp_base))
ax.set_ylim(0, 560)

# ── (b) Swing-up 커리큘럼: 전환 성공률 ────────────────────────────
ax = axes[1]
x2 = np.arange(1, N_sw + 1)
W2 = 40

# 성공률 이동평균 (0/1 → %)
sw_base_r = moving_avg(sw_base * 100, W2)
sw_mark_r = moving_avg(sw_mark * 100, W2)
sw_base_s = moving_std(sw_base * 100, W2)
sw_mark_s = moving_std(sw_mark * 100, W2)

shade_band(ax, x2, sw_base_r, sw_base_s * 0.5, COLOR_BASE)
shade_band(ax, x2, sw_mark_r, sw_mark_s * 0.5, COLOR_MARK)
ax.plot(x2, sw_base_r, color=COLOR_BASE, label='Baseline SAC')
ax.plot(x2, sw_mark_r, color=COLOR_MARK, label='Marking SAC')

# 단계 구분선
for xb, lbl in zip(STAGE_BOUNDS, STAGE_LABELS):
    ax.axvline(x=xb, color='gray', linestyle='--', alpha=0.55, linewidth=1.2)
    ax.text(xb + N_sw * 0.01, 93, lbl, fontsize=8, color='gray')

# 최종 성공률 주석
final_base_sw = np.mean(sw_base[-W2:]) * 100
final_mark_sw = np.mean(sw_mark[-W2:]) * 100
ax.annotate(f'{final_base_sw:.1f}%',
            xy=(N_sw, final_base_sw),
            xytext=(N_sw * 0.78, max(final_base_sw + 12, 20)),
            arrowprops=dict(arrowstyle='->', color=COLOR_BASE, lw=1.2),
            color=COLOR_BASE, fontsize=10)
ax.annotate(f'{final_mark_sw:.1f}%',
            xy=(N_sw, final_mark_sw),
            xytext=(N_sw * 0.78, min(final_mark_sw + 12, 88)),
            arrowprops=dict(arrowstyle='->', color=COLOR_MARK, lw=1.2),
            color=COLOR_MARK, fontsize=10)

ax.set_title('(b) Swing-up: Curriculum SAC Baseline vs Marking')
ax.set_xlabel('Episode')
ax.set_ylabel('Switch Success Rate (%)')
ax.legend(loc='upper left')
ax.set_xlim(0, N_sw)
ax.set_ylim(0, 100)

# ── (c) Residual RL vs Marking SAC ────────────────────────────────
ax = axes[2]

sw_res_r = moving_avg(sw_res  * 100, W2)
sw_res_s = moving_std(sw_res  * 100, W2)

shade_band(ax, x2, sw_mark_r, sw_mark_s * 0.5, COLOR_MARK)
shade_band(ax, x2, sw_res_r,  sw_res_s  * 0.5, COLOR_PINN)
ax.plot(x2, sw_mark_r, color=COLOR_MARK, label='Marking SAC (Ours)', linewidth=2.5)
ax.plot(x2, sw_res_r,  color=COLOR_PINN, label='Residual RL (Baseline)',
        linestyle='--', linewidth=2.0)

for xb, lbl in zip(STAGE_BOUNDS, STAGE_LABELS):
    ax.axvline(x=xb, color='gray', linestyle='--', alpha=0.55, linewidth=1.2)
    ax.text(xb + N_sw * 0.01, 93, lbl, fontsize=8, color='gray')

final_res = np.mean(sw_res[-W2:]) * 100
# 135도 이후 Residual RL 실패 표시
fail_x = STAGE_BOUNDS[1] + (N_sw - STAGE_BOUNDS[1]) // 4
fail_y = sw_res_r[fail_x] if fail_x < N_sw else 5
ax.annotate('Physics controller\ninterference',
            xy=(fail_x, fail_y),
            xytext=(fail_x - N_sw * 0.18, fail_y + 20),
            arrowprops=dict(arrowstyle='->', color=COLOR_PINN, lw=1.2),
            color=COLOR_PINN, fontsize=9)

ax.set_title('(c) Marking SAC vs Residual RL')
ax.set_xlabel('Episode')
ax.set_ylabel('Switch Success Rate (%)')
ax.legend(loc='upper left')
ax.set_xlim(0, N_sw)
ax.set_ylim(0, 100)

plt.tight_layout()
out1 = os.path.join(DATA_DIR, 'apisat_results.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
print(f"저장: {out1}")

# ══════════════════════════════════════════════════════════════════
# Figure 2: 단계별 최종 성공률 막대 그래프 (논문 Table 보완용)
# ══════════════════════════════════════════════════════════════════
fig2, ax4 = plt.subplots(figsize=(10, 5))

# 커리큘럼 4단계 경계 (에피소드 인덱스)
stage_edges = [0] + STAGE_BOUNDS + [N_sw]
stage_names = ['Stage1\n(45°)', 'Stage2\n(90°)',
               'Stage3\n(135°)', 'Stage4\n(180°)']

def stage_rate(arr, edges):
    """각 단계 마지막 1/3 구간의 평균 성공률 (수렴 후 값)"""
    rates = []
    for i in range(len(edges) - 1):
        seg = arr[edges[i]:edges[i+1]]
        tail = seg[len(seg)*2//3:]      # 마지막 1/3
        rates.append(float(np.mean(tail) * 100))
    return rates

rates_base = stage_rate(sw_base, stage_edges)
rates_mark = stage_rate(sw_mark, stage_edges)
rates_res  = stage_rate(sw_res,  stage_edges)

x = np.arange(4)
w = 0.25
bars_b = ax4.bar(x - w,  rates_base, w, label='Baseline SAC', color=COLOR_BASE, alpha=0.85)
bars_m = ax4.bar(x,      rates_mark, w, label='Marking SAC',  color=COLOR_MARK, alpha=0.85)
bars_r = ax4.bar(x + w,  rates_res,  w, label='Residual RL',  color=COLOR_PINN, alpha=0.85)

for bars in [bars_b, bars_m, bars_r]:
    for bar in bars:
        h = bar.get_height()
        if h > 1.0:
            ax4.text(bar.get_x() + bar.get_width() / 2., h + 1,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

ax4.set_title('Switch Success Rate by Curriculum Stage',
              fontsize=14, fontweight='bold')
ax4.set_xlabel('Curriculum Stage')
ax4.set_ylabel('Switch Success Rate (%)')
ax4.set_xticks(x)
ax4.set_xticklabels(stage_names)
ax4.legend()
ax4.set_ylim(0, 115)
ax4.grid(True, alpha=0.3, axis='y')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
out2 = os.path.join(DATA_DIR, 'apisat_bar.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
print(f"저장: {out2}")

# ══════════════════════════════════════════════════════════════════
# 수치 요약 출력 (논문 Table 작성용)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 50)
print("수치 요약 (논문 Table용)")
print("=" * 50)
print(f"CartPole 최종 보상 (마지막 50 ep 평균)")
print(f"  Baseline PPO : {np.mean(cp_base[-50:]):.1f}")
print(f"  Marking  PPO : {np.mean(cp_mark[-50:]):.1f}")
print()
print(f"Swing-up 최종 전환 성공률 (마지막 {W2} ep 평균)")
print(f"  Baseline SAC : {np.mean(sw_base[-W2:]) * 100:.1f}%")
print(f"  Marking  SAC : {np.mean(sw_mark[-W2:]) * 100:.1f}%")
print(f"  Residual RL  : {np.mean(sw_res[-W2:])  * 100:.1f}%")
print()
print("단계별 성공률:")
for i, name in enumerate(stage_names):
    print(f"  {name.replace(chr(10), ' ')}: "
          f"Base={rates_base[i]:.1f}%  Mark={rates_mark[i]:.1f}%  Res={rates_res[i]:.1f}%")
