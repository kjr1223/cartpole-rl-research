"""
UAM VRS 결과 시각화
1. VRS 포락선 다이어그램 (Vz/Vh vs Vx/Vh)
2. Stage 1 착지율 추이
3. 보상 함수 비교 (구 vs 신)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
})

# ────────────────────────────────────────────
# Fig A: VRS 포락선 다이어그램
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("UAM VRS Simulation Environment", fontsize=13, fontweight='bold')

ax = axes[0]
ax.set_title("VRS Risk Envelope (Momentum Theory)", fontweight='bold')

# 배경 구역 색칠
vz_range = np.linspace(0, 3.0, 300)
vx_range = np.linspace(0, 1.5, 300)
VZ, VX = np.meshgrid(vz_range, vx_range)

# VRS 위험 구역: 0.5 ≤ Vz/Vh ≤ 2.0 AND Vx/Vh ≤ 0.5
danger_mask  = (VZ >= 0.5) & (VZ <= 2.0) & (VX <= 0.5)
caution_mask = (VZ >= 0.3) & (VZ <= 2.2) & (VX <= 0.7) & ~danger_mask

ax.contourf(VZ, VX, danger_mask.astype(float),
            levels=[0.5, 1.5], colors=['#FC8181'], alpha=0.5)
ax.contourf(VZ, VX, caution_mask.astype(float),
            levels=[0.5, 1.5], colors=['#F6AD55'], alpha=0.4)

# 경계선
ax.axvline(x=0.5, color='#E53E3E', linewidth=2, linestyle='--', label='VRS onset (Vz/Vh=0.5)')
ax.axvline(x=2.0, color='#C53030', linewidth=2, linestyle='-', label='VRS exit (Vz/Vh=2.0)')
ax.axhline(y=0.5, color='#E53E3E', linewidth=2, linestyle=':', label='Lateral limit (Vx/Vh=0.5)')

# Glauert 보정 곡선 (VRS 내부 추력 저하)
vz_vrs = np.linspace(0.5, 2.0, 100)
# Glauert polynomial: y = 1 - 0.5x + 0.25x² + 0.25x³  (정규화된 추력 비율)
thrust_ratio = 1 - 0.5*vz_vrs + 0.25*vz_vrs**2 + 0.25*vz_vrs**3
thrust_ratio = thrust_ratio / thrust_ratio.max() * 0.45  # Vx 축에 스케일링해서 표시
ax.plot(vz_vrs, thrust_ratio, 'b--', linewidth=1.5, alpha=0.7, label='Glauert correction')

# 안전 하강 경로 예시
vz_safe = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.3])
vx_safe = np.array([0.8, 0.75, 0.70, 0.65, 0.60, 0.55])
ax.plot(vz_safe, vx_safe, 'g-o', linewidth=2.5, markersize=5,
        label='Safe descent path', zorder=5)

# 위험 하강 경로 예시
vz_danger = np.array([0.0, 0.3, 0.6, 1.0, 1.4])
vx_danger = np.array([0.8, 0.6, 0.4, 0.3, 0.2])
ax.plot(vz_danger, vx_danger, 'r--x', linewidth=2, markersize=7,
        label='VRS entry path', zorder=5, color='#C53030')

# 구역 레이블
ax.text(1.25, 0.25, 'DANGER\n(VRS)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#C53030',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(0.15, 0.25, 'SAFE', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#276749')
ax.text(2.5, 0.8, 'SAFE\n(High speed)', ha='center', va='center',
        fontsize=9, color='#276749')

ax.set_xlabel('Normalized Descent Velocity  Vz / Vh')
ax.set_ylabel('Normalized Forward Velocity  Vx / Vh')
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.5)
ax.legend(loc='upper right', fontsize=8.5)

# ────────────────────────────────────────────
# Fig B: 착지율 추이 (Stage 1, 에피소드별)
# ────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("Stage 1 Landing Success Rate (20–40 m)", fontweight='bold')

# uam_fresh_training.log에서 직접 읽은 데이터
episodes = [10,20,30,40,50,60,70,80,90,100,
            110,120,130,140,150,160,170,180,190,200,
            210,220,230,240,250,260,270,280,290,300,
            310,320,330,340,350,360,370,380,390,400,
            410,420,430,440,450,460,470,480,490,500,
            510,520,530,540,550,560,570,580,590,600,
            610,620,630,640,650,660,670,680,690,700,
            710,720,730,740,750,760,770,780,790,800,
            810,820,830,840,850,860,870,880,890,900,
            910,920,930,940,950,960,970,980,990,1000]

landing_rates = [0.0,0.0,0.0,0.0,2.0,3.3,2.9,2.5,3.3,3.0,
                 4.5,5.0,9.2,8.6,8.0,7.5,7.1,6.7,6.3,6.0,
                 5.7,5.5,5.2,5.0,4.8,4.6,4.4,4.3,4.5,4.3,
                 4.2,4.1,3.9,4.4,4.3,5.3,5.9,5.8,5.6,6.0,
                 7.1,7.4,7.7,8.0,8.4,8.5,8.3,8.5,8.4,8.4,
                 9.2,9.2,9.2,9.3,9.3,9.1,8.9,9.1,9.3,9.7,
                 9.5,9.5,9.5,9.8,9.7,9.7,10.3,10.3,10.1,10.1,
                 10.0,10.3,10.1,10.0,9.9,9.7,9.9,9.7,9.6,9.6,
                 9.9,10.1,10.0,9.9,9.8,9.7,9.5,9.5,9.6,9.6,
                 9.5,9.6,9.7,9.6,9.8,9.7,9.7,9.7,9.6,9.5]

ax2.plot(episodes, landing_rates, color='#3182CE', linewidth=1.5,
         alpha=0.4, label='Landing rate')

# 이동평균
window = 10
ma = np.convolve(landing_rates, np.ones(window)/window, mode='valid')
ep_ma = episodes[window-1:]
ax2.plot(ep_ma, ma, color='#3182CE', linewidth=2.5, label=f'MA({window})')

# 주요 이벤트 표시
ax2.axvline(x=50,  color='orange', linewidth=1.5, linestyle='--', alpha=0.8)
ax2.text(55, 1.5, 'First\nlanding\n(ep50)', fontsize=8, color='darkorange')

ax2.axvline(x=670, color='green', linewidth=1.5, linestyle='--', alpha=0.8)
ax2.text(680, 7, '10%\npeak\n(ep670)', fontsize=8, color='darkgreen')

ax2.axhline(y=9.5, color='#E53E3E', linewidth=1.5, linestyle=':',
            label='Final: 9.5%')
ax2.text(10, 10.0, 'Final: 9.5% (95/1000)', fontsize=9,
         color='#E53E3E', fontweight='bold')

ax2.fill_between(episodes, landing_rates, alpha=0.1, color='#3182CE')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Cumulative Landing Rate (%)')
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 13)
ax2.legend(loc='lower right', fontsize=9)

# 이전 방법 비교 주석
ax2.annotate('All previous methods\nstayed at 0%\n(survival bonus problem)',
             xy=(200, 0.3), xytext=(300, 4),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=8.5, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', alpha=0.9))

plt.tight_layout()
plt.savefig('/home/jrkim/cartpole_project/uam_vrs_results.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("저장: uam_vrs_results.png")


# ────────────────────────────────────────────
# Fig C: 커리큘럼 단계별 착지율 비교 (Bar chart)
# ────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(8, 5))
ax3.set_title("UAM Curriculum: Landing Rate by Stage", fontweight='bold')

stages    = ['Stage 1\n(20–40 m)', 'Stage 2\n(60–90 m)', 'Stage 3\n(100–150 m)']
rates     = [9.5, 2.3, 0.0]
colors    = ['#3182CE', '#E67E22', '#BDC3C7']
statuses  = ['Completed', 'Completed', 'Planned']

bars = ax3.bar(stages, rates, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, rate, status in zip(bars, rates, statuses):
    ypos = bar.get_height() + 0.15
    ax3.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{rate}%\n({status})',
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             color='#2D3748')

ax3.set_ylabel('Landing Success Rate (%)')
ax3.set_ylim(0, 14)
ax3.axhline(y=0, color='black', linewidth=0.8)

# Stage 3 미정 표시
ax3.text(2, 1.5, 'Under\nplanning', ha='center', fontsize=9, color='#718096')

# 설명 박스
props = dict(boxstyle='round', facecolor='#EBF8FF', alpha=0.8)
ax3.text(0.02, 0.97,
         'Reward redesign (no survival bonus)\nenabled first-ever landing at Stage 1',
         transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=props, color='#2B6CB0')

plt.tight_layout()
plt.savefig('/home/jrkim/cartpole_project/uam_stage_comparison.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("저장: uam_stage_comparison.png")
