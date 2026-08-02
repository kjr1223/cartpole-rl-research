"""
uam_comparison_figure.py
========================
베이스라인 vs 제안 방법 비교 그래프 생성
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 데이터 ──────────────────────────────────────────────
labels  = ['Landing\nSuccess Rate', 'VRS Entry Rate\n(per step)']
baseline = [1.0,  15.3]
proposed = [59.0,  0.9]

x     = np.arange(len(labels))
width = 0.32

# ── 색상 ────────────────────────────────────────────────
C_BASE = '#E74C3C'   # 빨강 — 베이스라인
C_PROP = '#2980B9'   # 파랑 — 제안 방법

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars_b = ax.bar(x - width/2, baseline, width, color=C_BASE, alpha=0.88,
                label='Baseline (Pure SAC, No VRS Awareness)', zorder=3)
bars_p = ax.bar(x + width/2, proposed, width, color=C_PROP, alpha=0.88,
                label='Proposed (SAC + VRS-aware Reward + Ontology)', zorder=3)

# 수치 레이블
for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.0,
            f'{h:.1f}%', ha='center', va='bottom', fontsize=10,
            color=C_BASE, fontweight='bold')

for bar in bars_p:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.0,
            f'{h:.1f}%', ha='center', va='bottom', fontsize=10,
            color=C_PROP, fontweight='bold')

# 개선율 화살표 + 텍스트 (첫번째, 두번째 바만)
improvements = [
    (0, baseline[0], proposed[0], '+58pp'),
    (1, baseline[1], proposed[1], '−14.4pp'),
]
x = np.arange(len(labels))
for idx, b_val, p_val, txt in improvements:
    x_pos = x[idx] + width/2 + 0.22
    y_lo  = min(b_val, p_val) + 1.5
    y_hi  = max(b_val, p_val) - 1.0
    ax.annotate('', xy=(x_pos, y_hi), xytext=(x_pos, y_lo),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.4))
    ax.text(x_pos + 0.06, (y_lo + y_hi)/2, txt,
            va='center', ha='left', fontsize=9, color='#333', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Rate (%)', fontsize=11)
ax.set_ylim(0, 75)
ax.set_title('Baseline vs. Proposed Method\n(Deterministic Evaluation, 200 Episodes)',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out = '/home/jrkim/cartpole_project/uam_comparison_figure.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[저장] {out}")
