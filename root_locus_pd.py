import control as ct
import matplotlib.pyplot as plt

# 플랜트 G(s) = 1 / (s^2 + 2s + 5)
G = ct.tf([1], [1, 2, 5])

# PD 제어기 C(s) = s + 3 (영점 z = -3 추가)
C_pd = ct.tf([1, 3], [1])

plt.figure(figsize=(6, 4))
ct.root_locus(G * C_pd, grid=True)
plt.title('Root Locus with PD Zero (Chapter 6)')
plt.xlabel('Real Axis (Real-part)')
plt.ylabel('Imaginary Axis (Imag-part)')
plt.show()
