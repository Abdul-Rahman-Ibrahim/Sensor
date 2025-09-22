# # Re-run after reset: equilibrium curve s vs v

# import numpy as np
# import matplotlib.pyplot as plt

# # Equilibrium condition: v = s^3 - s
# s_vals = np.linspace(-3, 3, 400)
# v_eq = s_vals**3 - s_vals

# plt.figure(figsize=(7, 5))
# plt.plot(v_eq, s_vals, label=r'Equilibria: $v = s^3 - s$')
# plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
# plt.xlabel("Voltage v")
# plt.ylabel("State s (equilibrium)")
# plt.title("Equilibrium States: s vs v (ds/dt = 0)")
# plt.grid(True)
# plt.legend()
# plt.show()

# Plot s vs v at equilibrium (ds/dt = 0) with stability indicated.
# Stability from linearization: ds/dt = (v - s^3 + s)/tau
# f'(s*) = (1/tau) * (1 - 3 s*^2).  Stable if f'(s*) < 0  <=> |s*| > 1/sqrt(3).

import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(-3, 3, 1200)
v = s**3 - s

s_thresh = 1/np.sqrt(3)

# Masks for branches
stable_mask_left  = s <= -s_thresh
unstable_mask_mid = (s > -s_thresh) & (s < s_thresh)
stable_mask_right = s >= s_thresh

plt.figure(figsize=(7,5))
# Plot stable branches (solid)
plt.plot(v[stable_mask_left],  s[stable_mask_left],  label="Stable branch")
plt.plot(v[stable_mask_right], s[stable_mask_right], label="_nolegend_")  # same style

# Plot unstable branch (dashed)
plt.plot(v[unstable_mask_mid], s[unstable_mask_mid], linestyle="--", label="Unstable branch")

# Mark fold (turning) points at s=±1/√3
v_fold = s_thresh**3 - s_thresh
plt.scatter([v_fold, -v_fold], [s_thresh, -s_thresh], marker="o", label=r"Fold points $|s|=1/\sqrt{3}$")

plt.axhline(0, linestyle="--", linewidth=0.8)
plt.axvline(0, linestyle="--", linewidth=0.8)
plt.xlabel("Voltage v")
plt.ylabel("State s (equilibrium)")
plt.title("Equilibrium s vs v with stability (ds/dt = 0)")
plt.grid(True)
plt.legend()
plt.show()
