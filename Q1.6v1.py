import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

#------------------------------------
# Step 1: Elliptical geometry and filament deposition
#------------------------------------

# 1. Elliptical trajectory from Q1.4
T_print = 20.0
omega = 2*np.pi/T_print

x_c, y_c = 0.11, 0.11
amp_x, amp_y = 0.04, 0.025

time = np.linspace(0, T_print, 2001)

x = x_c + amp_x * np.cos(omega * time)
y = y_c + amp_y * np.sin(omega * time)

# 2. Tool-head velocity along the ellipse
x_dot = -amp_x * omega * np.sin(omega * time)
y_dot = amp_y * omega * np.cos(omega * time)

speed = np.sqrt(x_dot**2 + y_dot**2)

# 3. Deposited filament length
# Numerically integrate speed over time
length = cumulative_trapezoid(speed, time, initial=0)

L = length[-1]  # Total ellipse circumference [m]

# 4. Spatial discretization of the filament
N = 200
ds = L / N

# Position of each segment along the filament
s = (np.arange(N) + 0.5) * ds

# 5. Determine deposition time of each segment
# Invert the length(time) relationship numerically
t_dep = np.interp(s, length, time)

# 6. Verification
assert np.all(np.diff(length) > 0)
assert np.all(np.diff(t_dep) > 0)
assert np.allclose([x[0], y[0]], [x[-1], y[-1]])
assert np.isclose(length[0], 0)

print(f"Total filament length: {L:.5f} m")
print(f"Number of segments: {N}")
print(f"Segment length: {ds * 1000:.3f} mm")
print(f"First deposition time: {t_dep[0]:.3f} s")
print(f"Last deposition time: {t_dep[-1]:.3f} s")

#------------------------------------
# Step 2: Visualization
#------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Show deposition progress halfway through printing
half_idx = np.searchsorted(time, T_print / 2)

axes[0].plot(x, y, "--", label="Full ellipse")
axes[0].plot(
    x[:half_idx + 1],
    y[:half_idx + 1],
    linewidth=3,
    label="Deposited at t = 10 s"
)
axes[0].set(
    title="Filament deposition",
    xlabel="x [m]",
    ylabel="y [m]"
)
axes[0].set_aspect("equal")
axes[0].legend()
axes[0].grid(True)

# Growing computational domain
axes[1].plot(time, length)
axes[1].set(
    title="Growing filament length",
    xlabel="Time [s]",
    ylabel="Deposited length [m]"
)
axes[1].grid(True)

# Deposition time of every spatial segment
axes[2].plot(s, t_dep)
axes[2].set(
    title="Segment deposition times",
    xlabel="Position along filament [m]",
    ylabel="Deposition time [s]"
)
axes[2].grid(True)

plt.tight_layout()
plt.show()
