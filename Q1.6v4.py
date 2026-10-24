
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp


#------------------------------------
# Step 1: Elliptical geometry and filament deposition
#------------------------------------

# 1. Elliptical trajectory from Q1.4
T_print = 20.0
omega = 2 * np.pi / T_print

x_c, y_c = 0.11, 0.11
amp_x, amp_y = 0.04, 0.025

time = np.linspace(0, T_print, 2001)

x = x_c + amp_x * np.cos(omega * time)
y = y_c + amp_y * np.sin(omega * time)

# 2. Tool-head velocity along the ellipse
x_dot = -amp_x * omega * np.sin(omega * time)
y_dot = amp_y * omega * np.cos(omega * time)

speed = np.hypot(x_dot, y_dot)

# 3. Deposited filament length
length = cumulative_trapezoid(speed, time, initial=0)
L = length[-1]

# 4. Spatial discretization
N = 200
ds = L / N

# Segment centers (used for temperature plotting)
s = (np.arange(N) + 0.5) * ds

# Segment boundaries (used for deposition timing)
s_edges = np.linspace(0, L, N + 1)

# 5. Deposition times at segment boundaries
t_edges = np.interp(s_edges, length, time)

# 6. Verification
assert np.all(np.diff(length) > 0)
assert np.all(np.diff(t_edges) > 0)
assert np.allclose([x[0], y[0]], [x[-1], y[-1]])
assert np.isclose(length[0], 0)
assert np.isclose(t_edges[0], 0)
assert np.isclose(t_edges[-1], T_print)

print(f"Total filament length: {L:.5f} m")
print(f"Number of segments: {N}")
print(f"Segment length: {ds * 1000:.3f} mm")


#------------------------------------
# Step 2: Thermal PDE simulation
#------------------------------------

# 1. Thermal parameters

# PLA material properties
rho = 1240.0       # Density [kg/m^3]
c_p = 1800.0       # Specific heat capacity [J/(kg K)]
k = 0.13           # Thermal conductivity [W/(m K)]

# Representative square filament cross-section
w = 0.0005         # Filament width [m]

# Heat transfer coefficients
h_air = 30.0       # Air heat transfer [W/(m^2 K)]
h_bed = 600.0      # Bed contact conductance [W/(m^2 K)]

# Operating temperatures
T_air = 22.0       # Ambient temperature [deg C]
T_bed = 60.0       # Heated bed temperature [deg C]
T_dep = 200.0      # Deposition temperature [deg C]

# PDE coefficients
alpha = k / (rho * c_p)
beta_air = 3 * h_air / (rho * c_p * w)
beta_bed = h_bed / (rho * c_p * w)

#------------------------------------
# Step 2: Thermal PDE simulation
#------------------------------------

# 1. Thermal parameters

# PLA material properties
rho = 1240.0       # Density [kg/m^3]
c_p = 1800.0       # Specific heat capacity [J/(kg K)]
k = 0.13           # Thermal conductivity [W/(m K)]

# Representative square filament cross-section
w = 0.0005         # Filament width [m]

# Heat transfer coefficients
h_air = 30.0       # Air heat transfer [W/(m^2 K)]
h_bed = 600.0      # Bed contact conductance [W/(m^2 K)]

# Operating temperatures
T_air = 22.0       # Ambient temperature [deg C]
T_bed = 60.0       # Heated bed temperature [deg C]
T_dep = 200.0      # Deposition temperature [deg C]

# PDE coefficients
alpha = k / (rho * c_p)
beta_air = 3 * h_air / (rho * c_p * w)
beta_bed = h_bed / (rho * c_p * w)


# 2. Spatially discretized heat equation
def thermal_dynamics(t, temperature, periodic=False):

    laplacian = np.zeros_like(temperature)

    if periodic:
        # Closed loop: periodic boundary conditions
        laplacian = (
            np.roll(temperature, 1)
            - 2 * temperature
            + np.roll(temperature, -1)
        ) / ds**2

    elif len(temperature) > 1:
        # Open filament: zero axial heat flux at both ends
        laplacian[1:-1] = (
            temperature[:-2]
            - 2 * temperature[1:-1]
            + temperature[2:]
        ) / ds**2

        laplacian[0] = (
            temperature[1] - temperature[0]
        ) / ds**2

        laplacian[-1] = (
            temperature[-2] - temperature[-1]
        ) / ds**2

    # Conduction and heat exchange
    return (
        alpha * laplacian
        - beta_air * (temperature - T_air)
        - beta_bed * (temperature - T_bed)
    )


# 3. Growing computational domain

# Discrete approximation:
# Segment i becomes active at its leading-edge time.
# The first segment is initialized at t = 0.

temperature = np.array([T_dep])

# Store temperature distribution exactly halfway
snapshot_time = T_print / 2
T_half = None

for i in range(N):

    t_start = t_edges[i]
    t_end = t_edges[i + 1]

    # Cool currently deposited filament
    sol = solve_ivp(
        lambda t, T: thermal_dynamics(t, T),
        (t_start, t_end),
        temperature,
        method="BDF",
        dense_output=True,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Save temperature exactly at t = 10 s
    if (
        T_half is None
        and t_start <= snapshot_time <= t_end + 1e-12
    ):
        T_half = sol.sol(snapshot_time).copy()

    temperature = sol.y[:, -1]

    # Activate next segment at deposition temperature
    if i < N - 1:
        temperature = np.append(temperature, T_dep)

# Printing completed at t = T_print
T_finished = temperature.copy()

assert T_half is not None
assert len(T_finished) == N

print("\nPrinting completed.")
print(f"Minimum temperature: {T_finished.min():.2f} deg C")
print(f"Maximum temperature: {T_finished.max():.2f} deg C")


# 4. Cooling after printing

# The printed ellipse is now a closed filament loop.
# Stop when the maximum temperature difference reaches 1 deg C.

tolerance = 1.0
max_cooling_time = 160.0


def uniformity_event(t, temperature):
    return np.ptp(temperature) - tolerance


uniformity_event.terminal = True
uniformity_event.direction = -1

initial_spread = np.ptp(T_finished)

if initial_spread <= tolerance:
    # The filament is already nearly uniform
    t_uniform = T_print
    T_after = T_finished.copy()

    cooling_plot_time = np.array([T_print])
    cooling_spread = np.array([initial_spread])

else:
    sol_cooling = solve_ivp(
        lambda t, T: thermal_dynamics(
            t, T, periodic=True
        ),
        (T_print, T_print + max_cooling_time),
        T_finished,
        method="BDF",
        events=uniformity_event,
        dense_output=True,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol_cooling.success:
        raise RuntimeError(sol_cooling.message)

    # Check whether the event was reached
    if len(sol_cooling.t_events[0]) > 0:
        t_uniform = float(sol_cooling.t_events[0][0])
        t_end = t_uniform
    else:
        t_uniform = None
        t_end = float(sol_cooling.t[-1])

    # Temperature at stopping time
    T_after = sol_cooling.sol(t_end)

    # Sample solution for visualization
    cooling_plot_time = np.linspace(
        T_print, t_end, 300
    )

    cooling_plot_states = sol_cooling.sol(
        cooling_plot_time
    )

    cooling_spread = np.ptp(
        cooling_plot_states, axis=0
    )


# 5. Verify cooling results

if t_uniform is not None:
    print(f"\nNearly uniform at t = {t_uniform:.3f} s")
    print(
        "Cooling after printing = "
        f"{t_uniform - T_print:.3f} s"
    )
else:
    print("\nUniformity criterion not reached.")
    print(
        f"Final temperature spread: "
        f"{np.ptp(T_after):.3f} deg C"
    )

# Analytical equilibrium temperature
T_eq = (
    3 * h_air * T_air + h_bed * T_bed
) / (3 * h_air + h_bed)

print(f"Expected equilibrium temperature: {T_eq:.2f} deg C")


# 6. Visualization

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Temperature distribution along filament
axes[0].plot(
    s[:len(T_half)],
    T_half,
    label="During printing (10 s)"
)

axes[0].plot(
    s,
    T_finished,
    label="Printing completed (20 s)"
)

if t_uniform is not None:
    after_label = "Nearly uniform"
else:
    after_label = "End of cooling simulation"

axes[0].plot(
    s,
    T_after,
    label=after_label
)

axes[0].set(
    title="Filament temperature distribution",
    xlabel="Position along filament [m]",
    ylabel="Temperature [deg C]"
)

axes[0].grid(True)
axes[0].legend()


# Temperature uniformity after printing
axes[1].plot(
    cooling_plot_time,
    cooling_spread,
    label="Temperature difference"
)

axes[1].axhline(
    tolerance,
    linestyle="--",
    label="1 deg C threshold"
)

axes[1].set(
    title="Temperature uniformity",
    xlabel="Time [s]",
    ylabel="Max(T) - Min(T) [deg C]"
)

axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()