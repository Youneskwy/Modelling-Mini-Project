
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
# Step 2 New: Mesh refinement and convergence analysis
#------------------------------------

# 1. Run the thermal simulation for a given mesh size
def simulate_mesh(N):

    ds = L / N
    s = (np.arange(N) + 0.5) * ds

    # Deposition times at segment boundaries
    s_edges = np.linspace(0, L, N + 1)
    t_edges = np.interp(s_edges, length, time)

    # Discretized heat equation
    def thermal_dynamics(t, temperature, periodic=False):

        laplacian = np.zeros_like(temperature)

        if periodic:
            # Closed filament: periodic boundary conditions
            laplacian = (
                np.roll(temperature, 1)
                - 2 * temperature
                + np.roll(temperature, -1)
            ) / ds**2

        elif len(temperature) > 1:
            # Open filament: zero axial heat flux
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

        return (
            alpha * laplacian
            - beta_air * (temperature - T_air)
            - beta_bed * (temperature - T_bed)
        )

    # 2. Growing domain during printing
    temperature = np.array([T_dep])

    for i in range(N):

        sol = solve_ivp(
            lambda t, T: thermal_dynamics(t, T),
            (t_edges[i], t_edges[i + 1]),
            temperature,
            method="BDF",
            rtol=1e-7,
            atol=1e-9
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        temperature = sol.y[:, -1]

        if i < N - 1:
            temperature = np.append(temperature, T_dep)

    # Temperature distribution at t = 20 s
    T_finished = temperature.copy()

    # 3. Cooling of the closed filament
    tolerance = 1.0

    def uniformity_event(t, T):
        return np.ptp(T) - tolerance

    uniformity_event.terminal = True
    uniformity_event.direction = -1

    if np.ptp(T_finished) <= tolerance:
        return 0.0, s, T_finished

    sol_cooling = solve_ivp(
        lambda t, T: thermal_dynamics(
            t, T, periodic=True
        ),
        (T_print, T_print + 160),
        T_finished,
        method="BDF",
        events=uniformity_event,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol_cooling.success:
        raise RuntimeError(sol_cooling.message)

    if len(sol_cooling.t_events[0]) == 0:
        raise RuntimeError(
            f"Uniformity criterion not reached for N={N}"
        )

    t_uniform = sol_cooling.t_events[0][0]
    cooling_duration = t_uniform - T_print

    return cooling_duration, s, T_finished


# 4. Mesh refinement study

mesh_sizes = [100, 200, 400]
results = []

for N in mesh_sizes:

    t_cool, s, T_finished = simulate_mesh(N)

    results.append({
        "N": N,
        "ds": L / N,
        "t_cool": t_cool,
        "s": s,
        "temperature": T_finished
    })


# 5. Compare cooling times

print("\nMesh refinement results")
print("-" * 51)
print(f"{'N':>5} {'ds [mm]':>12} {'t_cool [s]':>13} {'Delta [s]':>13}")
print("-" * 51)

for i, result in enumerate(results):

    if i == 0:
        difference = "-"
    else:
        delta = abs(
            result["t_cool"] - results[i-1]["t_cool"]
        )
        difference = f"{delta:.5f}"

    print(
        f"{result['N']:5d} "
        f"{result['ds']*1000:12.4f} "
        f"{result['t_cool']:13.5f} "
        f"{difference:>13}"
    )


# 6. Visualization

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cooling time versus mesh size
axes[0].plot(
    [r["N"] for r in results],
    [r["t_cool"] for r in results],
    "o-"
)

axes[0].set(
    title="Cooling time convergence",
    xlabel="Number of segments N",
    ylabel="Cooling time [s]"
)
axes[0].grid(True)

# Temperature distributions at printing completion
for result in results:

    axes[1].plot(
        result["s"],
        result["temperature"],
        label=f"N = {result['N']}"
    )

axes[1].set(
    title="Temperature at t = 20 s",
    xlabel="Position along filament [m]",
    ylabel="Temperature [deg C]"
)

axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
