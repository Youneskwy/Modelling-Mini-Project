
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#------------------------------------
# Step 1: Reuse the motor model from Q1.4v5
#------------------------------------

# 1. Physical parameters
m, r_m, J_m = sp.symbols("m r_m J_m", positive=True)
K_m, L_m, R_m = sp.symbols("K_m L_m R_m", positive=True)

mu_lx, mu_ly, mu_r = sp.symbols(
    "mu_lx mu_ly mu_r", nonnegative=True
)

# 2. Equivalent inertia and damping from Q1.2
J_sys = J_m + m * r_m**2 / 2

d = r_m**2 / 4 * sp.Matrix([
    [mu_lx + mu_ly, mu_lx - mu_ly],
    [mu_lx - mu_ly, mu_lx + mu_ly]
])

r = mu_r * r_m**2 * sp.Matrix([
    [3, 1],
    [1, 3]
])

D = d + r

c_x = sp.simplify(D[0, 0] + D[0, 1])
c_y = sp.simplify(D[0, 0] - D[0, 1])


# 3. Decoupled state-space matrices from Q1.3
def state_matrices(c):

    A = sp.Matrix([
        [0, 0, 1],
        [0, -R_m/L_m, -2*K_m/(r_m*L_m)],
        [0, r_m*K_m/(2*J_sys), -c/J_sys]
    ])

    B = sp.Matrix([
        [0],
        [1/L_m],
        [0]
    ])

    return A, B


A_x, B_x = state_matrices(c_x)
A_y, B_y = state_matrices(c_y)


# 4. Numerical parameters from Q1.4v5
params = {
    m: 0.25,
    r_m: 0.0063662,
    J_m: 3.8e-7,
    K_m: 0.032,
    L_m: 0.000710,
    R_m: 16.4,
    mu_lx: 0.20,
    mu_ly: 0.25,
    mu_r: 0.04
}

Ax_num = np.array(A_x.subs(params), dtype=float)
Ay_num = np.array(A_y.subs(params), dtype=float)

Bx_num = np.array(B_x.subs(params), dtype=float).ravel()
By_num = np.array(B_y.subs(params), dtype=float).ravel()


#------------------------------------
# Step 2: Construct a combined dynamics function/Add the nozzle dynamics from Q1.5
#------------------------------------

# Motor states: [x, i_x, x_dot, y, i_y, y_dot]
# Motor inputs: [u_m,l, u_m,r]

# 1. nozzle parameters
L_n = 0.005      # Inductance [H]
C_n = 50e-6      # Capacitance [F]
R_n = 12.0       # Heater resistance [Ohm]

C_tn = 12.0      # Nozzle thermal capacity [J/K]
R_tn = 5.0       # Thermal resistance [K/W]
T_e = 22.0       # Ambient temperature [deg C]

# 2. Combined motor and nozzle dynamics

# State:
# [x, i_x, x_dot, y, i_y, y_dot, T_n, v_Cn, i_Ln]
#
# Input:
# [u_m,l, u_m,r, u_n]

def sys(state, u):

    u_l, u_r, u_n = u

    # Motor dynamics from Q1.3
    u_x = u_l + u_r
    u_y = u_l - u_r

    dx = Ax_num @ state[:3] + Bx_num * u_x
    dy = Ay_num @ state[3:6] + By_num * u_y

    # Nozzle dynamics from Q1.5
    T_n, v_Cn, i_Ln = state[6:9]

    dT_n = (
        v_Cn**2 / R_n
        - (T_n - T_e) / R_tn
    ) / C_tn

    dv_Cn = (
        i_Ln - v_Cn / R_n
    ) / C_n

    di_Ln = (
        u_n - v_Cn
    ) / L_n

    return np.concatenate((
        dx,
        dy,
        [dT_n, dv_Cn, di_Ln]
    ))



#------------------------------------
# Step 3: Verify the combined physical model
#------------------------------------

# 1. Initial state
state0 = np.array([
    0.08, 0.0, 0.0,   # Horizontal states
    0.08, 0.0, 0.0,   # Vertical states
    T_e, 0.0, 0.0    # Nozzle states
])


# 2. Verify motor dynamics
tests = [
    ("Horizontal motion", [1.0, 1.0, 0.0]),
    ("Vertical motion", [1.0, -1.0, 0.0])
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, u) in zip(axes, tests):

    time_motor = np.linspace(0, 0.5, 400)

    sol = solve_ivp(
        lambda t, state: sys(state, u),
        (0, 0.5),
        state0,
        t_eval=time_motor,
        method="BDF",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    x = sol.y[0]
    y = sol.y[3]

    ax.plot(time_motor, x, label="x position")
    ax.plot(time_motor, y, label="y position")
    ax.set_title(name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()

    print(f"\n{name}")
    print(f"Delta x = {x[-1] - x[0]:.6f} m")
    print(f"Delta y = {y[-1] - y[0]:.6f} m")

    if name == "Horizontal motion":
        assert np.max(np.abs(y - y[0])) < 1e-10
    else:
        assert np.max(np.abs(x - x[0])) < 1e-10

    # The nozzle must remain at ambient temperature
    assert np.max(np.abs(sol.y[6] - T_e)) < 1e-8

plt.tight_layout()
plt.show()


# 3. Verify nozzle heating
time_heat = np.linspace(0, 120, 500)

sol_heat = solve_ivp(
    lambda t, state: sys(state, [0, 0, 24]),
    (0, 120),
    state0,
    t_eval=time_heat,
    method="BDF",
    rtol=1e-8,
    atol=1e-10
)

if not sol_heat.success:
    raise RuntimeError(sol_heat.message)


# 4. Verify nozzle cooling
time_cool = np.linspace(120, 240, 500)

sol_cool = solve_ivp(
    lambda t, state: sys(state, [0, 0, 0]),
    (120, 240),
    sol_heat.y[:, -1],
    t_eval=time_cool,
    method="BDF",
    rtol=1e-8,
    atol=1e-10
)

if not sol_cool.success:
    raise RuntimeError(sol_cool.message)

assert sol_heat.y[6, -1] > T_e
assert sol_cool.y[6, -1] < sol_heat.y[6, -1]

# Motor positions should remain constant
assert np.max(np.abs(sol_heat.y[0] - state0[0])) < 1e-10
assert np.max(np.abs(sol_heat.y[3] - state0[3])) < 1e-10

print("\nNozzle temperature verification")
print(f"Initial temperature: {T_e:.2f} deg C")
print(f"After heating: {sol_heat.y[6, -1]:.2f} deg C")
print(f"After cooling: {sol_cool.y[6, -1]:.2f} deg C")


# 5. Plot nozzle temperature
plt.figure(figsize=(8, 4))

plt.plot(
    sol_heat.t, sol_heat.y[6],
    label="Heater ON (24 V)"
)

plt.plot(
    sol_cool.t, sol_cool.y[6],
    label="Heater OFF (0 V)"
)

plt.axhline(
    T_e, linestyle="--",
    label="Ambient temperature"
)

plt.title("Nozzle thermal response")
plt.xlabel("Time [s]")
plt.ylabel("Nozzle temperature [deg C]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
