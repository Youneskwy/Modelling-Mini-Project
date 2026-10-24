
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#------------------------------------
# Step 1: Implement the model derived in Q1.3
#------------------------------------

# Physical parameters
m, r_m, J_m = sp.symbols("m r_m J_m", positive=True)
K_m, L_m, R_m = sp.symbols("K_m L_m R_m", positive=True)

mu_lx, mu_ly, mu_r = sp.symbols(
    "mu_lx mu_ly mu_r", nonnegative=True
)

# Equivalent inertia from Q1.2
J_sys = J_m + m * r_m**2 / 2

# Linear bearing friction from Q1.2
d = r_m**2 / 4 * sp.Matrix([
    [mu_lx + mu_ly, mu_lx - mu_ly],
    [mu_lx - mu_ly, mu_lx + mu_ly]
])

# Rolling bearing friction from Q1.2
r = mu_r * r_m**2 * sp.Matrix([
    [3, 1],
    [1, 3]
])

# Total damping
D = d + r

a = D[0, 0]
b = D[0, 1]

# Decoupled damping coefficients from Q1.3
c_x = sp.simplify(a + b)
c_y = sp.simplify(a - b)


# Construct the decoupled state-space matrices
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



#------------------------------------
# Step 2: Simulate the system response to a step input
#------------------------------------

# 1. Numerical Parameters (SI units)

# Motor reference:
# FAULHABER 2232A024SR - 24 V DC motor
#
# Tool-head mass:
# Estimated compact extruder assembly
#
# Motor pulley:
# GT2 belt, 20 teeth, 2 mm pitch
#
# Friction:
# Assumed equivalent viscous damping coefficients

params = {
    m: 0.25,         # Tool-head mass [kg]
    r_m: 0.0063662,  # Motor pulley radius [m]
    J_m: 3.8e-7,     # Motor inertia [kg*m^2]
    K_m: 0.032,      # Motor constant [Nm/A]
    L_m: 0.000710,   # Motor inductance [H]
    R_m: 16.4,       # Motor resistance [Ohm]
    mu_lx: 0.20,     # Linear damping x [Ns/m]
    mu_ly: 0.25,     # Linear damping y [Ns/m]
    mu_r: 0.04       # Rolling damping [Ns/m]
}

# 2. Substitute the parameters into the state-space matrices
Ax_num = np.array(A_x.subs(params), dtype=float)
Ay_num = np.array(A_y.subs(params), dtype=float)

Bx_num = np.array(B_x.subs(params), dtype=float).ravel()
By_num = np.array(B_y.subs(params), dtype=float).ravel()

# 3. Simulate constant motor voltages
def simulate(u_l, u_r):

    # Transform motor voltages using Q1.3
    u_x = u_l + u_r
    u_y = u_l - u_r

    time = np.linspace(0, 0.5, 400)

    # Initial states: [position, current, velocity]
    # Tool head initially at rest
    x0 = [0.08, 0.0, 0.0]
    y0 = [0.08, 0.0, 0.0]

    def dynamics_x(t, state):
        return Ax_num @ state + Bx_num * u_x

    def dynamics_y(t, state):
        return Ay_num @ state + By_num * u_y

    sol_x = solve_ivp(
        dynamics_x, (0, 0.5), x0,
        t_eval=time, method="BDF"
    )

    sol_y = solve_ivp(
        dynamics_y, (0, 0.5), y0,
        t_eval=time, method="BDF"
    )

    if not (sol_x.success and sol_y.success):
        raise RuntimeError("Numerical integration failed.")

    return time, sol_x.y[0], sol_y.y[0]

#4. Run two demo experiments
tests = [
    ("Horizontal motion", 1.0, 1.0),
    ("Vertical motion", 1.0, -1.0)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, u_l, u_r) in zip(axes, tests):

    time, x, y = simulate(u_l, u_r)

    ax.plot(time, x, label="x position")
    ax.plot(time, y, label="y position")

    ax.set_title(name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()

    print(f"\n{name}")
    print(f"Delta x = {x[-1] - x[0]:.6f} m")
    print(f"Delta y = {y[-1] - y[0]:.6f} m")


plt.tight_layout()
plt.show()