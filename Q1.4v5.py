
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

#------------------------------------
# Step 3: Define the desired trajectories
#------------------------------------

# 1. Time and trajectory parameters
t= sp.symbols("t", real=True)

T = 20.0              # Duration of the trajectory [s]
omega = 2 * np.pi / T # Fundamental angular frequency [rad/s]

# Assumed workspace dimensions for the tool-head
d_x = 0.22    # Workspace width [m]
d_y = 0.22    # Workspace height [m]

# Center of both trajectories
x_c = d_x / 2
y_c = d_y / 2

# 2. Eliptical trajectory
amp_x = 0.04               # Horizontal semi-axis [m]
amp_y = 0.025               # Vertical semi-axis [m]

x_ellipse = x_c + amp_x * sp.cos(omega * t)
y_ellipse = y_c + amp_y * sp.sin(omega * t)

# 3. Lissajous trajectory
A_liss = 0.04             # Amplitude [m]

freq_x = 8
freq_y = 5
Delta = 7 * sp.pi / 8     # Phase shift [rad]

x_lissajous = x_c + A_liss * sp.sin(freq_x * omega * t)
y_lissajous = y_c + A_liss * sp.sin(freq_y * omega * t + Delta)

# 4 Convert symbolic trajectories to numerical functions
time = np.linspace(0, T, 1000)

def evaluate_trajectory(x_expr, y_expr):

    x_func = sp.lambdify(t, x_expr, "numpy")
    y_func = sp.lambdify(t, y_expr, "numpy")

    x_values = np.asarray(x_func(time), dtype=float)
    y_values = np.asarray(y_func(time), dtype=float)

    return x_values, y_values

trajectories = [
    ("Elliptical", x_ellipse, y_ellipse),
    ("Lissajous", x_lissajous, y_lissajous)
]

#------------------------------------
# Step 4: Calculate the required motor input voltages for the desired trajectories
#------------------------------------

# 1. Inverse dynamics from Q1.3
def inverse_dynamics(position, damping):

    velocity = sp.diff(position, t)
    acceleration = sp.diff(position, t, 2)

    # Required transformed motor current
    current = (2 / (r_m * K_m)) * (
        J_sys * acceleration + damping * velocity
    )

    # Required transformed voltage
    voltage = (
        L_m * sp.diff(current, t)
        + R_m * current
        + (2 * K_m / r_m) * velocity
    )

    return current, voltage

# 2. Calculate and visualize the motor voltages
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, x_expr, y_expr) in zip(axes, trajectories):

    # Calculate transformed currents and voltages
    i_x, u_x = inverse_dynamics(x_expr, c_x)
    i_y, u_y = inverse_dynamics(y_expr, c_y)

    # Transform voltages back to physical motors
    u_l = (u_x + u_y) / 2
    u_r = (u_x - u_y) / 2

    # Substitute numerical parameters
    u_l_func = sp.lambdify(t, u_l.subs(params), "numpy")
    u_r_func = sp.lambdify(t, u_r.subs(params), "numpy")

    u_l_values = u_l_func(time)
    u_r_values = u_r_func(time)

    # Plot motor voltages
    ax.plot(time, u_l_values, label="Left motor")
    ax.plot(time, u_r_values, label="Right motor")

    ax.set_title(name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Motor voltage [V]")
    ax.grid(True)
    ax.legend()

    print(f"\n{name}")
    print(f"Maximum |u_l| = {np.max(np.abs(u_l_values)):.4f} V")
    print(f"Maximum |u_r| = {np.max(np.abs(u_r_values)):.4f} V")

plt.tight_layout()
plt.show()

#------------------------------------
# Step 5: Numerical simulation and trajectory tracking
#------------------------------------

# 1. Convert symbolic expressions into numerical functions
def numerical(expr):
    return sp.lambdify(t, expr.subs(params), "numpy")


# 2. Simulate a desired trajectory
def simulate_trajectory(name, x_expr, y_expr):

    # Desired position and velocity
    x_func = numerical(x_expr)
    y_func = numerical(y_expr)

    vx_func = numerical(sp.diff(x_expr, t))
    vy_func = numerical(sp.diff(y_expr, t))

    # Required currents and voltages from inverse dynamics
    ix, ux = inverse_dynamics(x_expr, c_x)
    iy, uy = inverse_dynamics(y_expr, c_y)

    ix_func = numerical(ix)
    iy_func = numerical(iy)

    # Physical motor voltages
    ul_func = numerical((ux + uy) / 2)
    ur_func = numerical((ux - uy) / 2)

    # Combined state:
    # [x, i_x, x_dot, y, i_y, y_dot]

    def dynamics(t_now, state):

        # Evaluate actual motor voltages
        ul = float(ul_func(t_now))
        ur = float(ur_func(t_now))

        # Transform to Cartesian voltage inputs
        ux_now = ul + ur
        uy_now = ul - ur

        # State-space equations from Q1.3
        dx = Ax_num @ state[:3] + Bx_num * ux_now
        dy = Ay_num @ state[3:] + By_num * uy_now

        return np.concatenate((dx, dy))

    # 3. Initial conditions: ideal trajectory tracking
    state0_ideal = [
        float(x_func(0)),
        float(ix_func(0)),
        float(vx_func(0)),
        float(y_func(0)),
        float(iy_func(0)),
        float(vy_func(0))
    ]

    # 4. Initial conditions: tool head initially at rest
    state0_rest = [
        float(x_func(0)),
        0.0,
        0.0,
        float(y_func(0)),
        0.0,
        0.0
    ]

    # 5. Numerical integration
    def solve(initial_state):
        sol = solve_ivp(
            dynamics,
            (0, T),
            initial_state,
            t_eval=time,
            method="BDF",
            rtol=1e-8,
            atol=1e-10
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol

    sol_ideal = solve(state0_ideal)
    sol_rest = solve(state0_rest)

    # 6. Desired trajectory
    xd = np.asarray(x_func(time), dtype=float)
    yd = np.asarray(y_func(time), dtype=float)

    # 7. Calculate position tracking errors
    error_ideal = np.hypot(
        sol_ideal.y[0] - xd,
        sol_ideal.y[3] - yd
    )

    error_rest = np.hypot(
        sol_rest.y[0] - xd,
        sol_rest.y[3] - yd
    )

    print(f"\n{name}")
    print(f"Maximum ideal tracking error: "
          f"{np.max(error_ideal):.3e} m")
    print(f"Maximum from-rest tracking error: "
          f"{np.max(error_rest):.3e} m")

    # 8. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # XY trajectories
    axes[0].plot(
        sol_ideal.y[0], sol_ideal.y[3],
        label="Ideal initial state"
    )

    axes[0].plot(
        sol_rest.y[0], sol_rest.y[3],
        label="Initially at rest"
    )

    axes[0].plot(
        xd, yd, "k--",
        label="Desired trajectory"
    )

    axes[0].set_title("Tool-head trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)
    axes[0].legend()

    # Tracking error over time
    axes[1].plot(time, error_ideal, label="Ideal initial state")
    axes[1].plot(time, error_rest, label="Initially at rest")

    axes[1].set_title("Position tracking error")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Error [m]")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


# 9. Run both trajectories
for name, x_expr, y_expr in trajectories:
    simulate_trajectory(name, x_expr, y_expr)
