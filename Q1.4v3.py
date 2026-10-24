
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
A_x = 0.04                # Horizontal semi-axis [m]
A_y = 0.025               # Vertical semi-axis [m]

x_ellipse = x_c + A_x * sp.cos(omega * t)
y_ellipse = y_c + A_y * sp.sin(omega * t)

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

# 5. Plot and verify the desired shapes

trajectories = [
    ("Ellipse", x_ellipse, y_ellipse),
    ("Lissajous (8:5)", x_lissajous, y_lissajous)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, x_expr, y_expr) in zip(axes, trajectories):

    xd, yd = evaluate_trajectory(x_expr, y_expr)

    # Verify that the trajectory closes
    assert np.allclose(
        [xd[0], yd[0]],
        [xd[-1], yd[-1]],
        atol=1e-10
    )

    # Verify that the trajectory remains inside the workspace
    assert np.all((xd >= 0) & (xd <= d_x))
    assert np.all((yd >= 0) & (yd <= d_y))

    ax.plot(xd, yd, label="Desired trajectory")
    ax.scatter(xd[0], yd[0], label="Starting point")

    ax.set_title(name)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, d_x)
    ax.set_ylim(0, d_y)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    print(f"{name}: closed trajectory within workspace.")

plt.tight_layout()
plt.show()
