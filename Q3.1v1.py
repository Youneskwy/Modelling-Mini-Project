
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
# Step 2: Construct a combined dynamics function
#------------------------------------

# Motor states: [x, i_x, x_dot, y, i_y, y_dot]
# Motor inputs: [u_m,l, u_m,r]

def sys(state, u):

    u_l, u_r = u

    # Transform physical motor voltages
    u_x = u_l + u_r
    u_y = u_l - u_r

    # Decoupled dynamics from Q1.3
    dx = Ax_num @ state[:3] + Bx_num * u_x
    dy = Ay_num @ state[3:] + By_num * u_y

    return np.concatenate((dx, dy))


#------------------------------------
# Step 3: Verify the combined motor model
#------------------------------------

# 1. Initial conditions
state0 = np.array([
    0.08, 0.0, 0.0,
    0.08, 0.0, 0.0
])

time = np.linspace(0, 0.5, 400)


# 2. Simulate constant motor voltages
def simulate(u_l, u_r):

    sol = solve_ivp(
        lambda t, state: sys(state, [u_l, u_r]),
        (time[0], time[-1]),
        state0,
        t_eval=time,
        method="BDF",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol


# 3. Verification experiments
tests = [
    ("Horizontal motion", 1.0, 1.0),
    ("Vertical motion", 1.0, -1.0)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, u_l, u_r) in zip(axes, tests):

    sol = simulate(u_l, u_r)

    x = sol.y[0]
    y = sol.y[3]

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

    # Verify decoupling
    if name == "Horizontal motion":
        assert np.max(np.abs(y - y[0])) < 1e-10
    else:
        assert np.max(np.abs(x - x[0])) < 1e-10

print("\nCombined motor model verified.")

plt.tight_layout()
plt.show()
