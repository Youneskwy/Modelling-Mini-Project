
import sympy as sp

# Q1.4 - Step 1: Implement the model derived in Q1.3

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


# Verify the damping coefficients against Q1.3
expected_cx = r_m**2 * mu_lx / 2 + 4*mu_r*r_m**2
expected_cy = r_m**2 * mu_ly / 2 + 2*mu_r*r_m**2

assert sp.simplify(c_x - expected_cx) == 0
assert sp.simplify(c_y - expected_cy) == 0

# Verify matrix dimensions
assert A_x.shape == A_y.shape == (3, 3)
assert B_x.shape == B_y.shape == (3, 1)

print("c_x =", c_x)
print("c_y =", c_y)
print("Symbolic model verified.")
