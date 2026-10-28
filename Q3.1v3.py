
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
# Step 3: Implement the FSM from Q2.1
#------------------------------------

# 1. Define discrete states
states = [
    "Printing",
    "Saving location",
    "Moving to wipe tower",
    "Wiping",
    "Cooling nozzle",
    "Loading filament",
    "Heating nozzle",
    "Moving to saved location"
]

# 2. Define transitions and corresponding events
transitions = {
    "Printing": {
        "C": "Saving location",
        "F": "Saving location"
    },
    "Saving location": {
        "location_saved": "Moving to wipe tower"
    },
    "Moving to wipe tower": {
        "wipe_tower_reached": "Wiping"
    },
    "Wiping": {
        "wiping_done": "Cooling nozzle"
    },
    "Cooling nozzle": {
        "nozzle_cold": "Loading filament"
    },
    "Loading filament": {
        "filament_loaded": "Heating nozzle"
    },
    "Heating nozzle": {
        "nozzle_hot": "Moving to saved location"
    },
    "Moving to saved location": {
        "saved_position_reached": "Printing"
    }
}


# 3. Define the transition function
def update_fsm(mode, event, continuous_state, saved_position):

    # Check whether the event is valid in the current state
    if event not in transitions[mode]:
        raise ValueError(
            f"Invalid event '{event}' in state '{mode}'"
        )

    next_mode = transitions[mode][event]

    # Save the actual tool-head position on replacement request
    if mode == "Printing":
        saved_position = (
            float(continuous_state[0]),
            float(continuous_state[3])
        )

    return next_mode, saved_position


#------------------------------------
# Step 4: Verify the FSM transitions
#------------------------------------

# 1. Initial motor state
motor_state = np.array([
    0.08, 0.0, 0.0,
    0.08, 0.0, 0.0
])

# 2. Events for one filament replacement
replacement_events = [
    "location_saved",
    "wipe_tower_reached",
    "wiping_done",
    "nozzle_cold",
    "filament_loaded",
    "nozzle_hot",
    "saved_position_reached"
]


# 3. Test both independent replacement triggers
for trigger in ["C", "F"]:

    mode = "Printing"
    saved_position = None
    history = [mode]

    print(f"\nFilament replacement triggered by {trigger}")
    print(f"Initial state: {mode}")

    for event in [trigger] + replacement_events:

        mode, saved_position = update_fsm(
            mode, event, motor_state, saved_position
        )

        history.append(mode)
        print(f"{event} -> {mode}")

    # Verify final state and saved location
    assert mode == "Printing"
    assert np.allclose(saved_position, [0.08, 0.08])

    print(f"Saved position: {saved_position}")
    print("FSM sequence verified.")


# 4. Visualize the FSM sequence
mode_numbers = [states.index(mode) for mode in history]

plt.figure(figsize=(10, 5))

plt.step(
    range(len(history)),
    mode_numbers,
    where="post",
    marker="o"
)

plt.yticks(range(len(states)), states)
plt.xlabel("Transition number")
plt.ylabel("Discrete state")
plt.title("Finite State Machine - Filament Replacement")
plt.grid(True)
plt.tight_layout()
plt.show()