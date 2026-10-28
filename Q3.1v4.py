
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
# Step 3: Hybrid FSM and physical guards
#------------------------------------

# 1. Operating thresholds
T_low = 60.0
T_high = 200.0
T_start = 200.0

epsilon_p = 1e-4   # Position tolerance [m]
epsilon_v = 1e-3   # Velocity tolerance [m/s]

# Equilibrium heating voltage at T_start
u_hold = np.sqrt(
    R_n * (T_start - T_e) / R_tn
)

u_heat = 24.0

# 2. FSM states
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


# 3. Position and velocity guard
def position_guard(state, target):

    position = state[[0, 3]]
    velocity = state[[2, 5]]

    return (
        np.linalg.norm(position - target) <= epsilon_p
        and np.linalg.norm(velocity) <= epsilon_v
    )


# 4. FSM transition with physical guard checking
def update_fsm(mode, event, state, saved_position):

    if event not in transitions[mode]:
        raise ValueError(
            f"Invalid event '{event}' in mode '{mode}'"
        )

    if event == "wipe_tower_reached":
        if not position_guard(state, np.array([0.0, 0.0])):
            raise RuntimeError("Wipe tower guard not satisfied.")

    elif event == "saved_position_reached":
        if not position_guard(
            state, np.asarray(saved_position)
        ):
            raise RuntimeError("Return position guard not satisfied.")

    elif event == "nozzle_cold":
        if state[6] > T_low + 1e-6:
            raise RuntimeError("Cooling guard not satisfied.")

    elif event == "nozzle_hot":
        if state[6] < T_high - 1e-6:
            raise RuntimeError("Heating guard not satisfied.")

    # Reset map: save actual tool-head position
    if mode == "Printing" and event in ("C", "F"):
        saved_position = state[[0, 3]].copy()

    next_mode = transitions[mode][event]

    return next_mode, saved_position


#------------------------------------
# Step 4: Motor trajectories from Q1.4
#------------------------------------

# 1. Numerical motor coefficients
J_num = float(J_sys.subs(params))
cx_num = float(c_x.subs(params))
cy_num = float(c_y.subs(params))

rm_num = float(params[r_m])
Km_num = float(params[K_m])
Lm_num = float(params[L_m])
Rm_num = float(params[R_m])


# 2. Quintic motion profile
def motion_profile(t, duration):

    tau = np.clip(t / duration, 0.0, 1.0)

    velocity = (
        30*tau**2 - 60*tau**3 + 30*tau**4
    ) / duration

    acceleration = (
        60*tau - 180*tau**2 + 120*tau**3
    ) / duration**2

    jerk = (
        60 - 360*tau + 360*tau**2
    ) / duration**3

    return velocity, acceleration, jerk


# 3. Inverse dynamics from Q1.4
def required_voltage(displacement, damping, t, duration):

    velocity, acceleration, jerk = motion_profile(t, duration)

    current = 2 / (rm_num * Km_num) * (
        J_num * displacement * acceleration
        + damping * displacement * velocity
    )

    current_dot = 2 / (rm_num * Km_num) * (
        J_num * displacement * jerk
        + damping * displacement * acceleration
    )

    return (
        Lm_num * current_dot
        + Rm_num * current
        + (2 * Km_num / rm_num) * displacement * velocity
    )


#------------------------------------
# Step 5: Hybrid simulation
#------------------------------------

# 1. Initial state: preheated printer, motor at rest
state = np.array([
    0.08, 0.0, 0.0,
    0.08, 0.0, 0.0,
    T_start, u_hold, u_hold / R_n
])

t_now = 0.0
mode = "Printing"
saved_position = None

# Store continuous and discrete simulation histories
history = []
mode_log = [(t_now, mode)]


# 2. Integrate continuous dynamics in the active mode
def integrate(state0, t0, t1, mode, input_func, event=None):

    sol = solve_ivp(
        lambda t, z: sys(z, input_func(t)),
        (t0, t1),
        state0,
        method="BDF",
        events=event,
        max_step=0.5,
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    if event is not None and len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"Physical event not detected in mode '{mode}'."
        )

    history.append((mode, sol.t.copy(), sol.y.copy()))

    return sol.y[:, -1].copy(), float(sol.t[-1]), sol


# 3. Apply a discrete transition
def switch(mode, event, state, saved_position, t_now):

    next_mode, saved_position = update_fsm(
        mode, event, state, saved_position
    )

    mode_log.append((t_now, next_mode))

    print(f"t = {t_now:.3f} s: {mode} -> {next_mode}")

    return next_mode, saved_position


# 4. Simulate a motion using the physical motor model
def move(state0, t0, target, mode, duration=2.0):

    target = np.asarray(target, dtype=float)
    start = state0[[0, 3]].copy()
    displacement = target - start

    def motor_inputs(t):

        local_t = t - t0

        ux = required_voltage(
            displacement[0], cx_num, local_t, duration
        )

        uy = required_voltage(
            displacement[1], cy_num, local_t, duration
        )

        return [
            (ux + uy) / 2,
            (ux - uy) / 2,
            u_hold
        ]

    state1, t1, sol = integrate(
        state0, t0, t0 + duration,
        mode, motor_inputs
    )

    # Check physical arrival at the end of the motion
    if not position_guard(state1, target):
        raise RuntimeError(
            f"Motion target not reached: {target}"
        )

    return state1, t1, sol


# 5. Automatic temperature events
def temperature_event(threshold, direction):

    def event(t, state):
        return state[6] - threshold

    event.terminal = True
    event.direction = direction

    return event


#------------------------------------
# Step 6: Execute one filament replacement
#------------------------------------

# 1. New colour requested
mode, saved_position = switch(
    mode, "C", state, saved_position, t_now
)

mode, saved_position = switch(
    mode, "location_saved", state, saved_position, t_now
)

# 2. Park: vertical down, then horizontal left
state, t_now, sol_vertical = move(
    state, t_now, (state[0], 0.0), mode
)

state, t_now, sol_horizontal = move(
    state, t_now, (0.0, 0.0), mode
)

mode, saved_position = switch(
    mode, "wipe_tower_reached",
    state, saved_position, t_now
)

# 3. Wiping: illustrative discrete completion event
state, t_now, _ = integrate(
    state, t_now, t_now + 2.0, mode,
    lambda t: [0.0, 0.0, u_hold]
)

mode, saved_position = switch(
    mode, "wiping_done", state, saved_position, t_now
)

# 4. Heater OFF: cool until the physical threshold
state, t_now, _ = integrate(
    state, t_now, t_now + 300.0, mode,
    lambda t: [0.0, 0.0, 0.0],
    event=temperature_event(T_low, -1)
)

t_cold = t_now

mode, saved_position = switch(
    mode, "nozzle_cold", state, saved_position, t_now
)

# 5. Load new filament: illustrative discrete event
state, t_now, _ = integrate(
    state, t_now, t_now + 2.0, mode,
    lambda t: [0.0, 0.0, 0.0]
)

mode, saved_position = switch(
    mode, "filament_loaded", state, saved_position, t_now
)

# 6. Heater ON: heat until the physical threshold
state, t_now, _ = integrate(
    state, t_now, t_now + 300.0, mode,
    lambda t: [0.0, 0.0, u_heat],
    event=temperature_event(T_high, 1)
)

t_hot = t_now

mode, saved_position = switch(
    mode, "nozzle_hot", state, saved_position, t_now
)

# 7. Return to saved position
state, t_now, sol_return = move(
    state, t_now, saved_position, mode
)

mode, saved_position = switch(
    mode, "saved_position_reached",
    state, saved_position, t_now
)

# 8. Final verification
assert mode == "Printing"
assert position_guard(state, saved_position)

print("\nHybrid simulation verified.")
print(f"Saved position: {saved_position}")
print(f"Final position: {state[[0, 3]]}")
print(f"Final nozzle temperature: {state[6]:.2f} deg C")
print(f"Cold threshold reached at: {t_cold:.2f} s")
print(f"Hot threshold reached at: {t_hot:.2f} s")
print(f"Replacement completed at: {t_now:.2f} s")


#------------------------------------
# Step 7: Visualization
#------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Tool-head motion
axes[0].plot(
    sol_vertical.y[0], sol_vertical.y[3],
    label="Vertical parking"
)
axes[0].plot(
    sol_horizontal.y[0], sol_horizontal.y[3],
    label="Horizontal parking"
)
axes[0].plot(
    sol_return.y[0], sol_return.y[3],
    "--", label="Return"
)

axes[0].scatter(
    [0, saved_position[0]],
    [0, saved_position[1]]
)
axes[0].set(
    title="Tool-head movement",
    xlabel="x [m]",
    ylabel="y [m]"
)
axes[0].set_aspect("equal")
axes[0].grid(True)
axes[0].legend()


# 2. Nozzle temperature throughout the simulation
times = np.concatenate([item[1] for item in history])
temperatures = np.concatenate([
    item[2][6] for item in history
])

axes[1].plot(times, temperatures, label="Nozzle temperature")
axes[1].axhline(
    T_low, linestyle="--", label="Low threshold"
)
axes[1].axhline(
    T_high, linestyle=":", label="High threshold"
)
axes[1].set(
    title="Nozzle temperature",
    xlabel="Time [s]",
    ylabel="Temperature [deg C]"
)
axes[1].grid(True)
axes[1].legend()


# 3. Discrete FSM mode versus simulation time
mode_times = [item[0] for item in mode_log]
mode_indices = [states.index(item[1]) for item in mode_log]

mode_times.append(t_now)
mode_indices.append(states.index(mode))

axes[2].step(
    mode_times, mode_indices,
    where="post"
)
axes[2].set_yticks(range(len(states)), states)
axes[2].set(
    title="Hybrid FSM transitions",
    xlabel="Time [s]",
    ylabel="Discrete mode"
)
axes[2].grid(True)

plt.tight_layout()
plt.show()
