import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid

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
# Step 2: Combined motor and nozzle dynamics from Q1.5
#------------------------------------
# 1. Nozzle parameters
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

# 3. Hybrid invariants and discrete operation durations
workspace_x = 0.22
workspace_y = 0.22
wiping_duration = 2.0
loading_duration = 2.0
parked_modes = {
    "Wiping",
    "Cooling nozzle",
    "Loading filament",
    "Heating nozzle"
}

def check_invariant(mode, state):
    x, y = state[0], state[3]
    T_n = state[6]
    # The tool head must remain inside the workspace
    if not (
        -epsilon_p <= x <= workspace_x + epsilon_p
        and -epsilon_p <= y <= workspace_y + epsilon_p
    ):
        raise RuntimeError("Workspace invariant violated.")
    # The tool head must remain parked during replacement
    if mode in parked_modes:
        if not position_guard(state, np.array([0.0, 0.0])):
            raise RuntimeError(
                f"Parked-position invariant violated in {mode}."
            )
    # Temperature invariants
    if mode == "Cooling nozzle" and T_n < T_low - 1e-4:
        raise RuntimeError("Cooling invariant violated.")
    if mode == "Loading filament" and T_n > T_low + 1e-4:
        raise RuntimeError("Loading temperature invariant violated.")
    if mode == "Heating nozzle" and T_n > T_high + 1e-4:
        raise RuntimeError("Heating invariant violated.")

# 4. Position and velocity guard

def position_guard(state, target):
    position = state[[0, 3]]
    velocity = state[[2, 5]]
    return (
        np.linalg.norm(position - target) <= epsilon_p
        and np.linalg.norm(velocity) <= epsilon_v
    )

# 5. FSM transition function with guards and reset maps

def update_fsm(mode, event, state, saved_position, t_now):
    check_invariant(mode, state)
    if event not in transitions[mode]:
        raise ValueError(
            f"Invalid event '{event}' in mode '{mode}'"
        )
    # Physical position guards
    if event == "wipe_tower_reached":
        if not position_guard(state, np.array([0.0, 0.0])):
            raise RuntimeError("Wipe tower guard not satisfied.")
    elif event == "saved_position_reached":
        if saved_position is None or not position_guard(
            state, np.asarray(saved_position)
        ):
            raise RuntimeError("Return position guard not satisfied.")
    # Physical temperature guards
    elif event == "nozzle_cold":
        if state[6] > T_low + 1e-6:
            raise RuntimeError("Cooling guard not satisfied.")
    elif event == "nozzle_hot":
        if state[6] < T_high - 1e-6:
            raise RuntimeError("Heating guard not satisfied.")
    # Time-based discrete events
    elif event == "wiping_done":
        elapsed = t_now - mode_entry_time["Wiping"]
        if elapsed < wiping_duration - 1e-9:
            raise RuntimeError("Wiping not completed.")
    elif event == "filament_loaded":
        elapsed = t_now - mode_entry_time["Loading filament"]
        if elapsed < loading_duration - 1e-9:
            raise RuntimeError("Filament loading not completed.")
    elif event == "location_saved":
        if saved_position is None:
            raise RuntimeError("Print position not saved.")
    # Reset map: store the print location when C or F occurs
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
# Step 4b: Reuse Q1.4 inverse dynamics for printing and stopping
#------------------------------------
T_print = 20.0
T_stop = 2.0
omega = 2 * np.pi / T_print
x_c, y_c = 0.11, 0.11
amp_x, amp_y = 0.04, 0.025


def ellipse_derivatives(t):
    phase = omega * t
    return (
        np.array([x_c + amp_x * np.cos(phase),
                  y_c + amp_y * np.sin(phase)]),
        np.array([-amp_x * omega * np.sin(phase),
                   amp_y * omega * np.cos(phase)]),
        np.array([-amp_x * omega**2 * np.cos(phase),
                  -amp_y * omega**2 * np.sin(phase)]),
        np.array([amp_x * omega**3 * np.sin(phase),
                  -amp_y * omega**3 * np.cos(phase)])
    )


def axis_inverse_current(velocity, acceleration, damping):
    return 2 / (rm_num * Km_num) * (
        J_num * acceleration + damping * velocity
    )


def axis_inverse_voltage(velocity, acceleration, jerk, damping):
    current = axis_inverse_current(velocity, acceleration, damping)
    current_dot = 2 / (rm_num * Km_num) * (
        J_num * jerk + damping * acceleration
    )
    return (Lm_num * current_dot + Rm_num * current
            + 2 * Km_num / rm_num * velocity)


def printing_inputs(t):
    _, velocity, acceleration, jerk = ellipse_derivatives(t)
    ux = axis_inverse_voltage(velocity[0], acceleration[0], jerk[0], cx_num)
    uy = axis_inverse_voltage(velocity[1], acceleration[1], jerk[1], cy_num)
    return [(ux + uy) / 2, (ux - uy) / 2, u_hold]


def braking_coefficients(position, velocity, acceleration, duration):
    # Quintic interpolation with continuous initial position, speed and current.
    # Final speed and acceleration are zero; no physical state is reset.
    target = position + 0.5 * duration * velocity
    c0 = position
    c1 = velocity * duration
    c2 = 0.5 * acceleration * duration**2
    c345 = np.linalg.solve(
        np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]),
        np.array([target - c0 - c1 - c2,
                  -c1 - 2*c2,
                  -2*c2])
    )
    return np.r_[c0, c1, c2, c345]


def polynomial_derivatives(coefficients, elapsed, duration):
    tau = np.clip(elapsed / duration, 0.0, 1.0)
    v = np.polynomial.polynomial.polyval(
        tau, np.polynomial.polynomial.polyder(coefficients)
    ) / duration
    a = np.polynomial.polynomial.polyval(
        tau, np.polynomial.polynomial.polyder(coefficients, 2)
    ) / duration**2
    j = np.polynomial.polynomial.polyval(
        tau, np.polynomial.polynomial.polyder(coefficients, 3)
    ) / duration**3
    return v, a, j

#------------------------------------
# Step 5: Hybrid simulation
#------------------------------------
# 1. Ideal Q1.4 initial state: the print head is already moving
p0, v0, a0, _ = ellipse_derivatives(0.0)
state = np.array([
    p0[0], axis_inverse_current(v0[0], a0[0], cx_num), v0[0],
    p0[1], axis_inverse_current(v0[1], a0[1], cy_num), v0[1],
    T_start, u_hold, u_hold / R_n
])
t_now = 0.0
mode = "Printing"
saved_position = None
mode_entry_time = {"Printing": t_now}

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
        dense_output=True,
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
    for sample in sol.y.T:
        check_invariant(mode, sample)
    history.append((mode, sol.t.copy(), sol.y.copy()))
    return sol.y[:, -1].copy(), float(sol.t[-1]), sol

# 3. Apply a discrete transition and reset the mode clock

def switch(mode, event, state, saved_position, t_now):
    next_mode, saved_position = update_fsm(
        mode, event, state, saved_position, t_now
    )
    # Check whether the destination mode is admissible
    check_invariant(next_mode, state)
    # Reset the discrete mode clock
    mode_entry_time[next_mode] = t_now
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
# 0. Print the full ellipse with the Q1.4 motor model
state, t_now, sol_print = integrate(
    state, t_now, T_print, mode, printing_inputs
)
p_end, v_end, _, _ = ellipse_derivatives(T_print)
assert np.linalg.norm(state[[0, 3]] - p_end) < 1e-5
assert np.linalg.norm(state[[2, 5]] - v_end) < 1e-5
print(f"\nEllipse printed at t = {t_now:.3f} s")
print(f"Printing endpoint: {state[[0, 3]]}")
print(f"Printing endpoint speed: {np.linalg.norm(state[[2, 5]])*1000:.3f} mm/s")

# 1. New colour requested
mode, saved_position = switch(
    mode, "C", state, saved_position, t_now
)
# Brake without resetting motor positions, speeds or currents.
# The saved position remains the actual printing endpoint.
start_p = state[[0, 3]].copy()
start_v = state[[2, 5]].copy()
start_a = np.array([
    (Ax_num @ state[:3])[2],
    (Ay_num @ state[3:6])[2]
])
stop_coeffs = [
    braking_coefficients(start_p[0], start_v[0], start_a[0], T_stop),
    braking_coefficients(start_p[1], start_v[1], start_a[1], T_stop)
]
stop_start = t_now


def stop_inputs(t):
    vx, ax, jx = polynomial_derivatives(stop_coeffs[0], t-stop_start, T_stop)
    vy, ay, jy = polynomial_derivatives(stop_coeffs[1], t-stop_start, T_stop)
    ux = axis_inverse_voltage(vx, ax, jx, cx_num)
    uy = axis_inverse_voltage(vy, ay, jy, cy_num)
    return [(ux + uy) / 2, (ux - uy) / 2, u_hold]


state, t_now, sol_stop = integrate(
    state, t_now, t_now + T_stop, mode, stop_inputs
)
assert np.linalg.norm(state[[2, 5]]) < 1e-5
assert np.linalg.norm(state[[1, 4]]) < 1e-5
print(f"Printing stopped smoothly at t = {t_now:.3f} s")

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
    state, t_now, t_now + wiping_duration, mode,
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
    state, t_now, t_now + loading_duration, mode,
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
# Step 6b: Verify that invalid transitions are rejected
#------------------------------------

def expect_rejection(description, test_mode, event, test_state, test_time):
    try:
        update_fsm(
            test_mode, event, test_state,
            saved_position, test_time
        )
    except RuntimeError:
        print(f"PASS: {description}")
    else:
        raise AssertionError(
            f"Invalid transition was accepted: {description}"
        )

# Test 1: Arrival event before reaching the wipe tower
wrong_position = state.copy()
wrong_position[[0, 3]] = [0.05, 0.05]
expect_rejection(
    "Premature wipe-tower arrival rejected",
    "Moving to wipe tower",
    "wipe_tower_reached",
    wrong_position,
    t_now
)

# Prepare a stationary state at the wipe tower
parked_state = state.copy()
parked_state[[0, 2, 3, 5]] = 0.0

# Test 2: Cooling event while nozzle is still too hot
parked_state[6] = 100.0
expect_rejection(
    "Premature cooling event rejected",
    "Cooling nozzle",
    "nozzle_cold",
    parked_state,
    t_now
)

# Test 3: Heating event below the high threshold
parked_state[6] = 190.0
expect_rejection(
    "Premature heating event rejected",
    "Heating nozzle",
    "nozzle_hot",
    parked_state,
    t_now
)

# Test 4: Wiping completion after only one second
parked_state[6] = T_high
expect_rejection(
    "Premature wiping completion rejected",
    "Wiping",
    "wiping_done",
    parked_state,
    mode_entry_time["Wiping"] + 1.0
)
print("\nHybrid guard tests passed.")

#------------------------------------
# Step 7: Visualization
#------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Tool-head motion
axes[0].plot(
    sol_print.y[0], sol_print.y[3], label="Printed ellipse"
)
axes[0].plot(
    sol_stop.y[0], sol_stop.y[3], label="Controlled stop"
)
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

#------------------------------------
# Step 8: Filament cooling with deposition times from actual motor motion
#------------------------------------
# Thermal parameters unchanged from Q1.6v4
rho_fil = 1240.0
cp_fil = 1800.0
k_fil = 0.13
w_fil = 0.0005
h_air = 30.0
h_bed = 600.0
T_air = 22.0
T_bed = 60.0
T_dep = 200.0
alpha_fil = k_fil / (rho_fil * cp_fil)
beta_air = 3 * h_air / (rho_fil * cp_fil * w_fil)
beta_bed = h_bed / (rho_fil * cp_fil * w_fil)


def thermal_dynamics(t, temperature, ds, periodic=False):
    laplacian = np.zeros_like(temperature)
    if periodic:
        laplacian = (
            np.roll(temperature, 1) - 2 * temperature
            + np.roll(temperature, -1)
        ) / ds**2
    elif len(temperature) > 1:
        laplacian[1:-1] = (
            temperature[:-2] - 2 * temperature[1:-1]
            + temperature[2:]
        ) / ds**2
        laplacian[0] = (temperature[1] - temperature[0]) / ds**2
        laplacian[-1] = (temperature[-2] - temperature[-1]) / ds**2
    return (
        alpha_fil * laplacian
        - beta_air * (temperature - T_air)
        - beta_bed * (temperature - T_bed)
    )


# Deposition times calculated from the simulated trajectory, not the ideal one
N_fil = 200
time_geom = np.linspace(0, T_print, 2001)
printed_states = sol_print.sol(time_geom)
print_speed = np.hypot(printed_states[2], printed_states[5])
printed_length = cumulative_trapezoid(print_speed, time_geom, initial=0)
L_fil = printed_length[-1]
ds_fil = L_fil / N_fil
s_fil = (np.arange(N_fil) + 0.5) * ds_fil
s_edges = np.linspace(0, L_fil, N_fil + 1)
t_edges = np.interp(s_edges, printed_length, time_geom)
assert np.all(np.diff(printed_length) > 0)
assert np.all(np.diff(t_edges) > 0)

# Reuse the Q1.6 growing-domain discretization
filament_temperature = np.array([T_dep])
for i in range(N_fil):
    sol = solve_ivp(
        lambda t, T: thermal_dynamics(t, T, ds_fil),
        (t_edges[i], t_edges[i + 1]),
        filament_temperature,
        method="BDF", rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    filament_temperature = sol.y[:, -1]
    if i < N_fil - 1:
        filament_temperature = np.append(filament_temperature, T_dep)
T_fil_initial = filament_temperature.copy()

uniformity_tolerance = 1.0


def filament_uniformity_event(t, temperature):
    return np.ptp(temperature) - uniformity_tolerance


filament_uniformity_event.terminal = False
filament_uniformity_event.direction = -1

# The closed filament keeps cooling from t = 20 s onward
sol_filament = solve_ivp(
    lambda t, T: thermal_dynamics(t, T, ds_fil, periodic=True),
    (T_print, t_now),
    T_fil_initial,
    method="BDF",
    events=filament_uniformity_event,
    dense_output=True,
    rtol=1e-7, atol=1e-9
)
if not sol_filament.success:
    raise RuntimeError(sol_filament.message)
if len(sol_filament.t_events[0]) == 0:
    raise RuntimeError("Filament uniformity criterion not reached.")
t_fil_uniform = float(sol_filament.t_events[0][0])
T_fil_uniform = sol_filament.sol(t_fil_uniform)
T_fil_final = sol_filament.sol(t_now)
T_eq = (3*h_air*T_air + h_bed*T_bed) / (3*h_air + h_bed)
assert np.ptp(T_fil_final) < uniformity_tolerance
assert np.all(np.isfinite(T_fil_final))

print("\nFilament cooling coupled to simulated deposition")
print(f"Filament length: {L_fil:.5f} m")
print(f"Filament uniform after printing: {t_fil_uniform - T_print:.3f} s")
print(f"Final temperature spread: {np.ptp(T_fil_final):.6f} deg C")
print(f"Equilibrium temperature: {T_eq:.2f} deg C")

#------------------------------------
# Step 9: Filament temperature visualization
#------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(s_fil, T_fil_initial, label="Printing completed")
axes[0].plot(s_fil, T_fil_uniform, label="Nearly uniform")
axes[0].plot(s_fil, T_fil_final, "--", label="Replacement completed")
axes[0].set(
    title="Printed filament temperature",
    xlabel="Position along filament [m]",
    ylabel="Temperature [deg C]"
)
axes[0].grid(True)
axes[0].legend()

time_cool = np.linspace(T_print, t_now, 500)
filament_spread = np.ptp(sol_filament.sol(time_cool), axis=0)
axes[1].plot(time_cool - T_print, filament_spread,
             label="Temperature difference")
axes[1].axhline(uniformity_tolerance, linestyle="--",
                label="1 deg C threshold")
axes[1].axvline(t_fil_uniform - T_print, linestyle=":",
                label="Uniformity reached")
axes[1].set(
    title="Filament cooling during replacement",
    xlabel="Time since replacement started [s]",
    ylabel="Max(T) - Min(T) [deg C]"
)
axes[1].set_xlim(0, 30)
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.show()