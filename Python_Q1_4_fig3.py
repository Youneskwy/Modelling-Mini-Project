import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- Jouw originele matrices ---
A_x = np.array([
    [0 , 0, 1],
    [0 , -5, -10],
    [0 , 10, -5]
])
B_x_base = np.array([[0], [1/0.05], [0]])
C_x = np.array([[1, 0, 0]])

A_y = np.array([
    [0 , 0, 1],
    [0 , -5, -10],
    [0 , 10, -5]
])
B_y_base = np.array([[0], [1/0.05], [0]])
C_y = np.array([[1, 0, 0]])

A_mimo = np.block([
    [A_x, np.zeros((3, 3))],
    [np.zeros((3, 3)), A_y]
])

B_mimo = np.block([
    [B_x_base, np.zeros((3, 1))],
    [np.zeros((3, 1)), B_y_base]
])

C_mimo = np.block([
    [C_x, np.zeros((1, 3))],
    [np.zeros((1, 3)), C_y]
])

D_mimo = np.zeros((2, 2))

sys_mimo = signal.StateSpace(A_mimo, B_mimo, C_mimo, D_mimo)

# --- Ludieke Input: De Spirograaf (Bloem) ---
# We nemen veel puntjes (5000) en tekenen lang (20 pi) voor een vloeiende lijn
t = np.linspace(0, 20 * np.pi, 5000)

# u_x en u_y zijn een mix van een trage golf (t) en een snelle golf (7t).
# Let op hoe de snelle golf amplitude 6 heeft, dit is om de demping 
# van je model te compenseren!
u_x = 3 * np.cos(t) + 6 * np.cos(7 * t)
u_y = 3 * np.sin(t) - 6 * np.sin(7 * t)

U = np.vstack((u_x, u_y)).T

# Simuleren
_, Y, _ = signal.lsim(sys_mimo, U, t)

y_x = Y[:, 0]
y_y = Y[:, 1]

# --- Plotten ---
plt.figure(figsize=(8, 8))

# Nog een leuke truc: we laten de eerste 400 meetpunten (het inslingeren 
# vanuit stilstand naar volle snelheid) weg in de plot.
# Hierdoor zie je alleen het perfecte afgeronde patroon!
plt.plot(y_x[400:], y_y[400:], color='magenta', linewidth=1.5)

plt.xlabel("Positie X-as")
plt.ylabel("Positie Y-as")
plt.title("Spirograaf / Bloempatroon")
plt.grid(True)
plt.axis('equal')
plt.show()
