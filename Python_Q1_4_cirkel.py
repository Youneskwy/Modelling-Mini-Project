import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


A_x = np.array([
    [0 , 0, 1],
    [0 , -3, -2],
    [0 , 2, -3]
    ])
B_x_base = np.array([[0], [1/0.05], [0]])
#voor output meten we enkel de positie
C_x = np.array([[1,0,0]])

A_y = np.array([
    [0 , 0, 1],
    [0 , -3, -2],
    [0 , 2, -3]
    ])
B_y_base = np.array([[0], [1/0.05], [0]])
#voor output meten we enkel de positie
C_y = np.array([[1,0,0]])

#MIMO State space (4 states, 2 outputs, 2 inputs)

A_mimo = np.block([
    [A_x, np.zeros((3, 3))],
    [np.zeros((3, 3)), A_y]
    ])

B_mimo = np.block([
    [B_x_base, np.zeros((3, 1))],
    [np.zeros((3,1)), B_y_base]
    ])

C_mimo = np.block([
    [C_x, np.zeros((1, 3))],
    [np.zeros((1, 3)), C_y]
    ])

D_mimo = np.zeros((2,2))


sys_mimo = signal.StateSpace(A_mimo, B_mimo, C_mimo, D_mimo)

t = np.linspace(0,15*np.pi,2000)

#u_x = (5 * np.cos(t)) / (1 + (np.sin(t))**2)
#u_y = (5 * np.sin(t) * np.cos(t)) / (1 + (np.sin(t))**2)

u_x = 4 * np.cos(t)
u_y = 2 * np.cos(2 * t)

U = np.vstack((u_x, u_y)).T

_, Y, _ = signal.lsim(sys_mimo, U, t)

y_x = Y[:,0]
y_y = Y[:,1]

plt.figure(figsize=(7,7))
plt.plot(y_x, y_y, color='red')
plt.xlabel("Positie X-as")
plt.ylabel("Positie Y-as")
plt.grid(True)
plt.axis('equal')
plt.show()
