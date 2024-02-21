import os
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation

os.system('cls')

# Creating an function for recalling the way of solution.
last_called_function = None

def set_last_called_function(func_name):
    global last_called_function
    last_called_function = func_name


# Constants
k = 1.0 # k is stiffness constant of springs, in N/m, N is Newton, m is meter.
m = 1.0 # m is mass, in kg.
L = 5.0 # Rest length of springs.
dt = 0.01 # time step.

# Masses of each ball is defined as following
M1 = 1*m
M2 = 3*m
M3 = 2*m

B = [M1,M2,M3]

# Stiffness contants of each spring is defined as following
S1 = 1*k
S2 = 2*k

S = [S1,S2]

# Positions of each ball defined as following
x1 = float(rnd.randrange(0,10))
x2 = float(rnd.randrange((x1+1),(x1+11)))
x3 = float(rnd.randrange((x2+1),(x2+11)))


# Velocities of each ball defined as following
v1 = float(rnd.randrange(-10,10))
v2 = float(rnd.randrange(-10,10))
v3 = float(rnd.randrange(-10,10))

t = 0.0

# Euler Method Simulation
t_values = []
Mx1 = []
Mx2 = []
Mx3 = []
V1 = []
V2 = []
V3 = []


#Verlet integration

def Verlet_integration(t_values,Mx1,Mx2,Mx3,V1,V2,V3,S,B,k,L,dt,x1,x2,x3,v1,v2,v3,t):

    set_last_called_function("Verlet_integration")

    i = 0
    while t <= 20.0:
        t_values.append(t)
        Mx1.append(x1)
        Mx2.append(x2)
        Mx3.append(x3)
        V1.append(v1)
        V2.append(v2)
        V3.append(v3)

        # Distance between two balls
        L1 = x2 - x1
        L2 = x3 - x2

        # Acceleration of the Balls
        a1 = S[0] * ( L1 - L ) / B[0]  # Acceleration
        a2 = ( -S[0] * ( L1 - L ) + S[1] * ( L2 - L) ) / B[1]
        a3 = -S[1] * ( L2 - L ) / B[2]

        # Update on the positions using Euler Method.
        x1 = x1 + v1 * dt + a1 * dt**2 / 2  # Verlet integration for next position of first Ball
        x2 = x2 + v2 * dt + a2 * dt**2 / 2  # Verlet integration for next position of second Ball
        if x1 > x2:
            x1 = Mx1[i]
            x2 = Mx2[i]
            v1_prime = ( v1 * ( B[0] - B[1] ) + 2 * B[1] * v2 ) / ( B[0] + B[1] )
            v2_prime = ( v2 * ( B[1] - B[0] ) + 2 * B[0] * v1 ) / ( B[1] + B[0] )
            x1 = x1 + v1_prime * dt + a1 * dt**2 / 2
            x2 = x2 + v2_prime * dt + a2 * dt**2 / 2
            v1 = v1_prime
            v2 = v2_prime
        x3 = x3 + v3 * dt + a3 * dt**2 / 2  # Verlet integration for next position of third Ball
        if x2 > x3:
            x3 = Mx3[i]
            x2 = Mx2[i]
            v2_prime = ( v2 * ( B[1] - B[2] ) + 2 * B[2] * v3 ) / ( B[1] + B[2] )
            v3_prime = ( v3 * ( B[2] - B[1] ) + 2 * B[1] * v2 ) / ( B[2] + B[1] )
            x2 = x2 + v2_prime * dt + a2 * dt**2 / 2
            x3 = x3 + v3_prime * dt + a3 * dt**2 / 2
            v2 = v2_prime
            v3 = v3_prime

        # Calculation of the next acceleration of the balls
        
        # Distance between two balls
        L1_next = x2 - x1
        L2_next = x3 - x2

        # Acceleration of the Balls
        a1_next = k * ( L1_next - L ) / B[0]  # Acceleration
        a2_next = ( -k * ( L1_next - L ) + k * ( L2_next - L) ) / B[1]
        a3_next = -k * ( L2_next - L ) / B[2]

        # Update on the velocities using Euler Method.
        v1 = v1 + (a1 + a1_next) / 2 * dt    # Verlet integration for velocity of first Ball
        v2 = v2 + (a2 + a2_next) / 2 * dt    # Verlet integration for velocity of sdecond Ball
        v3 = v3 + (a3 + a3_next) / 2 * dt    # Verlet integration for velocity of third Ball


        t += dt
        i += 1

    #Plotting the results
    plt.clf()
    plt.plot(t_values, Mx1, label="Ball 1")
    plt.plot(t_values, Mx2, label="Ball 2")
    plt.plot(t_values, Mx3, label="Ball 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Harmonic Oscillator Simulation using Verlet Integration")
    plt.grid(True)
    plt.show()

# Time Evolution Operator

def Time_Evolution_Operator(t_values,Mx1,Mx2,Mx3,V1,V2,V3,S,B,k,L,dt,x1,x2,x3,v1,v2,v3,t):

    set_last_called_function("Time_Evolution_Operator")

    fig, ax = plt.subplots()

    # Define a matrix
    A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [-S[0] / B[0], S[0] / B[0], 0, 0, 0, 0],
    [S[0] / B[1], -(S[0] + S[1]) / B[1], S[1] / B[1], 0, 0, 0],
    [0, S[1] / B[2], -S[1] / B[2], 0, 0, 0]
    ], dtype=float)

    # Compute the Time Evolution Operator
    U = expm(np.array(A)*dt)

    r = np.array([x1, x2, x3, v1, v2, v3])  # initial state vector

    while t <= 20.0:
        t_values.append(t)
        Mx1.append(r[0])
        Mx2.append(r[1] + L)
        Mx3.append(r[2] + (2*L))

        # Update the state vector
        r = np.dot(U, r)

        t += dt

    # Plotting the results
    plt.plot(t_values, Mx1, label="Ball 1")
    plt.plot(t_values, Mx2, label="Ball 2")
    plt.plot(t_values, Mx3, label="Ball 3")
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title("Harmonic Oscillator Simulation using Time Evolution Operator")
    plt.legend()
    plt.grid(True)
    plt.show()

# Creating an entry for choosing the method.
while True:
    try:
        haydaa = int(input("For Verlet Integration, Enter '0'\nFor Time Evolution Operator, Enter '1'\nEntry:   "))
        if haydaa == 1 or haydaa == 0:
            break
        else:
            print("!!!  INCORRECT ENTRY, TRY AGAIN  !!!")
    except ValueError:
        print("!!!  INCORRECT ENTRY, TRY AGAIN  !!!")

if haydaa == 1:
    yeter = Time_Evolution_Operator(t_values, Mx1, Mx2, Mx3, V1, V2, V3, S, B, k, L, dt, x1, x2, x3, v1, v2, v3, t)
elif haydaa == 0:
    aman = Verlet_integration(t_values, Mx1, Mx2, Mx3, V1, V2, V3, S, B, k, L, dt, x1, x2, x3, v1, v2, v3, t)

# Creating animation for the balls' movement
fig, ax = plt.subplots()
allah = (((max(Mx3)+5) - (min(Mx1)-5)) * 4 / 8)*2 / 50
circle3 = plt.Circle((t_values, Mx1[0]), (allah), fc='r')
circle1 = plt.Circle((t_values, Mx2[0]), (3*allah), fc='b')
circle2 = plt.Circle((t_values, Mx3[0]), (2*allah), fc='g')
# line1, = ax.plot([], [], linewidth = '0.5')[0]
# line2, = ax.plot([], [], linewidth = '0.5')[0]


def init():
    global last_called_function
    if  last_called_function == "Verlet_integration":
        plt.title("Verlet Integration")
    elif last_called_function == "Time_Evolution_Operator":
        plt.title("Time Evolution Operator")
    plt.xlabel('Position')
    if Mx1[-1] > Mx1[0]:
        ax.set_xlim((min(Mx1)-5), (max(Mx3)+5))
        arsa = ((max(Mx3)+5) - (min(Mx1)-5)) * 5 / 16
        ax.set_ylim(-arsa, arsa)
    else:
        ax.set_xlim((min(Mx1)-5), (max(Mx3)+5))
        arsa = ((max(Mx3)+5) - (min(Mx1)-5)) * 5 / 16
        ax.set_ylim(-arsa, arsa)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    # line1.set_data([], [])
    # line2.set_data([], []) 
    return circle1, circle2, circle3, #line1, line2

a1, a2 = [], []

def animate(i):
    x1 = Mx1[i]
    x2 = Mx2[i]
    x3 = Mx3[i]
   # a1 = (Mx2[i]-Mx1[i])
   # a2 = (Mx3[i]-Mx2[i])
   # line1.set_data(x2, x1)
   # line2.set_data(x3, x2)
    circle1.center = (x2, 0)
    circle2.center = (x3, 0)
    circle3.center = (x1, 0)
    return circle1, circle2, circle3, #line1, line2

animation = FuncAnimation(fig, animate, init_func=init, frames=len(t_values), interval=1, blit=True)
plt.show() 



