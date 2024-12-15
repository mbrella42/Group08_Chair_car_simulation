import numpy as np
import matplotlib.pyplot as plt

m1, d1, k1 = 1400, 1000, 1000
m2, d2, k2 = 100, 50, 50

def euler_method(A, B, x0, u, dt, T):
    n_steps = int(T / dt)
    x = np.zeros((n_steps, len(x0)))
    x[0] = x0

    u = u.reshape(-1, 1)
    for i in range(1, n_steps):
        x[i] = x[i - 1] + dt * (A @ x[i - 1] + B @ u[i - 1])
    return x


A = np.array([
    [0, 1, 0, 0],
    [-(k1 + k2) / m1, -(d1 + d2) / m1, k2 / m1, d2 / m1],
    [0, 0, 0, 1],
    [k2 / m2, d2 / m2, -k2 / m2, -d2 / m2]
])

#all zeroes for 3
B = np.array([
    [0],
    [0],
    [0],
    [0]
])


def simulate_and_plot(x0, T=30, dt=0.001):
    time = np.arange(0, T, dt)
    x = euler_method(A, B, x0, u=np.zeros((len(time), 1)), dt=dt, T=T)

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print("Fail due to instability.")
        return

    y1 = 0.3 + x[:, 0]
    y2 = 0.6 + x[:, 2]

    plt.figure(figsize=(10, 6))
    plt.plot(time, y1, label="Body Position")
    plt.plot(time, y2, label="Chair Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("System Response")
    plt.legend()
    plt.grid()
    plt.show()



x0_1 = [0, -0.2, 0, 0]
x0_2 = [0, 0, 0, -0.2]

print("chair initial speed -0.2:")
simulate_and_plot(x0_1)

print("body initial speed -0.2")
simulate_and_plot(x0_2)

