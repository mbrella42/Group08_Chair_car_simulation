import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

m1, d1, k1 = 1400, 1000, 1000
m2, d2, k2 = 100, 200, 100

def euler_method(x0, u, dt, T, d1_c = d1):
    d1 = d1_c

    A = np.array([
        [0, -(k1 + k2) / m1, 0, k2 / m1],
        [1, -(d1 + d2) / m1, 0, d2 / m1],
        [0, k2 / m2, 0, -k2 / m2],
        [0, d2 / m2, 1, -d2 / m2]
    ])

    B = np.array([
        [k1 / m1],
        [d1 / m1],
        [0],
        [0]
    ])

    n_steps = int(T / dt)
    x = np.zeros((n_steps, len(x0)))
    x[0] = x0

    # u = u.reshape(-1, 1)
    for i in range(1, n_steps):
        x[i] = x[i - 1] + dt * (A @ x[i - 1] + B @ u[i - 1])
    return x

def simulate_and_plot(x0, u, description, T=20, dt=0.001):
    time = np.arange(0, T, dt)
    x = euler_method(x0, u, dt=dt, T=T)

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print("Fail due to instability.")
        return

    y1 = 0.3 + x[:, 1]
    y2 = 0.6 + x[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(time, y1, label="Body Position")
    plt.plot(time, y2, label="Chair Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("System Response for " + description)
    plt.legend()
    plt.grid()
    plt.show()

def simulate_and_plot_with_ut(x0, u, description, T=20, dt=0.001):
    time = np.arange(0, T, dt)
    x = euler_method(x0, u, dt=dt, T=T)

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print("Fail due to instability.")
        return

    y1 = 0.3 + x[:, 1]
    y2 = 0.6 + x[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(time, y1, label="Body Position")
    plt.plot(time, y2, label="Chair Position")
    plt.plot(time, u, label="Road Input")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("System Response for " + description)
    plt.legend(loc='lower right', fontsize='small')
    plt.grid()
    plt.show()
    return x

def simulate_and_plot_5(x0, u, d1_array, description, T=20, dt=0.001):
    time = np.arange(0, T, dt)

    plt.figure(figsize=(10, 6))

    for d1_diff in d1_array:
        x = euler_method(x0, u, dt=dt, T=T, d1_c=d1_diff)  # You might need to pass d1_diff here

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"Simulation failed for d1 = {d1_diff} due to instability.")
            continue

        y1 = 0.3 + x[:, 1]
        y2 = 0.6 + x[:, 3]

        plt.plot(time, y1, label=f"Body Position (d1={d1_diff})")
        plt.plot(time, y2, '--', label=f"Chair Position (d1={d1_diff})")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("System Response for " + description)
    plt.legend(loc='lower right', fontsize='small')
    plt.grid()
    plt.show()

#Assignment 5
def input_5(dt=0.001):
    u = np.zeros((20000, 1))
    for i in range(20000):
        if i*dt < 1:
            u[i] = 0
        else:
            u[i] = 1
    return u

def input_6(dt, v):
    u = np.zeros((20000, 1))
    for i in range(20000):
        z = i*dt*v
        if z < 5 or z > 9.5:
            u[i] = 0
        else:
            u[i] = 0.1 * np.sin((np.pi / 4.5) * (z - 5))
    return u

def input_7(dt, v, s):
    u = np.zeros((20000, 1))
    for i in range(20000):
        z = i*dt*v
        if z < 5 or 9.5 < z < 9.5 + s or z > 14 + s:
            u[i] = 0
        elif 5 < z < 9.5:
            u[i] = 0.1 * np.sin((np.pi / 4.5) * (z - 5))
        else:
            u[i] = 0.1 * np.sin((np.pi / 4.5) * (z - (9.5 + s)))
    return u

def task3():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 50, 50

    x0_1 = [0, -0.2, 0, 0]
    x0_2 = [0, 0, 0, -0.2]

    print("body initial location -0.2:")
    simulate_and_plot(x0_1, np.zeros((20000, 1), ), "body initial location -0.2")

    print("chair initial location -0.2")
    simulate_and_plot(x0_2, np.zeros((20000, 1), ), "chair initial location -0.2")

def task4():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 200, 100

    x0_1 = [0, -0.2, 0, 0]
    x0_2 = [0, 0, 0, -0.2]

    print("body initial location -0.2:")
    simulate_and_plot(x0_1, np.zeros((20000, 1), ), "body initial location -0.2")

    print("chair initial location -0.2")
    simulate_and_plot(x0_2, np.zeros((20000, 1), ), "chair initial location -0.2")

def task5():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 200, 100

    d1_diff = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    x0 = [0, 0, 0, 0]

    print("sudden 1 m bump with different d1")
    simulate_and_plot_5(x0, input_5(0.001),d1_diff,"1m bump with different d1")

def task6():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 200, 100

    x0 = [0, 0, 0, 0]

    print("sinusoidal profile bump with v=3")
    x = simulate_and_plot_with_ut(x0, input_6(0.001, v=1),"sinusoidal profile bump with v=3")

    max_distance = 0
    for state in x:
        if (state[3] + 0.6) - (state[1]  + 0.3) > max_distance:
            max_distance = (state[3] + 0.6) - (state[1]  + 0.3)

    print("y2-y1 max : " + str(max_distance))

def task7():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 200, 100

    x0 = [0, 0, 0, 0]

    print("Two bumps, s=1.5")
    x = simulate_and_plot_with_ut(x0, input_7(0.001, 3, 1.5),"two sinusoidal profile bumps,s=1.5")

    y1 = 0.3 + x[:, 1]
    y2 = 0.6 + x[:, 3]

    y_diff = y2 - y1

    peaks, _ = find_peaks(y_diff)

    sorted_peaks = sorted(peaks, key = lambda p: y_diff[p], reverse=True)[:2]

    bump_values = [y_diff[p] for p in sorted_peaks]

    print("y2-y1 max:", bump_values)
    print("Peak indices/time:", sorted_peaks)

    print("Two bumps,s=0")
    x = simulate_and_plot_with_ut(x0, input_7(0.001, 3, 0), "two sinusoidal profile bumps,s=0")

    y1 = 0.3 + x[:, 1]
    y2 = 0.6 + x[:, 3]

    y_diff = y2 - y1

    peaks, _ = find_peaks(y_diff)

    sorted_peaks = sorted(peaks, key=lambda p: y_diff[p], reverse=True)[:2]

    bump_values = [y_diff[p] for p in sorted_peaks]

    print("y2-y1 max:", bump_values)
    print("Peak indices/time:", sorted_peaks)

def task8():
    global m1, d1, k1
    m1, d1, k1 = 1400, 1000, 1000
    global m2, d2, k2
    m2, d2, k2 = 100, 200, 100

    x0 = [0, 0, 0, 0]

    print("sinusoidal profile bump with v=3")
    x = simulate_and_plot_with_ut(x0, input_6(0.001, v=3),"sinusoidal profile bump with v=3")

    max_distance = 0
    for state in x:
        if (state[3] + 0.6) - (state[1]  + 0.3) > max_distance:
            max_distance = (state[3] + 0.6) - (state[1]  + 0.3)

    print("y2-y1 max : " + str(max_distance))

    print("sinusoidal profile bump v=10")
    x = simulate_and_plot_with_ut(x0, input_6(0.001, v=10),"sinusoidal profile bump with v=10")

    max_distance = 0
    for state in x:
        if (state[3] + 0.6) - (state[1]  + 0.3) > max_distance:
            max_distance = (state[3] + 0.6) - (state[1]  + 0.3)

    print("y2-y1 max : " + str(max_distance))

task3()
task4()
task5()
task6()
task7()
task8()
