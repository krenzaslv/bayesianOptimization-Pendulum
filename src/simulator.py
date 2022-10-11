from src.integrator import integrate
import numpy as np


def simulate():
    pi = 2
    kp = 10.0
    kd = 1
    x_t = np.array([0.01, 0])

    df = lambda x_t: np.array([[0, 1], [0, 0]]) @ x_t + np.array([0, 1]) * (
        kp * (pi - x_t[0]) - kd * x_t[1]
    )

    while True:
        x_t = integrate(x_t, df, 1e-3)
