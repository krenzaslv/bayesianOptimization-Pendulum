import numpy as np
import math
from src.integrator import integrate

# c = config
def U_star(x, c):
    return (
        c.m
        * c.l
        * c.l
        * (-c.k1 * (x[:, 0] - c.pi) - c.k2 * x[:, 1] - c.g / c.l * np.sin(x[:, 0]))
    )


def U_bo(x, c):
    return U_star(x, c) - c.kp_bo * (x[:, 0] - c.pi) - c.kd_bo * x[:, 1]


def dynamics_ideal(x, U, c):
    x_dot = np.array([x[:, 1], c.g / c.l * np.sin(x[:, 0]) + U / (c.m * c.l * c.l)]).T
    return integrate(x, x_dot, c.dt)


def dynamics_real(x, U, c):
    U_pert = U + c.kp * (x[:, 0] - c.pi) + c.kd * x[:, 1]  # Disturbance
    return dynamics_ideal(x, U_pert, c)
