import numpy as np
import math

# c = config
def U_star(x, c):
    return (
        c.m
        * c.l
        * c.l
        * (-c.k1 * (x[:, 0] - c.pi) - c.k2 * x[:, 1] - c.g / c.l * np.sin(x[:, 0]))
    )


def U_bo(x, c):
    return U_star(x, c) - c.k_p_bo * (x[:, 0] - c.pi) - c.k_d_bo * x[:, 1]


def dynamics_ideal(x, U, c, dt=1e-3):
    df = np.array([x[:, 1], c.g / c.l * np.sin(x[:, 0]) + U / (c.m * c.l * c.l)]).T
    return x + dt * df


def dynamics_real(x, U, c, dt=1e-3):
    U_pert = U + c.k_p * (x[:, 0] - c.pi) + c.k_d * x[:, 1]  # Disturbance
    return dynamics_ideal(x, U_pert, c, dt)
