import numpy as np
import math
from src.integrator import integrate
import copy
from src.tools import rad, clamp

# c = config
def U_star(x, c):
    return clamp(
        (
            c.m
            * c.l
            * c.l
            * (-c.k1 * (rad(x[0]) - c.pi) - c.k2 * x[1] - c.g / c.l * np.sin(x[0]))
        ),
        c.max_torque,
    )


def U_bo(x, c):
    return clamp(
        U_star(x, c) - c.kp_bo * (rad(x[0]) - c.pi) - c.kd_bo * x[1],
        c.max_torque,
    )


def dynamics_ideal(x, U, c):
    f = lambda t: np.array(
        [t[1], c.g / c.l * np.sin(t[0]) + U(x, c) / (c.m * c.l * c.l)]
    ).T
    res = integrate(x, f, c.dt)
    # res[0] = rad(res[0])
    return res


def dynamics_real(x, U, c):
    U_pert = (
        lambda t, c: U(t, c) + c.kp * (rad(t[0]) - c.pi) + c.kd * t[1]
    )  # Disturbance
    return dynamics_ideal(x, U_pert, c)
