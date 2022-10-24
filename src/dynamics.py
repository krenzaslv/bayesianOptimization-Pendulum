import numpy as np
from src.integrator import integrate
from src.tools import clamp, angleDiff


# c = config
def U_star(x, c):
    return clamp(
        (
            c.m
            * c.L
            * c.L
            * (-c.k1 * angleDiff(x[0], c.pi) - c.k2 * x[1] - c.g / c.L * np.sin(x[0]))
        ),
        c.max_torque,
    )


def U_bo(x, c):
    return clamp(
        U_star(x, c) + c.kp_bo * angleDiff(x[0], c.pi) + c.kd_bo * x[1],
        c.max_torque,
    )


def U_pert(x, c, U):
    return U(x, c) - c.kp * angleDiff(x[0], c.pi) - c.kd * x[1]


def dynamics_ideal(x, U, c):
    def f(u): return np.array(
        [u[1], c.g / c.L * np.sin(u[0]) + U(u, c) / (c.m * c.L * c.L)]
    ).T
    res = integrate(x, f, c.dt)
    return res


def dynamics_real(x, U, c):
    def U_p(t, c): return U_pert(t, c, U)  # Disturbance
    return dynamics_ideal(x, U_p, c)
