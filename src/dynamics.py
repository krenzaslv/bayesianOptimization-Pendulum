import numpy as np
import math

# c = config
def U(x, c):
    return (
        c.m
        * c.l
        * c.l
        * (-c.k1 * (x[:, 0] - c.pi) - c.k2 * x[:, 1] - c.g / c.l * np.sin(x[:, 0]))
        + c.k_p * (x[:, 0] - c.pi)
        + c.k_d * x[:, 1]
    )


# c = config
def dynamics(x, U, c, dt=1e-3):
    df = np.array([x[:, 1], c.g / c.l * np.sin(x[:, 0]) + U / (c.m * c.l * c.l)]).T
    return x + dt * df
