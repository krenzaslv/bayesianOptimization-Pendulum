import numpy as np


def U(x, m, l, g, k_d=0.0, k_p=0.0, pi=0):
    return m * l * l * (k_p * (x[:, 0] - pi) + k_d * x[:, 1] - g / l * np.sin(x[:, 0]))
