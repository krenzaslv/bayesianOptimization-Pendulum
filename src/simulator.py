from src.dynamics import dynamics, U
import time
import numpy as np


def simulate(config):

    x_t = config.x0
    X = np.zeros(shape=(config.n_simulation, 2))
    for i in range(config.n_simulation):
        x_t = dynamics(x_t, U(x_t, config), config)
        X[i, :] = x_t
    return X
