import time
import numpy as np


def simulate(config, dynamics, U):

    x_t = config.x0
    X = np.zeros(shape=(config.n_simulation, 2))
    for i in range(config.n_simulation):
        X[i, :] = x_t
        x_t = dynamics(x_t, U, config)
    return X
