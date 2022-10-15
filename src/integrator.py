import numpy as np
import scipy


def integrate(x, f, dt):
    # x_new = np.zeros_like(x)
    # x_dot = f(x)
    # x_new[:, 1] = x[:, 1] + dt * x_dot[:, 1]
    # x_new[:, 0] = x[:, 0] + dt * x_new[:, 1]
    _f = lambda t, x: f(x)
    sol = scipy.integrate.solve_ivp(_f, [0, dt], x)
    return sol.y[:, sol.y.shape[1] - 1]
