import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.dynamics import U
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


n = 20
pi = 2
x = np.random.uniform(-5, 5, size=(n, 3 * n))  # x, xdot, xddot
y = U(x, 1, 1, 1, 2, 3, pi)


def loss(k):
    return ((y - U(x, 1, 1, 1) - k[0] * (x[:, 0] - pi) - k[1] * x[:, 1]) ** 2).sum()


def train(n_iterations=100):
    res = gp_minimize(
        loss,  # the function to minimize
        [(-5.0, 5.0), (-5.0, 5.0)],  # the bounds on each dimension of x
        acq_func="EI",  # the acquisition function
        n_calls=30,  # the number of evaluations of f
        # n_random_starts=5,  # the number of random initialization points
        random_state=1234,
    )
    print("x_0=%.4f, x_1=%.4f f(x)=%.4f" % (res.x[0], res.x[1], res.fun))
