import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.simulator import simulate
from src.dynamics import U, dynamics
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy


class Trainer:
    def __init__(self, config):
        self.config = config

        self.X = simulate(config)

        config_star = copy.copy(config)
        config_star.k_d = 0
        config_star.k_p = 0
        # To this we don't normally have access
        self.X_star = simulate(config)

    def loss(self, k):
        config_bo = copy.copy(self.config)
        config_bo.k_d = k[0]
        config_bo.k_p = k[1]
        # X_bo = simulate(config_bo)
        X_bo = dynamics(self.X, U(self.X, config_bo), self.config)
        return ((self.X[1:, :] - X_bo[: X_bo.shape[0] - 1, :]) ** 2).sum().sum()

    def train(self):
        res = gp_minimize(
            self.loss,  # the function to minimize
            [(-5.0, 5.0), (-5.0, 5.0)],  # the bounds on each dimension of x
            acq_func="EI",  # the acquisition function
            n_calls=self.config.n_iterations,  # the number of evaluations of f
            n_random_starts=3,  # the number of random initialization points
            random_state=1234,
        )
        print("x_0=%.4f, x_1=%.4f f(x)=%.4f" % (res.x[0], res.x[1], res.fun))
