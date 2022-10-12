import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.simulator import simulate
from src.dynamics import U_bo, dynamics_real
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star

    def loss(self, k):
        self.config.k_d_bo = k[0]
        self.config.k_p_bo = k[1]
        X_bo = simulate(self.config, dynamics_real, U_bo)
        return ((self.X_star - X_bo) ** 2).sum().sum()

    def train(self):
        res = gp_minimize(
            self.loss,  # the function to minimize
            [(-5.0, 5.0), (-5.0, 5.0)],  # the bounds on each dimension of x
            acq_func="LCB",  # the acquisition function
            n_calls=self.config.n_iterations,  # the number of evaluations of f
            n_random_starts=3,  # the number of random initialization points
            # xi=[self.config.k_p_bo, self.config.k_p_bo],
            noise=1e-10,
            n_jobs=-1,
            random_state=1234,
        )
        print("x_0=%.4f, x_1=%.4f f(x)=%.4f" % (res.x[0], res.x[1], res.fun))
