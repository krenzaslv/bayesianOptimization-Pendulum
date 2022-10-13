import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.GPModel import ExactGPModel
from src.simulator import simulate
from src.dynamics import U_bo, dynamics_real
from src.optimizer import GPOptimizer, UCBAquisition, GPMin
from src.plotter import Plotter
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy
import math
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star
        self.plotter = Plotter()

    def loss(self, k):
        self.config.kd_bo = k[0]
        self.config.kp_bo = k[1]
        X_bo = simulate(self.config, dynamics_real, U_bo)
        norm = np.linalg.norm(self.X_star - X_bo, ord="fro")
        if norm > 100:
            norm = 100
        return [torch.tensor([k[0], k[1]]), norm]
        # return [
        #     torch.tensor([k[0], k[1]]),
        #     (k[0] - 0.1) * (k[0] - 0.1) * (k[1] - 0.2) * (k[1] - 0.2),
        # ]

    def train(self, plotting=True):
        fig = plt.figure()
        train_x = torch.zeros(self.config.n_opt_samples, 2)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.array([self.config.kd_bo, self.config.kp_bo])

        for i in range(self.config.n_opt_samples):
            # 1. collect Data
            l = self.loss(k)
            [train_x[i, :], train_y[i]] = l

            # 2. Fit GP
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                gpytorch.priors.NormalPrior(0, 1e-5)
            )
            model = ExactGPModel(train_x[: i + 1, :], train_y[: i + 1], likelihood)

            gpOptimizer = GPOptimizer(model, likelihood)
            gpOptimizer.optimize(
                train_x[: i + 1, :],
                train_y[: i + 1],
                self.config.n_opt_iterations,
            )

            # # 3. Find next x with UCB
            ucbAquisition = UCBAquisition(model, likelihood)
            k = ucbAquisition.optimize(self.config.n_opt_iterations)[0]

            #  4.Min
            gpMin = GPMin(model, likelihood)
            x_min = gpMin.optimize(self.config.n_opt_iterations)
            [x_min, y_min] = self.loss(x_min)

            if plotting:
                config = copy.copy(self.config)
                config.kp_bo = x_min[0].detach().numpy()
                config.kd_bo = x_min[1].detach().numpy()
                X_bo = simulate(config, dynamics_real, U_bo)
                self.plotter.plot(
                    model,
                    train_x,
                    train_y,
                    x_min,
                    y_min,
                    i,
                    self.X_star,
                    X_bo,
                    self.config,
                )
