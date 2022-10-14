import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.GPModel import ExactGPModel
from src.normalizer import Normalizer
from src.simulator import simulate
from src.dynamics import U_bo, dynamics_real
from src.optimizer import GPOptimizer, UCBAquisition
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
        config = copy.copy(self.config)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        norm = np.sqrt(((self.X_star - X_bo) ** 2).sum().sum())
        if norm > 20:
            norm = 20
        return [torch.tensor([k[0], k[1]]), norm, X_bo]
        # Toy quadratic function
        # return [
        #     torch.tensor([k[0], k[1]]),
        #     (k[0] - 0.1) * (k[0] - 0.1) * (k[1] - 0.2) * (k[1] - 0.2),
        #     X_bo,
        # ]

    def train(self, plotting=True):
        fig = plt.figure()
        train_x = torch.zeros(self.config.n_opt_samples, 2)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.array([self.config.kd_bo, self.config.kp_bo])

        for i in range(self.config.n_opt_samples):
            # 1. collect Data
            [x_k, y_k, X_bo] = self.loss(k)
            [train_x[i, :], train_y[i]] = [x_k, y_k]

            # 2. Fit GP
            xNormalizer = Normalizer()
            yNormalizer = Normalizer()
            x_train_n = xNormalizer.fit_transform(train_x[: i + 1, :])
            y_train_n = yNormalizer.fit_transform(train_y[: i + 1])
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
            )
            likelihood.noise = 1e-4
            # likelihood.noise_covar.raw_noise.requires_grad_(False)
            model = ExactGPModel(x_train_n, y_train_n, likelihood)

            gpOptimizer = GPOptimizer(model, likelihood, self.config.lr)
            gpOptimizer.optimize(
                x_train_n,
                y_train_n,
                self.config.n_opt_iterations,
            )

            # 3. Find next x with UCB
            ucbAquisition = UCBAquisition(
                model, likelihood, xNormalizer, self.config.beta, self.config.lr
            )
            k = ucbAquisition.optimize(self.config.n_opt_iterations)
            k = xNormalizer.itransform(k)

            # 4. Find GP Min
            # gpMin = GPMin(model, likelihood, xNormalizer, 0.1)  # self.config.lr)
            # x_min = xNormalizer.itransform(
            #     gpMin.optimize(100)
            # )  # self.config.n_opt_iterations))
            # print(x_min)

            if plotting:
                self.plotter.plot(
                    model,
                    train_x,
                    train_y,
                    i,
                    self.X_star,
                    X_bo,
                    # x_min,
                    self.config,
                    xNormalizer,
                    yNormalizer,
                    self.config,
                )
