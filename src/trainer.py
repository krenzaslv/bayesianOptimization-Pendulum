import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.GPModel import ExactGPModel, MLPMean
from src.normalizer import Normalizer
from src.simulator import simulate
from src.dynamics import U_bo, dynamics_real, U_pert
from src.optimizer import GPOptimizer, UCBAquisition
from src.plotter import Plotter
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy
import math
import matplotlib.pyplot as plt
from src.tools import clamp


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
        stepsize = math.floor(X_bo.shape[0] / self.config.n_evaluate)
        norm = np.linalg.norm(self.X_star[::stepsize] - X_bo[::stepsize])
        # norm = np.linalg.norm(self.X_star[::stepsize] - X_bo[::stepsize])

        return [torch.tensor([k[0], k[1]]), norm, X_bo]
        # Toy quadratic function
        # return [
        #     torch.tensor([k[0], k[1]]),
        #     (k[0] - 1) * (k[0] - 1) + (k[1] - 3) * (k[1] - 3),
        #     X_bo,
        # ]

    def train(self, plotting=True):
        fig = plt.figure()
        train_x = torch.zeros(self.config.n_opt_samples, 2)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.array([self.config.kd_bo, self.config.kp_bo])

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = 1e-4
        likelihood.noise_covar.raw_noise.requires_grad_(False)  # Dont optimize

        for i in range(self.config.n_opt_samples):
            # 1. collect Data
            [x_k, y_k, X_bo] = self.loss(k)
            [train_x[i, :], train_y[i]] = [x_k, y_k]
            xNormalizer = Normalizer()
            yNormalizer = Normalizer()
            x_train_n = xNormalizer.fit_transform(train_x[: i + 1, :])
            y_train_n = yNormalizer.fit_transform(train_y[: i + 1])

            # 2. Fit GP
            # mean_module = gpytorch.means.ConstantMean()
            mean_module = gpytorch.means.ZeroMean()
            # mean_module = gpytorch.means.LinearMean(2)
            # mean_module = MLPMean()
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
            model = ExactGPModel(
                x_train_n, y_train_n, likelihood, mean_module, covar_module
            )

            gpOptimizer = GPOptimizer(model, likelihood, self.config.lr_gp)
            gpOptimizer.optimize(
                x_train_n,
                y_train_n,
                self.config.n_opt_iterations_gp,
            )

            # 3. Find next k with UCB
            ucbAquisition = UCBAquisition(
                model, likelihood, xNormalizer, i, self.config
            )
            k = ucbAquisition.optimize(self.config.n_opt_iterations_aq)
            k = xNormalizer.itransform(k)

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
