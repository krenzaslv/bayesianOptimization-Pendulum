import typer
import numpy as np
import time
from rich.progress import track
from skopt import gp_minimize

from src.GPModel import ExactGPModel, MLPMean
from src.normalizer import Normalizer
from src.simulator import simulate
from src.logger import Logger
from src.dynamics import U_bo, dynamics_real, U_pert
from src.optimizer import GPOptimizer, UCBAquisition
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy
import math
import matplotlib.pyplot as plt
from src.tools import clamp
from skopt import gp_minimize


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star
        self.logger = Logger()

    def loss(self, k):
        config = copy.copy(self.config)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        stepsize = math.floor(X_bo.shape[0] / self.config.n_evaluate)
        norm = clamp(np.linalg.norm(self.X_star[::stepsize] - X_bo[::stepsize]), 3)
        norm *= norm
        # norm = np.linalg.norm(self.X_star[::stepsize] - X_bo[::stepsize])

        return [torch.tensor([k[0], k[1]]), norm, X_bo]
        # Toy quadratic function
        # return [
        #     torch.tensor([k[0], k[1]]),
        #     (k[0] - 1) * (k[0] - 1) + (k[1] - 3) * (k[1] - 3),
        #     X_bo,
        # ]

    def train(self):
        # res = gp_minimize(
        #     self.loss,
        #     [
        #         (self.config.domain_start_p, self.config.domain_end_p),
        #         (self.config.domain_start_d, self.config.domain_end_d),
        #     ],
        #     acq_func="LCB",
        #     n_calls=100,
        #     n_random_starts=20,
        #     noise=1e-4,
        #     random_state=1234,
        # )
        # print(res)

        train_x = torch.zeros(self.config.n_opt_samples, 2)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.array([self.config.kd_bo, self.config.kp_bo])

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = 1e-4
        likelihood.noise_covar.raw_noise.requires_grad_(False)  # Dont optimize

        yMin = 1e10

        for i in track(range(self.config.n_opt_samples), description="Training..."):
            # 1. collect Data
            [x_k, y_k, X_bo] = self.loss(k)
            [train_x[i, :], train_y[i]] = [x_k, y_k]
            xNormalizer = Normalizer()
            yNormalizer = Normalizer()
            x_train_n = xNormalizer.fit_transform(train_x[: i + 1, :])
            y_train_n = yNormalizer.fit_transform(train_y[: i + 1])
            if y_k < yMin:
                yMin = y_k

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
                model, likelihood, xNormalizer, i, self.config, yMin
            )
            k = ucbAquisition.optimize(self.config.n_opt_iterations_aq)
            k = xNormalizer.itransform(k)

            self.logger.log(model, X_bo, x_k, y_k, xNormalizer, yNormalizer)
