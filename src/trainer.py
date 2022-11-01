import copy
import math

import gpytorch
import numpy as np
from rich.progress import track
import torch

from src.GPModel import ExactGPModel
from src.dynamics import U_bo, dynamics_real
from src.logger import Logger
from src.normalizer import Normalizer, Normalizer2d
from src.optimizer import GPOptimizer, UCBAquisition
from src.simulator import simulate


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star

        self.logger = Logger(config)

    def loss(self, k):
        config = copy.copy(self.config)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        stepsize = math.floor(X_bo.shape[0] / self.config.n_evaluate)
        norm = np.linalg.norm(self.X_star[::stepsize] - X_bo[::stepsize])

        return [torch.tensor([k[0], k[1]]), norm, X_bo]

    def createModel(self, train_x_n, train_y_n, likelihood):
        mean_module = gpytorch.means.ZeroMean()
        # mean_module = MLPMean()

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                lengthscale_prior=gpytorch.priors.MultivariateNormalPrior(
                    loc=torch.tensor(
                        [self.config.init_lenghtscale,
                            self.config.init_lenghtscale]
                    ),
                    covariance_matrix=torch.tensor(
                        [
                            [self.config.init_variance, 0.0],
                            [0.0, self.config.init_variance],
                        ]
                    ),
                ),
                ard_num_dims=2,
            )
        )
        covar_module.base_kernel.lengthscale = self.config.init_lenghtscale

        model = ExactGPModel(
            train_x_n,
            train_y_n,
            likelihood,
            mean_module,
            covar_module,
        )
        return model

    def train(self):
        train_x = torch.zeros(self.config.n_opt_samples, 2)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.array([self.config.kd_bo, self.config.kp_bo])

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = torch.tensor([1e-4])
        likelihood.noise_covar.raw_noise.requires_grad_(False)  # Dont optimize

        yMin = 1e10
        for i in track(range(self.config.n_opt_samples), description="Training..."):
            # 1. collect Data
            [x_k, y_k, X_bo] = self.loss(k)
            [train_x[i, :], train_y[i]] = [x_k, y_k]
            xNormalizer = Normalizer2d()
            yNormalizer = Normalizer()
            train_x_n = xNormalizer.fit_transform(train_x[: i + 1])
            train_y_n = yNormalizer.fit_transform(train_y[: i + 1])

            if y_k < yMin:
                yMin = y_k
                print("Iteration: {}, yMin: {}".format(i, yMin))

            # 2. Fit GP
            model = self.createModel(train_x_n, train_y_n, likelihood)
            gpOptimizer = GPOptimizer(
                model, likelihood, i, self.config, self.logger.writer, self.config.lr_gp
            )
            loss_gp = gpOptimizer.optimize(
                train_x_n,
                train_y_n,
                self.config.n_opt_iterations_gp,
            )

            # 3. Find next k with UCB
            ucbAquisition = UCBAquisition(
                model, likelihood, xNormalizer, i, self.config, yMin, self.logger.writer
            )
            [k, loss_ucb] = ucbAquisition.optimize(
                self.config.n_opt_iterations_aq)
            k = xNormalizer.itransform(k)[0].detach().numpy()

            self.logger.log(
                model, i, X_bo, x_k, y_k, xNormalizer, yNormalizer, loss_gp, loss_ucb
            )
