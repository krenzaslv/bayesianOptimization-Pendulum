import gpytorch
import numpy as np
from rich.progress import track
import torch

from src.models.GPModel import ExactGPModel
from src.tools.logger import Logger
from src.tools.normalizer import Normalizer, Normalizer2d
from src.optim.optimizer import GPOptimizer
from src.optim.ucb_aquisition import UCBAquisition


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star

        self.logger = Logger(config)


    def createModel(self, train_x_n, train_y_n, likelihood):
        mean_module = gpytorch.means.ZeroMean()
        # mean_module = MLPMean()

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
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

    def train(self, loss):
        train_x = torch.zeros(self.config.n_opt_samples, self.config.dim)
        train_y = torch.zeros(self.config.n_opt_samples)
        k = np.zeros(self.config.dim)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = torch.tensor([self.config.init_variance])

        yMin = -1e10 # Something small
        for i in track(range(self.config.n_opt_samples), description="Training..."):
            # 1. collect Data
            [x_k, y_k, X_bo] = loss.evaluate(k)
            [train_x[i, :], train_y[i]] = [x_k, y_k]
            xNormalizer = Normalizer2d()
            yNormalizer = Normalizer()
            train_x_n = xNormalizer.fit_transform(train_x[: i + 1])
            train_y_n = yNormalizer.fit_transform(train_y[: i + 1])

            if y_k > yMin:
                yMin = y_k
                print("Iteration: {}, yMin: {}".format(i, yMin))

            # 2. Update GP
            model = self.createModel(train_x_n, train_y_n, likelihood)

            # 3. Find next k with UCB
            ucbAquisition = UCBAquisition(
                model, likelihood, xNormalizer, i, self.config, yMin, self.logger.writer
            )
            [k, loss_ucb] = ucbAquisition.optimize(
                self.config.n_opt_iterations_aq)
            k = xNormalizer.itransform(k)[0].detach().numpy()

            self.logger.log(
                model, i, X_bo, x_k, y_k, xNormalizer, yNormalizer, loss_ucb
            )
