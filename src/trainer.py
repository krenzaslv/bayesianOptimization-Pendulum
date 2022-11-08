import gpytorch
import numpy as np
from rich.progress import track
import torch

from src.tools.logger import Logger
from src.tools.normalizer import Normalizer, Normalizer2d
from src.optim.optimizer import GPOptimizer
from src.optim.ucb_aquisition import UCBAquisition


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star

        self.logger = Logger(config)


    def train(self, loss, model):
        train_x = torch.zeros(self.config.n_opt_samples, self.config.dim)
        train_y = torch.zeros(self.config.n_opt_samples,1)
        k = np.zeros(self.config.dim)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = torch.tensor([self.config.init_variance])
        

        yMin = -1e10  # Something small
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
            model.updateModel(train_x_n, train_y_n[:,0])


            # 3. Find next k with UCB
            if self.config.aquisition == "UCB":
                aquisition = UCBAquisition(model, xNormalizer, i, self.config, self.logger.writer)

            [k, loss_ucb] = aquisition.optimize(
                self.config.n_opt_iterations_aq)
            k = xNormalizer.itransform(k)[0].detach().numpy()

            self.logger.log(
                model, i, X_bo, x_k, y_k, xNormalizer, yNormalizer, loss_ucb
            )
