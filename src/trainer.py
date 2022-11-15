import numpy as np
from rich.progress import track
import torch
from src.optim.safe_opt import SafeOpt

from src.tools.logger import Logger
from src.tools.normalizer import Normalizer
from src.optim.ucb_aquisition import UCBAquisition


class Trainer:
    def __init__(self, config, X_star):
        self.config = config
        self.X_star = X_star

        self.logger = Logger(config)

    def train(self, loss, model, safePoints):
        train_x = torch.zeros(self.config.n_opt_samples + safePoints.shape[0], self.config.dim)
        train_y = torch.zeros(self.config.n_opt_samples + safePoints.shape[0], loss.dim)
        k = np.zeros(self.config.dim)

        # Record date from known safepoints
        for i in range(safePoints.shape[0]):
            [x_k, y_k, X_bo] = loss.evaluate(safePoints[i])
            [train_x[i, :], train_y[i, :]] = [x_k, y_k]

        yMin = -1e10*np.ones(loss.dim)  # Something small
        for i in track(range(safePoints.shape[0], self.config.n_opt_samples), description="Training..."):
            # 1. collect Data
            xNormalizer = Normalizer(self.config.normalize_data)
            yNormalizer = Normalizer(self.config.normalize_data)
            train_x_n = xNormalizer.fit_transform(train_x[: i ])
            train_y_n = yNormalizer.fit_transform(train_y[: i ])

            if y_k[0] > yMin[0]:
                yMin = y_k
                print("Iteration: {}, yMin: {}".format(i, yMin))

            # 2. Update GP
            model.updateModel(train_x_n, train_y_n)

            # 3. Find next k 
            aquisition= SafeOpt if self.config.aquisition == "SafeOpt" else UCBAquisition

            aquisition = aquisition(model, xNormalizer, yNormalizer, i, self.config, self.logger.writer, loss.dim)
            [k, loss_ucb] = aquisition.getNextPoint()
            k = xNormalizer.itransform(k)[0].detach().numpy()

            [x_k, y_k, X_bo] = loss.evaluate(k)
            [train_x[i, :], train_y[i, :]] = [x_k, y_k]

            self.logger.log(
                model, i-safePoints.shape[0], X_bo, x_k.detach(), y_k.detach(), xNormalizer, yNormalizer, loss_ucb.detach()
            )
