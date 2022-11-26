import numpy as np
from rich.progress import track
import torch

from src.tools.logger import Logger
from src.optim.ucb import UCB 
from src.optim.safe_opt import SafeOpt
from src.optim.safe_ucb import SafeUCB 
from rich import print


class Trainer:
    def __init__(self, config):
        self.config = config

        self.logger = Logger(config)

    def train(self, loss, model, safePoints):
        train_x = torch.zeros(self.config.n_opt_samples + safePoints.shape[0], self.config.dim)
        train_y = torch.zeros(self.config.n_opt_samples + safePoints.shape[0], loss.dim)
        k = np.zeros(self.config.dim)

        # Record data from known safepoint
        # for i in range(safePoints.shape[0]):
        #     [x_k, y_k, X_bo] = loss.evaluate(safePoints[i])
        #     [train_x[i, :], train_y[i, :]] = [x_k, y_k]

        yMin = -1e10*np.ones(loss.dim)  # Something small

        for i in track(range(0,self.config.n_opt_samples-1), description="Training..."):
            
            # 2. Find next k
            if self.config.aquisition == "SafeOpt":
                aquisition = SafeOpt 
            elif self.config.aquisition == "SafeUCB":
                aquisition = SafeUCB 
            else:
                aquisition = UCB

            aquisition = aquisition(model, i,
                                    self.config, self.logger.writer, loss.dim)

            if i < safePoints.shape[0]:
                [k, loss_ucb] = [safePoints[i], torch.tensor([0])]
            else:
                [k, loss_ucb] = aquisition.getNextPoint()

            # 3. Evaluate new k
            [x_k, y_k, X_bo] = loss.evaluate(k)
            [train_x[i, :], train_y[i, :]] = [x_k, y_k]


            # 1. Update GP
            train_x_n = train_x[: i + 1]
            train_y_n = train_y[: i + 1]
            model.updateModel(train_x_n, train_y_n)


            if torch.any(y_k[1:] < 0):
                print("[yellow][Warning][/yellow] Constraint violated at iteration {} with {}".format(i, y_k))

            if y_k[0] > yMin[0]:
                yMin = y_k
                print("[green][Info][/green] New minimum at Iteration: {}, yMin: {}".format(i, yMin))

            self.logger.log(
                model, i - 1, safePoints.shape[0], X_bo, train_x[i, :],
                train_y[i, :].detach(
                ), loss_ucb.detach()
            )
