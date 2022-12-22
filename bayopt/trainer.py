import numpy as np
from rich.progress import track
import torch

from bayopt.tools.logger import Logger
from bayopt.optim.ucb import UCB
from bayopt.optim.safe_opt import SafeOpt
from bayopt.tools.math import scale
from bayopt.optim.safe_ucb import SafeUCB
from bayopt.tools.data import Data
from rich import print


class Trainer:
    def __init__(self, config):
        self.config = config

        self.logger = Logger(config)

        self.data = Data(config)

    def train(self, loss, model, safePoints):
        # train_x = torch.zeros(self.config.n_opt_samples +
        #                       safePoints.shape[0], self.config.dim_params)
        # train_y = torch.zeros(self.config.n_opt_samples + safePoints.shape[0], loss.dim)
        k = np.zeros(self.config.dim_params)

        yMin = -1e10*np.ones(loss.dim)  # Something small

        state_dict = None

        for i in track(range(0, self.config.n_opt_samples-1), description="Training..."):

            # 2. Find next k
            if self.config.aquisition == "SafeOpt":
                aquisition = SafeOpt
            elif self.config.aquisition == "SafeUCB":
                aquisition = SafeUCB
            else:
                aquisition = UCB

            if i < safePoints.shape[0]:
                [k, loss_ucb] = [safePoints[i], torch.tensor([0])]
            else:
                gp = model(self.config, self.data, state_dict)
                aquisition = aquisition(gp, self.data, self.config, loss.dim)
                [k, loss_ucb] = aquisition.getNextPoint()

            # 3. Evaluate new k
            [x_k, y_k, X_bo] = loss.evaluate(k)
            self.data.append_data(x_k.reshape(1,-1), y_k.reshape(1, -1))

            if torch.any(y_k[1:] < 0):
                print(
                    "[yellow][Warning][/yellow] Constraint violated at iteration {} with {} at {}".format(i, y_k, k))

            if y_k[0] > yMin[0]:
                yMin = y_k
                print("[green][Info][/green] New minimum at Iteration: {},yMin:{} at {}".format(i, yMin, k))

            # self.logger.log(
            #     model, i - 1, safePoints.shape[0], X_bo, train_x[i, :],
            #     train_y[i, :].detach(
            #     ), loss_ucb.detach()
            # )
            loss.reset()
