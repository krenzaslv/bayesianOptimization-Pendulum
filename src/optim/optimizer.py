import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.tools.random import rand2d_torch
import math

class GPOptimizer:
    def __init__(self, model, likelihood, t, c, logger, lr=0.1):
        self.model = model
        self.likelihood = likelihood
        self.lr = lr
        self.t = t
        self.logger = logger
        self.c = c

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.c.weight_decay_gp
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        loss = 0
        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if self.t == self.c.n_opt_samples - 1:
                self.logger.add_scalar("Loss/GP_LAST", loss, i)

        return loss

