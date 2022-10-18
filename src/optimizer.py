import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand_torch
from matplotlib import cm
import math


class GPOptimizer:
    def __init__(self, model, likelihood, lr=0.1):
        self.model = model
        self.likelihood = likelihood
        self.lr = lr

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            # print("Iter %d/%d - Loss: %.3f" % (i + 1, i, loss.item()))

        # print("GP Loss: {}".format(loss))


class UCBAquisition:
    def __init__(self, model, likelihood, xNormalizer, t, c):
        self.model = model
        self.likelihood = likelihood
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c

    def ucb_loss(self, x):
        D = self.c.domain_end - self.c.domain_start
        D = D * D
        beta_t = 2 * np.log(
            D * self.t * self.t * math.pi * math.pi / (6 * self.c.gamma)
        )
        return x.mean - np.sqrt(beta_t) * x.variance
        # return -torch.sqrt(self.beta) * x.variance
        # return -torch.sqrt(self.beta) * x.variance

    def optimize(self, training_steps):
        self.model.eval()
        self.likelihood.eval()

        t = Variable(
            self.xNormalizer.transform(
                rand_torch(self.c.domain_start, self.c.domain_end, 2, 1000)
            ),
            requires_grad=True,
        )
        # optimizer = torch.optim.Adam([t], lr=self.c.lr_aq)
        optimizer = torch.optim.SGD([t], self.c.lr_aq)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(t)
            loss = torch.sum(self.ucb_loss(output))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                t = self.xNormalizer.itransform(t)
                t[:] = t.clamp(self.c.domain_start, self.c.domain_end)
                t = self.xNormalizer.transform(t)

        loss = self.ucb_loss(self.model(t))

        minIdx = torch.argmin(loss)
        # print("UCB Loss: {}".format(loss[minIdx]))
        return t[minIdx].detach().numpy()
        # return rand(-15, 15, 2)
