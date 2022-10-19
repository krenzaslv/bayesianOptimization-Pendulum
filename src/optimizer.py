import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand2d_torch
from matplotlib import cm
import math
from scipy.stats import norm


class GPOptimizer:
    def __init__(self, model, likelihood, lr=0.1):
        self.model = model
        self.likelihood = likelihood
        self.lr = lr

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()


class UCBAquisition:
    def __init__(self, model, likelihood, xNormalizer, t, c, yMin):
        self.model = model
        self.likelihood = likelihood
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c
        self.yMin = yMin

    def ucb_loss(self, x):
        D = (self.c.domain_end_p - self.c.domain_start_p) * (
            self.c.domain_end_d - self.c.domain_start_d
        )
        beta_t = 2 * np.log(
            D * self.t * self.t * math.pi * math.pi / (6 * self.c.gamma)
        )
        # beta_t = self.c.beta
        return x.mean - np.sqrt(beta_t) * self.c.scale_beta * x.variance
        # return -torch.sqrt(self.beta) * x.variance
        # return -torch.sqrt(self.beta) * x.variance
        #

    # TODO doesnt work...
    def ei_loss(self, x):
        m = x.mean.detach().numpy()
        sigma = x.mean.detach().numpy()
        sigma = x.variance
        u = (m - self.yMin) / sigma
        u = u.detach().numpy()
        ei = sigma * (u * norm().cdf(u) + norm().pdf(u))
        ei[sigma <= 0] = 0
        return ei

    def optimize(self, training_steps):
        self.model.eval()
        self.likelihood.eval()

        randInit = rand2d_torch(
            self.c.domain_start_p,
            self.c.domain_end_p,
            self.c.domain_start_d,
            self.c.domain_end_d,
            self.c.n_sample_points,
        )
        t = Variable(
            self.xNormalizer.transform(randInit),
            requires_grad=True,
        )

        optimizer = torch.optim.SGD([t], self.c.lr_aq)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        aquisition = self.ei_loss if self.c.aquisition == "ei" else self.ucb_loss

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(t)
            loss = torch.sum(aquisition(output))
            loss.backward()
            optimizer.step()

            # Filter out results that are not in the domain
            with torch.no_grad():
                t = self.xNormalizer.itransform(t)
                t[t[:, 0] < self.c.domain_start_p] = self.c.domain_start_p
                t[t[:, 0] > self.c.domain_end_p] = self.c.domain_end_p
                t[t[:, 1] < self.c.domain_start_d] = self.c.domain_start_d
                t[t[:, 1] > self.c.domain_end_d] = self.c.domain_end_d
                t = self.xNormalizer.transform(t)

        loss = aquisition(self.model(t))

        minIdx = torch.argmin(loss)
        return t[minIdx].detach().numpy()
