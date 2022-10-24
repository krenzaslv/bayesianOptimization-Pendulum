import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand2d_torch
from matplotlib import cm
import math
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal


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
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        loss = 0
        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if self.t == self.c.n_opt_samples - 1:
                self.logger.add_scalar("Loss/GP_LAST", loss, i)

        return loss


class UCBAquisition:
    def __init__(self, model, likelihood, xNormalizer, t, c, yMin, logger):
        self.model = model
        self.likelihood = likelihood
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c
        self.yMin = torch.tensor([yMin])
        self.logger = logger

    def ucb_loss(self, x):
        D = (self.c.domain_end_p - self.c.domain_start_p) * (
            self.c.domain_end_d - self.c.domain_start_d
        )
        beta_t = (
            self.c.beta
            if self.c.beta_fixed
            else 2 * np.log(D * (self.t) * (self.t) * math.pi / (6 * self.c.gamma))
        )
        return x.mean - np.sqrt(beta_t) * self.c.scale_beta * x.variance

    def ei_loss(self, x):
        m = x.mean
        sigma = x.variance.clamp_min(1e-9).sqrt()
        u = -(m - self.yMin.expand_as(m) + 10) / sigma
        dist = torch.distributions.normal.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ei = sigma * (dist.cdf(u) + u * torch.exp(dist.log_prob(u)))
        return -ei

    def getInitPoints(self):
        if self.c.ucb_use_set:
            xs = torch.linspace(
                self.c.domain_start_p, self.c.domain_end_p, self.c.ucb_set_n
            )
            ys = torch.linspace(
                self.c.domain_start_d, self.c.domain_end_d, self.c.ucb_set_n
            )
            x, y = torch.meshgrid(xs, ys, indexing="xy")
            init = torch.stack((x, y), dim=2)
            init = torch.reshape(init, (-1, 2))
            init = torch.cat((init, torch.tensor([[self.c.kp, self.c.kd]])), 0)

        else:
            init = rand2d_torch(
                self.c.domain_start_p,
                self.c.domain_end_p,
                self.c.domain_start_d,
                self.c.domain_end_d,
                self.c.n_sample_points,
            )

        t = Variable(
            self.xNormalizer.transform(init),
            requires_grad=True,
        )
        return t

    def optimize(self, training_steps):
        self.model.eval()
        self.likelihood.eval()

        t = self.getInitPoints()

        optimizer = torch.optim.Adam(
            [t], self.c.lr_aq, weight_decay=self.c.weight_decay_aq
        )
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        aquisition = self.ei_loss if self.c.aquisition == "ei" else self.ucb_loss

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(t)
            loss = aquisition(output).sum()
            loss.backward()
            optimizer.step()
            # t.clamp(-5, 5)
            if self.t == self.c.n_opt_samples - 1:
                loss = aquisition(self.model(t))
                self.logger.add_scalar("Loss/AQ_LAST", loss.min(), i)

        loss = aquisition(self.model(t))

        minIdx = torch.argmin(loss)

        return [t[minIdx], loss[minIdx]]
