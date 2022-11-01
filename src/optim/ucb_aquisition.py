import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.tools.random import rand2d_torch
import math

class UCBAquisition:
    def __init__(self, model, likelihood, xNormalizer, t, c, yMin, logger):
        self.model = model
        self.likelihood = likelihood
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c
        self.yMin = torch.tensor([yMin])
        self.logger = logger
        self.minIdxBuffer = []

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        aquisition = self.ucb_loss

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(t)
            loss = aquisition(output).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if self.t == self.c.n_opt_samples - 1:
                loss = aquisition(self.model(t))
                self.logger.add_scalar("Loss/AQ_LAST", loss.min(), i)

        loss = aquisition(self.model(t))

        minIdx = torch.argmin(loss)
        
        #Take next best value if already sampled
        k = 1
        while t[minIdx] in self.model.train_inputs[0]:
            k += 1
            _ ,minIdx = torch.kthvalue(loss, k)

        return [t[minIdx], loss[minIdx]]
