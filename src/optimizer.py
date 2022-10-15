import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand_torch
from matplotlib import cm


class GPOptimizer:
    def __init__(self, model, likelihood, lr=0.1):
        self.model = model
        self.likelihood = likelihood
        self.lr = lr

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()


class UCBAquisition:
    def __init__(self, model, likelihood, xNormalizer, beta=100, lr=0.1):
        self.model = model
        self.likelihood = likelihood
        self.beta = torch.tensor([beta])
        self.xNormalizer = xNormalizer
        self.lr = lr

    def ucb_loss(self, x):
        return x.mean - torch.sqrt(self.beta) * x.variance
        # return -torch.sqrt(self.beta) * x.variance

    def optimize(self, training_steps):
        self.model.eval()
        self.likelihood.eval()

        t = Variable(
            self.xNormalizer.transform(rand_torch(-30, 30, 2, 200)), requires_grad=True
        )
        # t = Variable(torch.tensor([[0.0, 0.0]]), requires_grad=True)
        optimizer = torch.optim.Adam([t], lr=self.lr)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(t)
            loss = torch.sum(self.ucb_loss(output))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                t = self.xNormalizer.itransform(t)
                t[:] = t.clamp(-30, +30)
                t = self.xNormalizer.transform(t)

        loss = self.ucb_loss(self.model(t))

        minIdx = torch.argmin(loss)

        return t[minIdx].detach().numpy()
        # return rand(-15, 15, 2)
