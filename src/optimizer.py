import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand_torch
from matplotlib import cm


class GPOptimizer:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()


class UCBAquisition:
    def __init__(self, model, likelihood, beta=1.0):
        self.model = model
        self.likelihood = likelihood
        self.beta = torch.tensor([beta])

    def ucb_loss(self, x):
        return x.mean + torch.sqrt(self.beta) * x.variance

    def optimize(self, training_steps):
        # self.model.eval()
        # self.likelihood.eval()

        # # TODO mutliple starting points
        # x = Variable(torch.tensor([[0.0, 0.0]]), requires_grad=True)
        # optimizer = torch.optim.Adam([x], lr=0.001)

        # # "Loss" for GPs - the marginal log likelihood
        # for i in range(training_steps):
        #     optimizer.zero_grad()
        #     output = self.model(x)
        #     loss = -self.ucb_loss(output)
        #     loss.backward()
        #     optimizer.step()

        # return x.detach().numpy()
        return rand(-15, 15, 2)
