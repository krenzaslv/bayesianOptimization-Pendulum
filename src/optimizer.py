import torch
import gpytorch
from torch.autograd import Variable
import numpy as np
from src.random import rand, rand_torch


class GPOptimizer:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood

    def optimize(self, train_x, train_y, training_steps):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_steps):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        # print(
        #     "loss: %.3f   lengthscale: %.3f   noise: %.3f"
        #     % (
        #         loss.item(),
        #         self.model.covar_module.base_kernel.lengthscale.item(),
        #         self.model.likelihood.noise.item(),
        #     )
        # )


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
        return rand(-20, 20, 2)


class GPMin:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood

    def optimize(self, training_steps, n_restarts=5):
        self.model.eval()
        self.likelihood.eval()

        # TODO mutliple starting points
        x = Variable(rand_torch(-20, 20, 2, 1), requires_grad=True)
        # x = Variable(torch.tensor([[0, 0]]), requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.01)

        # "Loss" for GPs - the marginal log likelihood

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model(x)
            loss = torch.sum(self.model(x).mean - self.model(x).variance)
            loss.backward()
            optimizer.step()

        return x[0].detach().numpy()
