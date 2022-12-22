from bayopt.optim.base_aquisition import BaseAquisition
import torch


class UCB(BaseAquisition):
    def __init__(self, model, data, c, dim):
        super().__init__(model, data, c, dim)
        self.fmin = 0
        self.dim = dim

    def evaluate(self, x):
        ucb = x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)

        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]
        return loss_perf
