from bayopt.optim.base_aquisition import BaseAquisition
import torch


class UCB(BaseAquisition):
    def __init__(self, model, data, c, dim):
        super().__init__(model, data, c, dim)
        self.fmin = 0
        self.dim = dim

    def evaluate(self, X):
        mean = X.mean.reshape(-1, self.dim)
        var = X.variance.reshape(-1, self.dim)
        ucb = (mean + self.c.scale_beta*torch.sqrt(self.c.beta*var)).reshape(-1, self.dim)
        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]

        return loss_perf
