from bayopt.optim.base_aquisition import BaseAquisition
import torch


class SafeUCB(BaseAquisition):
    def __init__(self, model, data, c, dim):
        super().__init__(model, data, c, dim)
        self.fmin = 0
        self.dim = dim

    def evaluate(self, X):
        # UPC-Safe
        mean = X.mean.reshape(-1, self.dim)
        var = X.variance.reshape(-1, self.dim)
        l = mean - self.c.scale_beta*torch.sqrt(self.c.beta*var)
        ucb = (mean + self.c.scale_beta*torch.sqrt(self.c.beta*var)).reshape(-1, self.dim)

        S = torch.all(l[:,1:] > self.fmin, axis=1)

        ucb[~S] = -1e10
        # self.init_points = self.init_points[S]

        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]
        return loss_perf
