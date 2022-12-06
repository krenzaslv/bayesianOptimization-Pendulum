from bayopt.optim.base_aquisition import BaseAquisition
import torch


class SafeUCB(BaseAquisition):
    def __init__(self, model, t, c, logger, dim):
        super().__init__(model, t, c, logger, dim)
        self.fmin = 0
        self.dim = dim

    def evaluate(self, x):
        ucb = (x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)).reshape(-1, self.dim)

        # UPC-Safe
        l = x.mean - self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
        S = torch.any(l.le(self.fmin), axis=1)

        ucb[S] = -1e10
        # self.init_points = self.init_points[S]

        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]
        return loss_perf
