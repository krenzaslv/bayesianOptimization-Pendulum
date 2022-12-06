from bayopt.optim.base_aquisition import BaseAquisition
import torch


class SafeUCB(BaseAquisition):
    def __init__(self, model, t, c, logger, dim):
        super().__init__(model, t, c, logger, dim)
        self.fmin = 0
        self.dim = dim

    def evaluate(self, x):
        ucb = torch.zeros(x[0].mean.shape[0], self.dim)
        for i in range(self.dim):
            ucb[:, i] = x[i].mean + self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)

        # UPC-Safe
        for i in range(1, self.dim):
            l = x[i].mean - self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)
            S = l.le(self.fmin)

            ucb[S] = -1e10
            # self.init_points = self.init_points[S]

        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]
        return loss_perf
