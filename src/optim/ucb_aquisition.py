from src.optim.base_aquisition import BaseAquisition
import torch


class UCBAquisition(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, yNormalizer, t, c, logger, dim)
        self.fmin = 0
        self.yNormalizer = yNormalizer
        self.dim = dim

    def loss(self, x):
        ucb = torch.zeros(x[0].mean.shape[0], self.dim)
        for i in range(self.dim):
            ucb[:,i] = x[i].mean + self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)

        if self.dim != 1 and self.c.use_constraints:
            #UPC-Safe
            for i in range(1, self.dim):
                l = x[i].mean - self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)
                S = l.ge(self.fmin)

                ucb = ucb[S]
                self.init_points = self.init_points[S]

        loss_perf = ucb if ucb.dim() == 1 else ucb[:, 0]
        maxIdx = torch.argmax(loss_perf)
        return [self.init_points[maxIdx], ucb[maxIdx]] 
