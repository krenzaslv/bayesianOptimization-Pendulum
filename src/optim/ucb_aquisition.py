from src.optim.base_aquisition import BaseAquisition
import torch


class UCBAquisition(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, yNormalizer, t, c, logger, dim)
        self.fmin = 0

    def loss(self, x):
        ucb = torch.zeros(x[0].mean.shape[0], self.dim)
        for i in range(self.dim):
            ucb[:,i] = x[i].mean + self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)
        if self.dim == 1:
            return ucb
        else:
            #UPC-Safe
            # l = x.mean - self.c.beta*torch.sqrt(x.variance)
            # S = torch.all(self.yNormalizer.itransform(l)[:,1:] > self.fmin, axis=1)
            # print(torch.max(l[:,1]))
            # print(ucb[~S].shape)
            # ucb[~S] = -1e10
            return ucb
