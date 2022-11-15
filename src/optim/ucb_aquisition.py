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
        if self.dim == 1 or not self.c.use_constraints:
            return ucb
        else:
            #UPC-Safe
            for i in range(1, self.dim):
                l = x[i].mean - self.c.scale_beta*torch.sqrt(self.c.beta*x[i].variance)
                S = l.le(self.fmin)
                print(ucb[S].shape)
                
                ucb[S,0] = -1e10
            return ucb
