from src.optim.base_aquisition import BaseAquisition
import torch


class UCBAquisition(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, yNormalizer, t, c, logger, dim)
        self.fmin = -1

    def loss(self, x):
        upc = x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
        if self.dim == 1:
            return upc
        else:
            #UPC-Safe
            l = x.mean - self.c.beta*torch.sqrt(x.variance)
            S = torch.all(self.yNormalizer.itransform(l)[:,1:] > self.fmin, axis=1)
            upc[~S] = -1e10
            return upc 
