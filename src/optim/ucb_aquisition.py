from src.optim.base_aquisition import BaseAquisition
import torch


class UCBAquisition(BaseAquisition):
    def __init__(self, model, xNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, t, c, logger, dim)

    def loss(self, x):
        return x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
