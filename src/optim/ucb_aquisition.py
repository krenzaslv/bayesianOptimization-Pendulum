import numpy as np
from src.optim.base_aquisition import BaseAquisition

class UCBAquisition(BaseAquisition):
    def __init__(self, model, xNormalizer, t, c, logger):
        super().__init__(model, xNormalizer, t, c, logger)

    def loss(self, x):
        return x.mean + np.sqrt(self.c.beta) * self.c.scale_beta * x.variance

