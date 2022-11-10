from src.optim.base_aquisition import BaseAquisition
import torch

class SafeOpt(BaseAquisition):
    def __init__(self, model, xNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, t, c, logger, dim)
        inputs = self.getInitPoints()
        self.Q_l = torch.empty(inputs.shape[0], dim)
        self.Q_u = torch.empty(inputs.shape[0], dim)
        self.S = torch.zeros(inputs.shape[0], dim, dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()
        self.fmin = 0


    def loss(self, x):
        # Update confidence interval
        self.Q_l = x.mean - self.c.beta*torch.sqrt(x.variance)
        self.Q_u = x.mean + self.c.beta*torch.sqrt(x.variance)

        #Compute Safe Set
        self.S = torch.all(self.Q_l > self.fmin, axis=1)

        # Set if maximisers
        self.M[:] = False
        print(self.Q_u.shape)
        print(self.S.shape)
        print(self.Q_u[self.S, 0])
        self.M[self.S] = self.Q_u[self.S, 0]# >= torch.max(self.Q_l[self.S, 0])
        # max_var = torch.max(self.Q_u[self.M] - self.Q_l[self.M]) / self.scaling[0]

        print(self.Q_l)
        print(self.S)
        return x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
    
