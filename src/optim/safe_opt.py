from src.optim.base_aquisition import BaseAquisition
import torch

class SafeOpt(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, t, c, logger, dim)
        inputs = self.getInitPoints()
        self.Q = torch.empty(inputs.shape[0], 2*dim)
        self.S = torch.zeros(inputs.shape[0], dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()
        self.fmin = 0
        self.yNormalizer = yNormalizer


    def loss(self, x):
        # Update confidence interval
        self.Q[:,::2] = x.mean - self.c.beta*torch.sqrt(x.variance)
        self.Q[:,1::2] = x.mean + self.c.beta*torch.sqrt(x.variance)

        #TODO first initial safe set... Safe set with normalization?
        #Compute Safe Set
        self.S[:] = torch.all(self.yNormalizer.itransform(self.Q[:,::2])[:,1:] > self.fmin, axis=1)

        # # Set if maximisers
        # self.M[:] = False
        # l, u = self.Q[:, :2].T

        # self.M[:] = False
        # self.M[self.S] = u[self.S] >= torch.max(l[self.S], dim=0)[0]
        # max_var = torch.max(u[self.M] - l[self.M], dim=0)[0]

        # # Optimistic set of possible expanders
        # l = self.Q[:, ::2]
        # u = self.Q[:, 1::2]

        # s = torch.logical_and(self.S, ~self.M)
        # print(self.Q[s].shape)

        #     # Remove points with a variance that is too small
        # print(torch.max((u[s, :] - l[s, :]) , axis=1)[0])
        # s[s] = torch.max((u[s, :] - l[s, :]) , axis=1)[0] > max_var
        # s[s] = torch.any(u[s, :] - l[s, :] > self.threshold * beta, axis=1)
        # # for i in range(len(G_safe)):


        # self.G[:] = False
        tmp = x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
        tmp[~self.S] = -1e10
        return x.mean + self.c.scale_beta*torch.sqrt(self.c.beta*x.variance)
        return tmp

 
