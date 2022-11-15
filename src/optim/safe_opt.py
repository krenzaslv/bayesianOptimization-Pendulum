from src.models.GPModel import ExactGPModel, ExactMultiTaskGP
from src.optim.base_aquisition import BaseAquisition
import torch


class SafeOpt(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, yNormalizer, t, c, logger, dim)

        self.Q = torch.empty(self.parameter_set.shape[0], 2*dim)
        self.S = torch.zeros(self.parameter_set.shape[0], dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()
        self.fmin = 0

    def loss(self, x):
        # Update confidence interval
        for i in range(self.dim):
            self.Q[:, 2*i] = x[i].mean - torch.sqrt(self.c.beta*x[i].variance)
            self.Q[:, 2*i+1] = x[i].mean + torch.sqrt(self.c.beta*x[i].variance)

        # Compute Safe Set
        self.S[:] = self.S[:] = torch.all(self.Q[:, 2::2] > self.fmin, axis=1)

        if not torch.any(self.S):
            print("Couldnt find safe set")

        # print(self.Q[self.S].shape)

        # Set of maximisers
        l, u = self.Q[:, :2].T

        self.M[:] = False
        self.M[self.S] = u[self.S] >= torch.max(l[self.S], dim=0)[0]
        max_var = torch.max(u[self.M] - l[self.M], dim=0)[0]

        # # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]
        s = self.S #torch.logical_and(self.S, ~self.M)
        # s[s] = (torch.max((u[s, :] - l[s, :]) , axis=1)[0] >max_var)
        # s[s] = torch.any(u[s, :] - l[s, :] > self.fmin, axis=1)

        if (torch.any(s)):
            # set of safe expanders
            G_safe = torch.zeros(torch.count_nonzero(s), dtype=bool)
            sort_index = torch.max(u[s, :] - l[s, :], axis=1)[0].argsort()
            for index in reversed(sort_index):
                for model in self.model.models[1:]:
                    #TODO somehow pickle can't handle fairy_model
                    fModel = ExactGPModel(self.c,
                                          torch.cat((model.train_x, self.parameter_set[s][index].reshape(1, -1))),
                                          torch.cat((model.train_y, u[s, i][index].flatten())))
                    fModel = ExactGPModel(self.c, model.train_x, model.train_y)
                    #TODO somehow fmodel doesnt work
                    pred = fModel(self.parameter_set[~self.S])
                    l2 = pred.mean #- torch.sqrt(self.c.beta*pred.variance)
                    # print(torch.max(l2))
                    if torch.any(l2 >= self.fmin):
                        print(torch.max(l2))
                    G_safe[index] = torch.any(l2 >= self.fmin)
                    if not G_safe[index]:
                        break
                if G_safe[index]:
                    break

            self.G[s] = G_safe

        MG = torch.logical_or(self.M, self.G)

        value = torch.max((u[MG] - l[MG]), axis=1)[0]
        return [self.parameter_set[MG, :][torch.argmax(value), :], torch.argmax(value)]
