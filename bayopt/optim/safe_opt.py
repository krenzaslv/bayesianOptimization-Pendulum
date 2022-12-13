from bayopt.optim.base_aquisition import BaseAquisition
from bayopt.models.GPModel import ExactGPModel
import torch


class SafeOpt(BaseAquisition):
    def __init__(self, model, t, c, logger, dim):
        super().__init__(model, t, c, logger, dim)

    def evaluate(self, X):
        self.Q = torch.empty(X.mean.shape[0], 2*self.dim)
        self.S = torch.zeros(X.mean.shape[0], dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()
        self.fmin = 0

        # Update confidence interval
        mean  = X.mean.reshape(-1, self.dim)
        var = X.variance.reshape(-1, self.dim)
        self.Q[:, ::2] = mean - torch.sqrt(self.c.beta*var)
        self.Q[:, 1::2] = mean + torch.sqrt(self.c.beta*var)

        # Compute Safe Set
        self.S[:] = torch.all(self.Q[:, 2::2] > self.fmin, axis=1)

        # Set of maximisers
        l, u = self.Q[:, :2].T

        if not torch.any(self.S):
            print("Couldnt find safe set")
            return torch.max(X.mean[:,0])



        self.M[:] = False
        self.M[self.S] = u[self.S] >= torch.max(l[self.S], dim=0)[0]
        max_var = torch.max(u[self.M] - l[self.M], dim=0)[0]

        # # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]
        s = torch.logical_and(self.S, ~self.M)
        s[s.clone()] = (torch.max((u[s, :] - l[s, :]), axis=1)[0] > max_var)
        s[s.clone()] = torch.any(u[s, :] - l[s, :] > self.fmin, axis=1)

        y = torch.ones_like(self.parameter_set)
        if (torch.any(s)):
            # set of safe expanders
            G_safe = torch.zeros(torch.count_nonzero(s), dtype=bool)
            sort_index = torch.max(u[s, :] - l[s, :], axis=1)[0].argsort()
            for index in reversed(sort_index):
                i = 0
                for model in self.model.models[1:]:
                    i += 1
                    fModel = ExactGPModel(self.c,
                                          torch.cat(
                                              (model.train_x, self.parameter_set[s][index].reshape(1, -1))),
                                          torch.cat((model.train_y, u[s, i][index].flatten())))
                    fModel.eval()

                    pred = fModel(model.train_x)
                    l2 = pred.mean - torch.sqrt(self.c.beta*pred.variance)
                    G_safe[index] = torch.any(l2 >= self.fmin)
                    if not G_safe[index]:
                        break
                if G_safe[index]:
                    break

            self.G[s] = G_safe

        MG = torch.logical_or(self.M, self.G)

        value = torch.max((u - l), axis=1)[0]
        value[~MG] = -1e10
        return value
