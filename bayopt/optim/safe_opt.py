from bayopt.optim.base_aquisition import BaseAquisition
from bayopt.models.GPModel import ExactGPModel, ExactMultiTaskGP
import torch


class SafeOpt(BaseAquisition):
    def __init__(self, model, data, c, dim):
        super().__init__(model, data, c, dim)

    def evaluate(self, X):
        self.Q = torch.empty(X.mean.shape[0], 2*self.dim)
        self.S = torch.zeros(X.mean.shape[0], dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()

        self.fmin = 0 

        # Update confidence interval
        mean = X.mean.reshape(-1, self.dim)
        var = X.variance.reshape(-1, self.dim)
        self.Q[:, ::2] = mean - self.c.scale_beta*torch.sqrt(self.c.beta*var)
        self.Q[:, 1::2] = mean + self.c.scale_beta*torch.sqrt(self.c.beta*var)

        # Compute Safe Set
        self.S[:] = torch.all(self.Q[:, 2::2] > self.fmin, axis=1)

        if not torch.any(self.S):
            l = self.Q[:, 2::2]
            res = -1e10*torch.ones_like(X.mean[:, 0])
            safestPoint = torch.argmax(l.min(axis=1)[0])
            res[safestPoint] = -1e5
            return res

        # Set of maximisers
        l, u = self.Q[:, :2].T
        self.M[:] = False
        self.M[self.S] = u[self.S] >= torch.max(l[self.S], dim=0)[0]
        max_var = torch.max(u[self.M] - l[self.M], dim=0)[0]

        # Optimistic set of possible expanders
        self.G[:] = False
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        s = torch.logical_and(self.S, ~self.M)
        idx = s.clone()
        s[idx] = (torch.max((u[idx, 1:] - l[idx, 1:]), axis=1)[0] > max_var)
        idx = s.clone()
        s[idx] = torch.any(u[idx, 1:] - l[idx, 1:] > 0, axis=1)

        if torch.any(s):
            # set of safe expanders
            G_safe = torch.zeros(torch.count_nonzero(s), dtype=bool)
            sort_index = torch.max(u[s, :] - l[s, :], axis=1)[0].argsort()
            for index in reversed(sort_index):
                fModel = self.model.get_fantasy_model(self.points[s][index].reshape(1,-1),u[s][index].unsqueeze(-1))
                fModel.eval()

                pred = fModel.posterior(self.points[~self.S])
                l2 = pred.mean.detach() - self.c.scale_beta*torch.sqrt(self.c.beta*pred.variance.detach())
                G_safe[index] = torch.any(torch.all(l2[:,1:] >= self.fmin, axis=1))

                if G_safe[index]:
                    break

            self.G[s] = G_safe

        MG = torch.logical_or(self.M, self.G)
        value = torch.max((u - l), axis=1)[0]
        value[~MG] = -1e10
        return value.double()
