from src.optim.base_aquisition import BaseAquisition
import torch

class SafeOpt(BaseAquisition):
    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        super().__init__(model, xNormalizer, yNormalizer, t, c, logger, dim)

        inputs = self.getInitPoints()
        self.Q = torch.empty(inputs.shape[0], 2*dim)
        self.S = torch.zeros(inputs.shape[0], dtype=bool)
        self.G = self.S.clone()
        self.M = self.S.clone()
        self.fmin = 0


    def loss(self, x):
        # Update confidence interval
        for i in range(self.dim):
            self.Q[:,2*i] = x[i].mean - self.c.beta*torch.sqrt(x[i].variance)
            self.Q[:,2*i+1] = x[i].mean + self.c.beta*torch.sqrt(x[i].variance)

        #Compute Safe Set
        self.S[:] = torch.all(self.yNormalizer.itransform(self.Q[:,::2])[:,1:] > self.fmin, axis=1)
        if not torch.any(self.S):
            print("Couldnt find")

        print(self.Q[self.S].shape)

        # Set of maximisers
        l, u = self.Q[:, :2].T

        self.M[:] = False
        self.M[self.S] = u[self.S] >= torch.max(l[self.S], dim=0)[0]
        max_var = torch.max(u[self.M] - l[self.M], dim=0)[0]

        # # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        s = self.S

       # set of safe expanders
        G_safe = torch.zeros(torch.count_nonzero(s))
        sort_index = range(len(G_safe)) 

        for index in sort_index:
            for model in self.model.models[1:]:
                fModel = model.get_fantasy_model(
                        self.parameter_set[s][index].reshape(1,-1),
                        u[s,i][index].reshape(1,-1)
                        )
                pred = fModel(self.parameter_set[~self.S])
                l2 = pred.mean - self.c.beta*torch.sqrt(pred.variance)
                G_safe[index] = torch.any(l2 >= self.fmin)
        self.G[s] = G_safe

        MG = torch.logical_or(self.M, self.G)
        t = torch.zeros_like(x[0].mean)
        t[MG] = torch.max((u[MG] - l[MG]), axis=1)[0]
        t[~MG] = -1e10
        print(t)
        return t

 
