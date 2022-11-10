import torch
from torch.autograd import Variable
from src.tools.random import rand2d_torch

class BaseAquisition:

    def __init__(self, model, xNormalizer, t, c, logger, dim):
        self.model = model
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c
        self.logger = logger
        self.dim = dim

    def getInitPoints(self):
        if self.c.ucb_use_set:
            xs = torch.linspace(
                self.c.domain_start_p, self.c.domain_end_p, self.c.ucb_set_n
            )
            ys = torch.linspace(
                self.c.domain_start_d, self.c.domain_end_d, self.c.ucb_set_n
            )
            x, y = torch.meshgrid(xs, ys, indexing="xy")
            init = torch.stack((x, y), dim=2)
            init = torch.reshape(init, (-1, 2))
            init = torch.cat((init, torch.tensor([[self.c.kp, self.c.kd]])), 0)

        else:
            init = rand2d_torch(
                self.c.domain_start_p,
                self.c.domain_end_p,
                self.c.domain_start_d,
                self.c.domain_end_d,
                self.c.n_sample_points,
            )

        t = Variable(
            self.xNormalizer.transform(init),
            requires_grad=True,
        )
        return t

    def optimize(self):
        self.model.eval()

        t = self.getInitPoints()

        loss = -self.loss(self.model(t)) # TODO maybe use likelihood
        loss_perf = loss if loss.dim() == 1 else loss[:,0]
        minIdx = torch.argmin(loss_perf)        

        if(self.c.skip_aready_samples):
            k = 1
            while t[minIdx] in self.model.train_inputs[0]:
                k += 1
                _ ,minIdx = torch.kthvalue(loss_perf, k)

        return [t[minIdx], loss[minIdx]]

    def loss(self):
        pass
