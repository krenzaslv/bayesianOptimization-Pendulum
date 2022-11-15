import torch
from torch.autograd import Variable
from src.tools.random import rand2d_torch


class BaseAquisition:

    def __init__(self, model, xNormalizer, yNormalizer, t, c, logger, dim):
        self.model = model
        self.xNormalizer = xNormalizer
        self.yNormalizer = yNormalizer
        self.t = t + 1
        self.c = c
        self.logger = logger
        self.dim = dim
        self.parameter_set = self.getInitPoints()
        self.init_points = self.getInitPoints()

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

    def getNextPoint(self):
        self.model.eval()


        [nextX, loss] = self.loss(self.model(self.parameter_set))

        if self.model.models[0].train_inputs[0].shape[0] != self.model.models[0].train_inputs[0].unique(dim=0).shape[0]:
            print("Already sampled")

        return [nextX, loss]

    def loss(self):
        pass
