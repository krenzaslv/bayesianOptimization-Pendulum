import torch
from torch.autograd import Variable
from rich import print

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
        self.n_double = 0

    def getInitPoints(self):
        xs = torch.linspace(
            self.c.domain_start_p, self.c.domain_end_p, self.c.set_size
        )
        ys = torch.linspace(
            self.c.domain_start_d, self.c.domain_end_d, self.c.set_size
        )
        x, y = torch.meshgrid(xs, ys, indexing="xy")
        init = torch.stack((x, y), dim=2)
        init = torch.reshape(init, (-1, 2))

        t = Variable(
            self.xNormalizer.transform(init),
            requires_grad=False,
        )
        return t

    def getNextPoint(self):
        self.model.eval()

        [nextX, loss] = self.loss(self.model(self.parameter_set))

        if self.model.models[0].train_inputs[0].shape[0] - self.n_double != self.model.models[0].train_inputs[0].unique(dim=0).shape[0]:
            print("[yellow][Warning][/yellow] Already sampled {}".format(nextX))

        return [nextX, loss]

    def loss(self):
        pass
