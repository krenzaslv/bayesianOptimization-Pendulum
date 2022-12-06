import torch
from torch.autograd import Variable
from rich import print
from bayopt.tools.rand import rand2n_torch
from botorch.acquisition import AnalyticAcquisitionFunction

class BaseAquisition(AnalyticAcquisitionFunction):

    def __init__(self, model, t, c, logger, dim):
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.t = t + 1
        self.c = c
        self.logger = logger
        self.dim = dim
        self.parameter_set = self.getInitPoints()
        self.init_points = self.getInitPoints()
        self.n_double = 0

    def getInitPoints(self):
        if self.c.set_init== "random":
            t = rand2n_torch(self.c.domain_start_p,self.c.domain_end_p, self.c.set_size, self.c.dim_params)
        else:
            init_points = []
            for i in range(self.c.dim_params):
                init_points.append(torch.linspace(self.c.domain_start_p, self.c.domain_end_p, self.c.set_size))
            X = torch.meshgrid(init_points, indexing="xy")
            init = torch.stack(X, dim=2)
            init = torch.reshape(init, (-1, self.c.dim_params))

            t = Variable(
                init,
                requires_grad=False,
            )
        return t

    def forward(self, X):
        pass

    def getNextPoint(self):
        self.model.eval()

        res = self.forward(self.model(self.parameter_set))

        nextX = self.parameter_set[torch.argmax(res)]
        loss = res.max()
        
        if self.model.models[0].train_inputs[0].shape[0] - self.n_double != self.model.models[0].train_inputs[0].unique(dim=0).shape[0]:
            print("[yellow][Warning][/yellow] Already sampled {}".format(nextX))
            self.n_double += 1

        return [nextX, loss]
