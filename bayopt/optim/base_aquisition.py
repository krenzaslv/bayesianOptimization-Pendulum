from numpy import argmax
import torch
from torch.autograd import Variable
from rich import print
from bayopt.tools.rand import rand2n_torch
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction, MCAcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.optimize import optimize_acqf
from botorch.utils import t_batch_mode_transform


class BaseAquisition(MCAcquisitionFunction):

    def __init__(self, model, t, c, logger, dim):
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.t = t + 1
        self.c = c
        self.logger = logger
        self.dim = dim
        self.parameter_set = self.getInitPoints()
        self.init_points = self.getInitPoints()
        self.n_double = 0

    def getInitPoints(self):
        if self.c.set_init == "random":
            t = rand2n_torch(self.c.domain_start, self.c.domain_end,
                             self.c.set_size, self.c.dim_params)
        else:
            init_points = []
            for i in range(self.c.dim_params):
                init_points.append(torch.linspace(self.c.domain_start[i],
                                   self.c.domain_end[i], self.c.set_size))
            X = torch.meshgrid(init_points, indexing="xy")
            init = torch.stack(X, dim=2)
            init = torch.reshape(init, (-1, self.c.dim_params))

            t = Variable(
                init,
                requires_grad=False,
            )
        return t
    # @t_batch_mode_transform()
    def forward(self, X):
        x = self.model.posterior(X)
        return self.evaluate(x)

    def evaluate(self, x):
        pass

    def optimize(self):
        bounds = torch.zeros(2, self.c.dim_params)
        bounds[0, :] = self.c.domain_start_p
        bounds[1, :] = self.c.domain_end_p
        # Xinit = gen_batch_initial_conditions(
        # self, bounds, q=10000, num_restarts=1, raw_samples=1
        # )
        # Xinit.reshape(1000,-1)
        # print(Xinit.shape)
        Xinit = self.getInitPoints()
        batch_candidates, batch_acq_values = gen_candidates_scipy(
                 initial_conditions=Xinit,
                 acquisition_function=self,
                 lower_bounds=bounds[0],
                 upper_bounds=bounds[1],
             )
        # batch_candidates, batch_acq_values = optimize_acqf(self,bounds, 3, 15, raw_samples=256, sequential=True)
        return [batch_candidates[torch.argmax(batch_acq_values)], batch_acq_values]

    def getNextPoint(self):
        self.model.eval()

        if self.c.set_init == "random":
            [nextX, loss] = self.optimize()
            # nextX = nextX[torch.argmax(loss)]
            loss = torch.argmax(loss)
        else:
            res = self.forward(self.parameter_set)

            nextX = self.parameter_set[torch.argmax(res)]
            loss = res.max()

            if self.model.models[0].train_inputs[0].shape[0] - self.n_double != self.model.models[0].train_inputs[0].unique(dim=0).shape[0]:
                print("[yellow][Warning][/yellow] Already sampled {}".format(nextX))
                self.n_double += 1

        return [nextX, loss]
