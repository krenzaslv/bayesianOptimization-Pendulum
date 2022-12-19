from numpy import argmax
import torch
from torch.autograd import Variable
from rich import print
from bayopt.tools.rand import rand2n_torch
from bayopt.tools.math import scale
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction, MCAcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.optimize import optimize_acqf
from botorch.utils import t_batch_mode_transform
from bayopt.tools.math import clamp2dTensor
import copy


class BaseAquisition(MCAcquisitionFunction):

    def __init__(self, model, t, c, logger, dim):
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.t = t + 1
        self.c = c
        self.logger = logger
        self.dim = dim
        self.n_double = 0
        self.X_pending = None
        self.maxIter = 10

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
            init = torch.stack([x.flatten() for x in X]).T
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
        bounds[0, :] = torch.from_numpy(self.c.domain_start)
        bounds[1, :] = torch.from_numpy(self.c.domain_end)
        # Xinit = gen_batch_initial_conditions(
        # self, bounds, q=10000, num_restarts=1, raw_samples=1
        # )
        # Xinit.reshape(1000,-1)
        # print(Xinit.shape)
        Xinit = self.getInitPoints()
        self.points = xInit
        batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=Xinit,
            acquisition_function=self,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )
        # batch_candidates, batch_acq_values = optimize_acqf(self,bounds, 1, 1,raw_samples==) #, raw_samples=256)#

        return [batch_candidates[torch.argmax(batch_acq_values)], batch_acq_values]

    def optimizeSwarm(self):
        xInit = self.getInitPoints()
        pInit = copy.deepcopy(xInit)
        vbounds = torch.tensor([0.1])
        vInit = rand2n_torch(-vbounds.repeat(xInit.shape[1], 1), vbounds.repeat(
            xInit.shape[1], 1), xInit.shape[0], xInit.shape[1])

        self.points = xInit
        res = self.forward(xInit)

        fBest = res.max()
        pBest = xInit[torch.argmax(res)]
        i = 0
        while i < self.maxIter and not self.hasSafePoints(xInit):
            xInit = self.getInitPoints()

            for j in range(100):
                xInit += vInit
                self.points = xInit
                # TODO make better
                xInit = clamp2dTensor(scale(xInit, self.c.domain_end-self.c.domain_start), 0,
                                      1) if self.c.normalize_data else clamp2dTensor(xInit, self.c.domain_start, self.c.domain_end)

                resTmp = self.forward(xInit)

                mask = resTmp > res

                res[mask] = resTmp[mask]

                pInit[mask] = xInit[mask]

                vInit = rand2n_torch(-vbounds.repeat(xInit.shape[1], 1), vbounds.repeat(
                    xInit.shape[1], 1), xInit.shape[0], xInit.shape[1])


            i += 1

        if not self.hasSafePoints(xInit):
            print("Could not find safe set")
        fBest = res.max()

        return [pBest, fBest]

    def hasSafePoints(self, X):
        xTmp = self.model.posterior(X)
        l = xTmp.mean - torch.sqrt(self.c.beta*xTmp.variance)
        S = torch.all(l[:, 1:] > 0, axis=1)
        return torch.any(S)

    def getNextPoint(self):
        self.model.eval()

        if self.c.acf_optim == "opt":
            [nextX, loss] = self.optimize()
            loss = torch.argmax(loss)

        elif self.c.acf_optim == "swarm":
            [nextX, loss] = self.optimizeSwarm()

        else:
            X = self.getInitPoints()
            self.points = X
            res = self.forward(X)

            nextX = X[torch.argmax(res)]
            loss = res.max()

            if self.model.models[0].train_inputs[0].shape[0] - self.n_double != self.model.models[0].train_inputs[0].unique(dim=0).shape[0]:
                print("[yellow][Warning][/yellow] Already sampled {}".format(nextX))
                self.n_double += 1
        print("nextX: {}/{}".format(scale(nextX, self.c.domain_end-self.c.domain_start), nextX))
        return [nextX, loss]
