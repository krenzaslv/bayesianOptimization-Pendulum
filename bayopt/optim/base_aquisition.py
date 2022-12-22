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

import numpy as np
import copy


class BaseAquisition(MCAcquisitionFunction):

    def __init__(self, model, data, c, dim):
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.c = c
        self.dim = dim
        self.n_double = 0
        self.data = data

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
        self.points = X
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
        xInit = self.getInitPoints()
        batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=xInit,
            acquisition_function=self,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )
        # batch_candidates, batch_acq_values = optimize_acqf(self,bounds, 1, 1,raw_samples==) #, raw_samples=256)#

        return [batch_candidates[torch.argmax(batch_acq_values)], batch_acq_values]

    def optimizeSwarm(self):
        i = 0
        # N Restarts if no safe set is found
        while i == 0 or i < self.c.swarmopt_n_restarts and not self.hasSafePoints(x):
            x = self.getInitPoints()
            p = copy.deepcopy(x)

            with torch.no_grad():
                res = self.forward(x)

            fBest = res.max()
            pBest = x[torch.argmax(res)]

            x = self.getInitPoints()
            v = rand2n_torch(-np.abs(self.c.domain_end-self.c.domain_start),
                             np.abs(self.c.domain_end-self.c.domain_start), self.c.set_size, self.c.dim_params)
            if i > 0:
                print("[green][Info][/green] Did not find safe set at iteration {}.".format(i-1))

            inertia_scale = self.c.swarmopt_w

            # Swarmopt
            for j in range(self.c.swarmopt_n_iterations):
                # Update swarm velocities
                r_p = rand2n_torch(torch.tensor([0]).repeat(self.c.dim_params), torch.tensor(
                    [1]).repeat(self.c.dim_params), self.c.set_size, self.c.dim_params)
                r_g = rand2n_torch(torch.tensor([0]).repeat(self.c.dim_params), torch.tensor(
                    [1]).repeat(self.c.dim_params), self.c.set_size, self.c.dim_params)
                v = inertia_scale*v  \
                    + self.c.swarmopt_p*r_p * (p-x) + self.c.swarmopt_g*r_g*(pBest-x)
                inertia_scale *= 0.95
                
                # Update swarm position
                x += v
                x = clamp2dTensor(x, self.c.domain_start, self.c.domain_end)

                with torch.no_grad():
                    resTmp = self.forward(x)

                mask = resTmp > res
                p[mask] = x[mask]
                res[mask] = resTmp[mask]

                if res.max() > fBest:
                    fBest = res.max()
                    pBest = x[torch.argmax(res)]

            # print(torch.count_nonzero(x <0))

            i += 1

        if not self.hasSafePoints(x):
            print("[yellow][Warning][/yellow] Could not find safe set")
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
        print("nextX: {}".format(nextX))  # scale(nextX, self.c.domain_end-self.c.domain_start),
        return [nextX.detach(), loss.detach()]
