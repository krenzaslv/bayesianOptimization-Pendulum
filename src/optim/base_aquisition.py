import torch
from torch.autograd import Variable
from src.tools.random import rand2d_torch

class BaseAquisition:

    def __init__(self, model, xNormalizer, t, c, logger):
        self.model = model
        self.xNormalizer = xNormalizer
        self.t = t + 1
        self.c = c
        self.logger = logger

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

    def optimize(self, training_steps):
        self.model.eval()

        t = self.getInitPoints()

        optimizer = torch.optim.Adam(
            [t], self.c.lr_aq, weight_decay=self.c.weight_decay_aq
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        for i in range(training_steps):
            optimizer.zero_grad()
            output = self.model.likelihood(self.model(t))
            loss = -self.loss(output).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if self.t == self.c.n_opt_samples - 1:
                loss = aquisition(self.model(t))
                self.logger.add_scalar("Loss/AQ_LAST", loss.min(), i)

        loss = -self.loss(self.model(t))

        minIdx = torch.argmin(loss)
        
        #Take next best value if already sampled. If not done leads to singular
        #matrix for finite grid
        if(self.c.skip_aready_samples):
            k = 1
            while t[minIdx] in self.model.train_inputs[0]:
                k += 1
                _ ,minIdx = torch.kthvalue(loss, k)

        return [t[minIdx], loss[minIdx]]

    def loss(self):
        pass
