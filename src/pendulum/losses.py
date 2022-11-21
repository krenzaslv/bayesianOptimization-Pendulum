from src.pendulum.dynamics import U_bo, dynamics_real
from src.pendulum.simulator import simulate
import copy
import numpy as np
import torch


class PendulumError():
    def __init__(self, X_star, c):
        self.X_star = X_star
        self.c = c
        self.dim = 1

    def evaluate(self, k):
        config = copy.copy(self.c)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        norm = -np.linalg.norm(self.X_star - X_bo)/np.sqrt(self.c.n_simulation)

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm]), X_bo]


class PendulumErrorWithConstraint():
    def __init__(self, X_star, c):
        self.X_star = X_star
        self.c = c
        self.dim = 3

        config = copy.copy(self.c)
        config.kp_bo = 0.0
        config.kd_bo = 0.0

        X_init = simulate(config, dynamics_real, U_bo)

        self.init_norm = np.linalg.norm(self.X_star - X_init)/np.sqrt(self.c.n_simulation)

    def evaluate(self, k):
        config = copy.copy(self.c)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)

        norm = self.init_norm - np.linalg.norm(self.X_star - X_bo)/np.sqrt(self.c.n_simulation)
        # # c1 = self.init_norm + 0.5*norm
        # c1 = np.max((self.X_star[:,0] - X_bo[:,0])**2)
        c1 = np.pi*np.pi/4 - np.max((self.X_star[:, 0] - X_bo[:, 0])**2)
        c2 = np.pi*np.pi - np.max((self.X_star[:, 1] - X_bo[:, 1])**2)
        # c3 = (np.absolute(k[1]) -3)

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm, c1, c2]), X_bo]