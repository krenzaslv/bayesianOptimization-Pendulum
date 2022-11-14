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

class PendulumErrorWitOuthConstraint():
    def __init__(self, X_star, c):
        self.X_star = X_star
        self.c = c
        self.dim = 2

    def evaluate(self, k):
        config = copy.copy(self.c)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        norm = -np.linalg.norm(self.X_star - X_bo)/np.sqrt(self.c.n_simulation)
        c1 = 1.0 + np.random.normal(0,0.1)# Mock constraint

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm, c1]), X_bo]

class PendulumErrorWithConstraint():
    def __init__(self, X_star, c):
        self.X_star = X_star
        self.c = c
        self.dim = 3

    def evaluate(self, k):
        config = copy.copy(self.c)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        norm = -np.linalg.norm(self.X_star - X_bo)/np.sqrt(self.c.n_simulation)
        # c1 = -(np.max(np.linalg.norm(1/self.c.n_simulation*(self.X_star - X_bo), axis=1)) - np.pi/2)
        c2 = (np.absolute(k[0]) -3)
        c3 = (np.absolute(k[1]) -3)

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm, c2, c3]), X_bo]
