from src.pendulum.dynamics import U_bo, dynamics_real
from src.pendulum.simulator import simulate
import math
import copy
import numpy as np
import torch

class PendulumError():
    def __init__(self, X_star, c):
        self.X_star = X_star
        self.c = c

    def evaluate(self, k):
        config = copy.copy(self.c)
        config.kp_bo = k[0]
        config.kd_bo = k[1]
        X_bo = simulate(config, dynamics_real, U_bo)
        # norm = np.linalg.norm(self.X_star - X_bo)/config.n_simulation
        norm = np.linalg.norm(self.X_star - X_bo)/self.c.n_simulation

        return [torch.tensor([k[0], k[1]]), -norm, X_bo]
