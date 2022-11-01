from src.pendulum.dynamics import U_bo, dynamics_real
from src.pendulum.simulator import simulate
import math
import copy
import numpy as np
import torch

#TODO disentangle from config...
def pendulum_loss(k, X_star, c):
    config = copy.copy(c)
    config.kp_bo = k[0]
    config.kd_bo = k[1]
    X_bo = simulate(config, dynamics_real, U_bo)
    norm = np.linalg.norm(X_star - X_bo)/c.n_simulation

    return [torch.tensor([k[0], k[1]]), norm, X_bo]