from bayopt.pendulum.dynamics import U_bo, U_pert, dynamics_real, U_star, dynamics_ideal
from bayopt.pendulum.simulator import simulate
from bayopt.tools.rand import rand
from bayopt.losses.loss import Loss
import copy
import numpy as np
import torch


class PendulumError(Loss):
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

class PendulumErrorWithConstraintRandomInit(Loss):
    def __init__(self, c, N =10):
        self.c = c
        self.dim = 3

        config = copy.copy(self.c)
        config.kp_bo = 0.0
        config.kd_bo = 0.0
        self.N = N

    def evaluate(self, k):
        c1 = 0
        c2 = 0
        norm = 0
        for i in range(self.N):
            config = copy.deepcopy(self.c)
            config.kp_bo = k[0]
            config.kd_bo = k[1]
            if i != self.N -1:
                config.x0[0] = rand(-np.pi/2, 0, 1)
                config.x0[1] = 0 #rand(-0.1, 0.1, 1)

            def U_p(t, c): return U_pert(t, c, U_bo)  # Disturbance

            X_bo = simulate(config, dynamics_real, U_p)
            X_star = simulate(config, dynamics_ideal, U_star)

            norm += (0.2 - np.linalg.norm(X_star-X_bo)/np.sqrt(self.c.n_simulation))/self.N
            c1 += (np.pi*np.pi/ 8- np.max((X_star[:, 0] - X_bo[:, 0])**2))/self.N
            # c2 += (np.pi*np.pi - np.max((X_star[:, 1] - X_bo[:, 1])**2))/self.N
            c2 += (3 - np.max(np.abs(X_bo[:, 1])))/self.N

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm, c1, c2]), X_bo]


class PendulumErrorWithConstraint(Loss):
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
        c1 = np.pi - np.max((self.X_star[:, 0] - X_bo[:, 0])**2)
        c2 = np.pi - np.max((self.X_star[:, 1] - X_bo[:, 1])**2)
        # c3 = (np.absolute(k[1]) -3)

        return [torch.tensor([k[0], k[1]]), torch.tensor([norm, c1, c2]), X_bo]
