import pytest
import numpy as np
import torch

from src.dynamics import dynamics_ideal, dynamics_real, U_star, U_bo, U_pert

torch.set_default_dtype(torch.float64)


class MockConfig:
    def __init__(self):
        self.kp = 4
        self.kd = 1
        self.k1 = 3
        self.k2 = 2
        self.kp_bo = 4
        self.kd_bo = 1
        self.m = 1
        self.g = 9.81
        self.l = 1
        self.dt = 5
        self.pi = 0
        self.max_torque = 1e10


def test_correctBajesianParametersRecoverUStar():
    c = MockConfig()
    x_0 = np.array([1, 2])
    U1 = U_star(x_0, c)
    U2 = U_pert(x_0, c, U_bo)
    assert U1 == pytest.approx(U2)


def test_correctBajesianParametersRecoverDynamicsIdeal():
    c = MockConfig()
    x_0 = np.array([1, 2])
    x_ideal = dynamics_ideal(x_0, U_star, c)
    x_real = dynamics_real(x_0, U_bo, c)

    assert np.linalg.norm(x_ideal - x_real) == pytest.approx(0.0)
