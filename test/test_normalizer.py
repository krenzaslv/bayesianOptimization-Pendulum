import pytest
import numpy as np
import torch

from src.normalizer import Normalizer, Normalizer2d
from src.random import rand2d_torch, rand_torch

torch.set_default_dtype(torch.float64)


def test_normalizerNormalizes2dData():
    x = rand2d_torch(
        1,
        3,
        2,
        4,
        200,
    )
    normalizer = Normalizer2d()
    x_n = normalizer.fit_transform(x)
    mean = x_n.mean(dim=0, keepdim=True)
    var = x_n.var(dim=0, keepdim=True)
    var_should = torch.tensor([1, 1])
    mean_should = torch.tensor([0, 0])

    assert (mean - mean_should).norm() == pytest.approx(0)
    assert (var - var_should).norm() == pytest.approx(0)
    assert (normalizer.itransform(x_n) - x).norm() == pytest.approx(0)


def test_normalizerNormalizes1dData():
    x = rand_torch(1, 3, 200)
    normalizer = Normalizer()
    x_n = normalizer.fit_transform(x)
    mean = x_n.mean()
    var = x_n.var()
    var_should = torch.tensor([1])
    mean_should = torch.tensor([0])

    assert (mean - mean_should).norm() == pytest.approx(0)
    assert (var - var_should).norm() == pytest.approx(0)
    assert (normalizer.itransform(x_n) - x).norm() == pytest.approx(0)
