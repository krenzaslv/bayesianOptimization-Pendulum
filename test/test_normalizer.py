import pytest
import numpy as np
import torch

from src.tools.normalizer import Normalizer
from src.tools.rand import rand2d_torch, rand_torch

torch.set_default_dtype(torch.float64)


def test_normalizerNormalizes2dData():
    x = torch.rand(200, 5)
    x[:0] += 1
    x[:1] += 2
    x[:2] += 3
    x[:3] += 4
    x[:4] += 5
    normalizer = Normalizer()
    x_n = normalizer.fit_transform(x)
    mean = x_n.mean(dim=0, keepdim=True)
    var = x_n.var(dim=0, keepdim=True)
    var_should = torch.tensor([1, 1, 1, 1, 1])
    mean_should = torch.tensor([0, 0, 0, 0, 0])

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
