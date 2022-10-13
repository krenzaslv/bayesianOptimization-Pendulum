import numpy as np
import torch


def rand(start, end, n, m=1):
    return (end - start) * np.random.rand(m, n) - (end - start) / 2 * np.ones(
        shape=(m, n)
    )


def rand_torch(start, end, n, m=1):
    return (end - start) * torch.rand(m, n) - (end - start) / 2 * torch.ones(m, n)
