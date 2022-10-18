import numpy as np
import torch


def rand(start, end, n, m=1):
    return (end - start) * np.random.rand(m, n) - (end - start) / 2 * np.ones(
        shape=(m, n)
    )


def rand_torch(start, end, n, m=1):
    return (end - start) * torch.rand(m, n) - (end - start) / 2 * torch.ones(m, n)


def rand2d_torch(start1, end1, start2, end2, m):
    rand = torch.ones(m, 2)
    rand[:, 0] = rand_torch(start1, end1, 1, m)[:, 0]
    rand[:, 1] = rand_torch(start2, end2, 1, m)[:, 0]
    return rand
