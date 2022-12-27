import torch

from sklearn.preprocessing import StandardScaler
from torch._C import dtype
from bayopt.tools.math import scale
import numpy as np

class Data:

    def __init__(self, config, train_x = None, train_y = None):
        self.c = config

        self.train_x = train_x
        self.train_y = train_y

        self.y_scaler = StandardScaler()

        if train_y is not None and train_y.dim() > 1:
            self.y_scaler.fit(self.train_y.detach().numpy())

    def append_data(self, train_x, train_y):
        if self.train_x == None and self.train_y == None:
            self.train_x = train_x #if train_x.dim != 1 else train_x.reshape(1,-1)
            self.train_y = train_y
        else:
            self.train_x = torch.cat([self.train_x, train_x])#.reshape(1,-1)])
            self.train_y = torch.cat([self.train_y, train_y])
