import torch

from sklearn.preprocessing import StandardScaler
from torch._C import dtype
from bayopt.tools.math import scale

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

        if train_y.dim() > 1:
            self.y_scaler.fit(self.train_y.detach().numpy())

    def get_scaled_train_x(self):
        return scale(self.train_x, self.c.domain_end-self.c.domain_start).double() if self.c.normalize_data else self.train_x.double()

    def get_scaled_train_y(self):
        if self.train_y.dim() == 1:
            return self.train_y.double()
        else:
            return torch.from_numpy(self.y_scaler.transform(self.train_y.detach().numpy())).double() if self.c.normalize_data else self.train_y.double()

    def transform_x(self, X):
        return scale(X, self.c.domain_end-self.c.domain_start).double() if self.c.normalize_data else X 

    def transform_y(self, Y):
        return torch.from_numpy(self.y_scaler.inverse_transform(Y)).double() if self.c.normalize_data else Y 
