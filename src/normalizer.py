import numpy as np


class Normalizer:
    def fit_transform(self, data, dim=0):
        if data.shape[0] > 1:
            self.mean = data.mean()
            self.std = data.std()
        else:
            self.mean = data.mean()
            self.std = 1
        return self.transform(data)
        # self.mean = 0
        # self.std = 1
        # return self.transform(data)

    def transform(self, data):
        return (data - self.mean) / self.std
        # return data

    def itransform(self, data):
        return data * self.std + self.mean
        # return data


class Normalizer2d:
    def fit_transform(self, data, dim=0):
        if data.shape[0] > 1:
            self.mean = data.mean(dim=dim, keepdim=True)
            self.std = data.std(dim=dim, keepdim=True)
        else:
            self.mean = data.mean(dim=dim, keepdim=True)
            self.std = 1
        return self.transform(data)
        # self.mean = 0
        # self.std = 1
        # return self.transform(data)

    def transform(self, data):
        return (data - self.mean) / self.std
        # return data

    def itransform(self, data):
        return data * self.std + self.mean
        # return data
