import torch
from sklearn.preprocessing import StandardScaler

class Normalizer:
    def __init__(self, normalize=True):
        self.scaler = StandardScaler(with_mean=normalize, with_std=normalize)

    def fit_transform(self, data):
        x = self.scaler.fit_transform(data.detach().numpy())
        return torch.from_numpy(x)

    def transform(self, data):
        x = self.scaler.transform(data.detach().numpy())
        return torch.from_numpy(x)

    def itransform(self, data):
        if data.dim() == 1:
            x = data.reshape(1, -1)
        x = self.scaler.inverse_transform(x.detach().numpy())

        return torch.from_numpy(x[0]) if data.dim() == 1 else torch.from_numpy(x)
