class Normalizer:
    def fit_transform(self, data, dim=0):
        if data.shape[0] > 1:
            self.mean = data.mean(dim=dim).detach().numpy()
            self.std = data.std(dim=dim).detach().numpy()
        else:
            self.mean = data[0].detach().numpy()
            self.std = 1
        return self.transform(data)

    def transform(self, data):
        return (data - self.mean) / self.std
        # return data

    def itransform(self, data):
        return data * self.std + self.mean
        # return data
