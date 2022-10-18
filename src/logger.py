import numpy as np
import torch


class Logger:
    def __init__(self):
        self.X_buffer = []
        self.model_buffer = []
        self.y_k_buffer = []
        self.x_k_buffer = []
        self.xNormalizer_buffer = []
        self.yNormalizer_buffer = []
        self.miny = 1e10

    def log(self, model, X_k, x_k, y_k, xNormalizer, yNormalizer):
        model.eval()
        self.X_buffer.append(X_k)
        self.model_buffer.append(model)
        self.x_k_buffer.append(x_k)
        self.y_k_buffer.append(y_k)
        self.xNormalizer_buffer.append(xNormalizer)
        self.yNormalizer_buffer.append(yNormalizer)

    def getDataFromEpoch(self, i):
        return [
            self.model_buffer[i],
            self.X_buffer[i],
            torch.reshape(torch.cat(self.x_k_buffer), (-1, 2)),
            np.array(self.y_k_buffer),
            self.xNormalizer_buffer[i],
            self.yNormalizer_buffer[i],
        ]
