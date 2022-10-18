import numpy as np
import torch
import pickle


class Logger:
    def __init__(self):
        self.X_buffer = []
        self.model_buffer = []
        self.y_k_buffer = []
        self.x_k_buffer = []
        self.xNormalizer_buffer = []
        self.yNormalizer_buffer = []
        self.y_min_buffer = []

    def log(self, model, X_k, x_k, y_k, xNormalizer, yNormalizer):
        model.eval()
        self.X_buffer.append(X_k)
        self.model_buffer.append(model)
        self.x_k_buffer.append(x_k)
        self.y_k_buffer.append(y_k)
        self.xNormalizer_buffer.append(xNormalizer)
        self.yNormalizer_buffer.append(yNormalizer)
        minIdx = np.argmin(np.array(self.y_k_buffer))
        self.y_min_buffer.append(self.y_k_buffer[minIdx])

    def getDataFromEpoch(self, i):
        return [
            self.model_buffer[i],
            self.X_buffer[: i + 1],
            torch.reshape(torch.cat(self.x_k_buffer), (-1, 2)),
            np.array(self.y_k_buffer),
            self.xNormalizer_buffer[i],
            self.yNormalizer_buffer[i],
            self.y_min_buffer[: i + 1],
        ]


def save(logger, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(logger, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
