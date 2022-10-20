import numpy as np
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, config):
        self.X_buffer = []
        self.model_buffer = []
        self.y_k_buffer = []
        self.x_k_buffer = []
        self.xNormalizer_buffer = []
        self.yNormalizer_buffer = []
        self.y_min_buffer = []
        self.writer = SummaryWriter()
        self.c = config

    def log(self, model, i, X_k, x_k, y_k, xNormalizer, yNormalizer, loss_gp, loss_aq):
        model.eval()
        self.X_buffer.append(X_k)
        self.model_buffer.append(model)
        self.x_k_buffer.append(x_k)
        self.y_k_buffer.append(y_k)
        self.xNormalizer_buffer.append(xNormalizer)
        self.yNormalizer_buffer.append(yNormalizer)
        minIdx = np.argmin(np.array(self.y_k_buffer))
        self.y_min_buffer.append(self.y_k_buffer[minIdx])

        self.writer.add_scalar("Loss/GP", loss_gp, i)
        self.writer.add_scalar("Loss/Aquisition", loss_aq, i)
        if i == self.c.n_opt_samples - 1:
            self.writer.add_hparams(
                {
                    "lr_gp": self.c.lr_gp,
                    "lr_aq": self.c.lr_aq,
                    "weight_decay_gp": self.c.weight_decay_gp,
                    "weight_decay_aq": self.c.weight_decay_aq,
                    "n_opt_iterations_aq": self.c.n_opt_iterations_aq,
                    "n_opt_iterations_gp": self.c.n_opt_iterations_gp,
                    "gamma": self.c.gamma,
                    "weight_decay_aq": self.c.weight_decay_aq,
                    "n_opt_samples": self.c.n_opt_samples,
                },
                {
                    "hparam/GP": loss_gp,
                    "hparam/AQ": loss_aq,
                    "hparam/yMin": self.y_min_buffer[i],
                },
            )

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

    def save(self, path):
        self.writer.flush()
        self.writer.close()
        self.writer = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)


def load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
