import numpy as np
import torch
import pickle
from src.tools.summary_writer import SummaryWriter
from src.models.GPModel import ExactGPModel, ExactMultiTaskGP
import copy

class Logger:
    def __init__(self, config):
        self.X_buffer = []
        self.model_buffer = []
        self.y_k_buffer = []
        self.x_k_buffer = []
        self.xNormalizer_buffer = []
        self.yNormalizer_buffer = []
        self.y_min_buffer = []
        self.loss_aq_buffer = []
        self.writer = SummaryWriter()
        self.c = config

    def log(self, model, i, X_k, x_k, y_k, xNormalizer, yNormalizer, loss_aq):
        model.eval()
        self.X_buffer.append(X_k)
        self.model_buffer.append(copy.deepcopy(model))
        self.x_k_buffer.append(x_k)
        self.y_k_buffer.append(y_k if y_k.dim == 1 else y_k[0])
        self.xNormalizer_buffer.append(xNormalizer)
        self.yNormalizer_buffer.append(yNormalizer)
        self.loss_aq_buffer.append(loss_aq)
        ykBuffer = np.array(self.y_k_buffer)
        minIdx = np.argmin(ykBuffer)
        self.y_min_buffer.append(self.y_k_buffer[minIdx])

        loss_aq = loss_aq if loss_aq.dim() == 0 else loss_aq[0]

        self.writer.add_scalar("Loss/Aquisition", loss_aq, i)
        self.writer.add_scalar("Loss/yMin", self.y_min_buffer[i], i)
        if i == self.c.n_opt_samples - 1:
            self.writer.add_hparams(
                {
                    "init_lenghtscale": self.c.init_lenghtscale,
                    "aquisition": self.c.aquisition,
                    "init_variance": self.c.init_variance,
                    "n_opt_samples": self.c.n_opt_samples,
                    "beta": self.c.beta,
                    "ucb_use_set": self.c.ucb_use_set,
                    "ucb_set_n": self.c.ucb_set_n,
                },
                {
                    # "hparam/GP": loss_gp,
                    # "hparam/AQ": loss_aq,
                    # "hparam/yMin": self.y_min_buffer[i],
                    "Loss/GP": None,
                    "Loss/yMin": None,
                    "Loss/Aquisition": None,
                    # "GP/lengthscale_p": None,
                    # "GP/lengthscale_d": None,
                },
            )

    def getDataFromEpoch(self, i):
        # model = ExactMultiTaskGP(self.c, len(self.loss_aq_buffer[0]))
        # model.load_state_dict(self.c, self.model_buffer[i])
        # model.eval()
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
