import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Plotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(221, projection="3d")
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.X_bo_buffer = []
        self.miny = 1e10

    def plot(
        self,
        model,
        train_x,
        train_y,
        i,
        X_star,
        X_bo,
        config,
        xNormalizer,
        yNormalizer,
        n_samples=200,
    ):
        model.eval()
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax2.set_ylim(-2, 4)
        self.ax3.set_ylim(-4, 4)

        x = torch.linspace(-3, 3, n_samples)
        x, y = torch.meshgrid(x, x, indexing="xy")
        inp = torch.stack((x, y), dim=2).float()
        with torch.autograd.no_grad():
            out = model(inp)
        var = out.variance.detach().numpy()
        mean = out.mean.detach().numpy()
        inp = xNormalizer.itransform(inp)

        minIdx = torch.argmin(train_y[: i + 1])
        x_min = train_x[minIdx]
        y_min = train_y[minIdx]
        print(x_min)

        self.ax.plot_surface(
            inp[:, :, 0],
            inp[:, :, 1],
            mean,
            alpha=0.1,
            facecolors=cm.jet(var / np.amax(var)),
        )

        self.ax.scatter(
            train_x[: i + 1, 0],
            train_x[: i + 1, 1],
            train_y[: i + 1],  # yNormalizer.transform(train_y[: i + 1]),
            **{"color": "red", "marker": "o"},
        )
        self.ax.plot(
            x_min[0],
            x_min[1],
            y_min,
            color="black",
            marker="X",
            markersize=20,
        )

        if train_y[i] < 1000:
            self.X_bo_buffer.append(X_bo)
            miny = torch.min(train_y[: i + 1])
            if i == 0 or self.miny > miny:
                self.X_min = X_bo
                self.miny = miny

        for bo in self.X_bo_buffer:
            self.ax2.plot(bo[:, 0], color="blue", alpha=0.1)

        self.ax2.plot(X_star[:, 0], color="red")
        self.ax2.plot(self.X_min[:, 0], color="orange")
        self.ax2.plot(config.pi * np.ones(X_star.shape[0]), color="red")

        for bo in self.X_bo_buffer:
            self.ax3.plot(bo[:, 1], color="blue", alpha=0.1)

        self.ax3.plot(X_star[:, 1], color="red")
        self.ax3.plot(self.X_min[:, 1], color="orange")

        self.fig.show()
        plt.pause(0.001)
