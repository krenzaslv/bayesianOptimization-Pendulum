import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm


class Plotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        sns.set_theme()
        grid = self.fig.add_gridspec(3, 2)
        self.ax = self.fig.add_subplot(grid[:, 0], projection="3d")
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, 1])
        self.ax4 = self.fig.add_subplot(grid[2, 1])
        self.X_bo_buffer = []
        self.y_min_buffer = []
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
        c,
        n_samples=50,
    ):
        plt.rc("font", family="serif")
        model.eval()
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax.set_ylim(c.domain_start, c.domain_end)
        self.ax.set_xlim(c.domain_start, c.domain_end)
        # self.ax.set_zlim(-1, 30)
        self.ax2.set_ylim(-4, 4)
        self.ax3.set_ylim(-4, 4)

        # x = xNormalizer.transform(torch.linspace(-20, 20, n_samples))
        grid = torch.linspace(c.domain_start, c.domain_end, n_samples)
        grid_x, grid_y = torch.meshgrid(grid, grid, indexing="xy")
        inp = torch.stack((grid_x, grid_y), dim=2).float()
        with torch.autograd.no_grad():
            out = model(xNormalizer.transform(inp))
        var = out.variance.detach().numpy()
        mean = out.mean.detach().numpy()
        # inp = xNormalizer.itransform(inp)

        minIdx = torch.argmin(train_y[: i + 1])
        x_min = train_x[minIdx]
        y_min = train_y[minIdx]
        self.y_min_buffer.append(y_min)

        self.ax.set_ylabel("kp")
        self.ax.set_xlabel("kd")
        self.ax.set_zlabel("f(x)")
        self.ax.plot_surface(
            inp[:, :, 0],
            inp[:, :, 1],
            yNormalizer.itransform(mean),
            vmax=10,
            alpha=0.3,
            facecolors=cm.jet(var / np.amax(var)),
        )

        self.ax.scatter(
            train_x[:i, 0],
            train_x[:i, 1],
            train_y[:i],  # yNormalizer.transform(train_y[: i + 1]),
            # s=300,
            color="red",
            marker="o",
        )

        self.ax.plot(
            train_x[i, 0],
            train_x[i, 1],
            train_y[i],  # yNormalizer.transform(train_y[: i + 1]),
            color="red",
            marker="X",
            markersize=40,
        )

        self.ax.plot(
            c.kp,
            c.kd,
            0,
            color="blue",
            marker="X",
            markersize=20,
        )

        with torch.autograd.no_grad():
            y_min = yNormalizer.itransform(
                model(xNormalizer.transform(x_min.reshape(1, 1, 2)))
                .mean.detach()
                .numpy()[0, 0]
            )

        self.ax.plot(
            x_min[0],
            x_min[1],
            y_min,
            color="black",
            marker="X",
            markersize=20,
        )

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
        self.ax2.set_ylabel("theta")
        self.ax2.set_xlabel("t")

        for bo in self.X_bo_buffer:
            self.ax3.plot(bo[:, 1], color="blue", alpha=0.1)

        self.ax3.plot(X_star[:, 1], color="red")
        self.ax3.plot(self.X_min[:, 1], color="orange")
        self.ax3.set_ylabel("theta_dot")
        self.ax3.set_xlabel("t")

        self.ax4.set_title(
            r"k_star =  [{} {} and k_hat = [{} {}]], error: {}".format(
                c.kp, c.kd, x_min[0], x_min[1], y_min
            )
        )
        self.ax4.set_ylabel("error")
        self.ax4.set_xlabel("t")
        self.ax4.plot(self.y_min_buffer[: i + 1])

        self.fig.show()
        plt.pause(0.001)
