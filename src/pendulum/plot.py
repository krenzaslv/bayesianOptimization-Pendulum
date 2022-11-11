import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from rich.progress import track


class PlotPendulum:
    def __init__(self, X_star, config, config_pendulum):
        self.setupAxis()
        self.X_bo_buffer = []
        self.y_min_buffer = []
        self.X_star = X_star
        self.config = config
        self.config_pendulum = config_pendulum

    def setupAxis(self):
        # plt.ion()
        self.fig = plt.figure()
        sns.set_theme()
        grid = self.fig.add_gridspec(3, 2)
        self.ax = self.fig.add_subplot(grid[:, 0], projection="3d")
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, 1])
        self.ax4 = self.fig.add_subplot(grid[2, 1])

    def clearSurface(self):
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax.set_ylim(self.config.domain_start_d, self.config.domain_end_d)
        self.ax.set_xlim(self.config.domain_start_p, self.config.domain_end_p)
        self.ax.set_ylabel("kp")
        self.ax.set_xlabel("kd")
        self.ax.set_zlabel("f(x)")
        self.ax2.set_ylabel("theta")
        self.ax2.set_xlabel("t")
        self.ax3.set_ylabel("theta_dot")
        self.ax3.set_xlabel("t")
        self.ax4.set_ylabel("error")
        self.ax4.set_xlabel("t")
        self.ax2.set_ylim(-4, 4)
        self.ax3.set_ylim(-4, 4)
        self.ax2.set_ylim(-4, 4)
        self.ax3.set_ylim(-4, 4)

    def createGrid(self):
        grid_x = torch.linspace(
            self.config.domain_start_p,
            self.config.domain_end_p,
            self.config.plotting_n_samples,
        )
        grid_y = torch.linspace(
            self.config.domain_start_d,
            self.config.domain_end_d,
            self.config.plotting_n_samples,
        )
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
        inp = torch.stack((grid_x, grid_y), dim=2).float()
        # inp= torch.reshape(inp, (-1, 2))
        return inp

    def plotSurface(
        self, i, inp, mean, var, x, y, x_min, y_min, xNormalizer, yNormalizer, model
    ):
        inp = inp.reshape(self.config.plotting_n_samples,self.config.plotting_n_samples, 2)
        self.ax.plot_surface(
            inp[:, :, 0],
            inp[:, :, 1],
            yNormalizer.itransform(mean)[:,0].reshape(self.config.plotting_n_samples,self.config.plotting_n_samples),
            vmax=10,
            alpha=0.3,
            facecolors=cm.jet(var[:,0] .reshape(self.config.plotting_n_samples,self.config.plotting_n_samples)/ np.amax(var)),
        )

        self.ax.scatter(
            x[:i, 0],
            x[:i, 1],
            y[:i],
            color="red",
            marker="o",
        )

        self.ax.plot(
            x[i, 0],
            x[i, 1],
            y[i],
            color="red",
            marker="X",
            markersize=40,
        )

        self.ax.plot(
            self.config_pendulum.kp,
            self.config_pendulum.kd,
            0,
            color="blue",
            marker="X",
            markersize=20,
        )

        # with torch.autograd.no_grad():
        #     y_min = yNormalizer.itransform(
        #         model(xNormalizer.transform(x_min.reshape(1, 1, 2)))
        #         .mean.detach()
        #         .numpy()[0, 0]
        #     )

        self.ax.plot(
            x_min[0],
            x_min[1],
            y_min,
            color="black",
            marker="X",
            markersize=20,
        )

    def plotIdx(self, logger, i):
        self.clearSurface()

        [
            model,
            X,
            x,
            y,
            xNormalizer,
            yNormalizer,
            y_min_buffer,
        ] = logger.getDataFromEpoch(i)

        inp = self.createGrid().reshape(-1,2)
        with torch.autograd.no_grad():
            out = model.likelihood(model(xNormalizer.transform(inp)))

        var = out.variance.detach().numpy()
        mean = out.mean.detach()

        maxIdx = np.argmax(y[: i + 1])
        x_min = x[maxIdx]
        y_min = y[maxIdx]

        self.plotSurface(
            i, inp, mean, var, x, y, x_min, y_min, xNormalizer, yNormalizer, model
        )

        for X_k in X:
            self.ax2.plot(X_k[:, 0], color="blue", alpha=0.1)
        self.ax2.plot(logger.X_buffer[0][:, 0], color="orange", label="initial")
        self.ax2.plot(logger.X_buffer[maxIdx][:, 0], color="green", label="bestfound")

        for X_k in X:
            self.ax3.plot(X_k[:, 1], color="blue", alpha=0.1)
        self.ax3.plot(logger.X_buffer[0][:, 1], color="orange", label="initial")
        self.ax3.plot(logger.X_buffer[maxIdx][:, 1], color="green", label="bestfound")

        self.ax4.set_title(
            r"k_star =  [{} {} and k_hat = [{} {}]], error: {}".format(
                self.config_pendulum.kp, self.config_pendulum.kd, x_min[0], x_min[1], y_min
            )
        )
        self.ax4.plot(y_min_buffer)

        self.ax2.plot(self.X_star[:, 0], color="red")
        self.ax2.plot(
            self.config_pendulum.pi * np.ones(self.X_star.shape[0]), color="red", label="ideal"
        )
        self.ax3.plot(
            self.config_pendulum.pi * np.ones(self.X_star.shape[0]), color="red", label="ideal"
        )
        self.ax2.legend()
        self.ax3.legend()
        plt.savefig("data/{0:0>3}.png".format(i))

        plt.pause(0.001)

    def plot(
        self,
        logger,
    ):
        N = len(logger.X_buffer)
        plt.pause(3)

        for i in track(range(N), description="Generating Plot..."):
            self.plotIdx(logger, i)
