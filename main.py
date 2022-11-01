import typer
import numpy as np
from src.trainer import Trainer
from src.pendulum.simulator import simulate
from src.pendulum.dynamics import dynamics_ideal, U_star
from src.config import Config
from src.tools.plot import Plot
import torch
import cProfile, pstats
from src.tools.logger import load
from src.tools.file import clearFiles, makeGIF
import matplotlib.pyplot as plt
from rich.progress import track
from src.losses.losses import pendulum_loss

# torch.set_default_dtype(torch.float64)
app = typer.Typer()


@app.command()
def profile(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):
    profiler = cProfile.Profile()
    profiler.enable()
    train(config_path)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats()


@app.command()
def make_gif():
    makeGIF()


@app.command()
def plot(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):
    clearFiles()
    config = Config(config_path)
    logger = load(config.save_file)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    X_star = simulate(config, dynamics_ideal, U_star)
    plotter = Plot(X_star, config)
    plotter.plot(logger)
    plt.show()


@app.command()
def plot_real(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):

    config = Config(config_path)
    grid_x = torch.linspace(
        config.domain_start_p,
        config.domain_end_p,
        config.plotting_n_samples,
    )
    grid_y = torch.linspace(
        config.domain_start_d,
        config.domain_end_d,
        config.plotting_n_samples,
    )
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
    inp = np.stack((grid_x, grid_y), axis=2)
    out = np.zeros(shape=(inp.shape[0], inp.shape[1]))
    X_star = simulate(config, dynamics_ideal, U_star)
    trainer = Trainer(config, X_star)
    ax = plt.axes(projection="3d")
    ax.plot(config.kp, config.kd, 0, marker="X", markersize="20")
    for i in track(range(inp.shape[0]), description="Simulating..."):
        for j in range(inp.shape[1]):
            [x_k, y_k, _] = trainer.loss(inp[i, j])
            out[i, j] = y_k
    ax.plot_surface(inp[:, :, 0], inp[:, :, 1], out)
    plt.show()


@app.command()
def plot_end(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):
    clearFiles()
    config = Config(config_path)
    logger = load(config.save_file)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    X_star = simulate(config, dynamics_ideal, U_star)
    plotter = Plot(X_star, config)
    endIdx = len(logger.X_buffer) - 1
    plotter.plotIdx(logger, endIdx)
    plt.show()


@app.command()
def train_pendulum(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):
    config = Config(config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    X_star = simulate(config, dynamics_ideal, U_star)
    trainer = Trainer(config, X_star)
    trainer.train(pendulum_loss)
    trainer.logger.save(config.save_file)


if __name__ == "__main__":
    app()
