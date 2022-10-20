import typer
import numpy as np
from src.trainer import Trainer
from src.simulator import simulate
from src.dynamics import dynamics_ideal, U_star
from src.config import Config
from src.plot import Plot
import torch
import cProfile, pstats
from src.logger import load
from src.file import clearFiles, makeGIF
import matplotlib.pyplot as plt

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
def train(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
):
    config = Config(config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    X_star = simulate(config, dynamics_ideal, U_star)
    trainer = Trainer(config, X_star)
    trainer.train()
    trainer.logger.save(config.save_file)


if __name__ == "__main__":
    app()
