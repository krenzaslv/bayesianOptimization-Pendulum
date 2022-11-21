import typer
import numpy as np
from src.trainer import Trainer
from src.pendulum.simulator import simulate
from src.pendulum.dynamics import U_bo, dynamics_ideal, dynamics_real, U_star
from src.config import Config
from src.pendulum.plot import PlotPendulum
import torch
from src.tools.logger import load
from src.tools.file import clearFiles, makeGIF
import matplotlib.pyplot as plt
from src.pendulum.losses import PendulumErrorWithConstraint
from src.pendulum.config import Config as PendulumConfig
from src.models.GPModel import ExactMultiTaskGP
from rich import print

app = typer.Typer()

@app.command()
def make_gif():
    makeGIF()

@app.command()
def plot(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
    i : int= typer.Option(-1, help="Which epoch to plot"),
):
    clearFiles()
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    logger = load(config.save_file)

    X_star = simulate(config_pendulum, dynamics_ideal, U_star)

    plotter = PlotPendulum(X_star, config, config_pendulum)
    if i == -1:
        plotter.plot(logger)
    else:
        plotter.plotIdx(logger, i)
    plt.show()

@app.command()
def plot_gym(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
    i : int= typer.Option(-1, help="Which epoch to simulate"),
):
    clearFiles()
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    config_pendulum.sim_type="Gym"
    logger = load(config.save_file)
    config_pendulum.kp_bo = logger.x_k_buffer[i][0]
    config_pendulum.kd_bo = logger.x_k_buffer[i][1]

    simulate(config_pendulum, dynamics_real, U_bo)

@app.command()
def plot_end(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    clearFiles()
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)

    logger = load(config.save_file)

    X_star = simulate(config_pendulum, dynamics_ideal, U_star)

    plotter = PlotPendulum(X_star, config, config_pendulum)
    endIdx = len(logger.X_buffer) - 1
    plotter.plotIdx(logger, endIdx)
    plt.show()


@app.command()
def train(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    print("[green]Using: {}".format(config.aquisition))

    # TODO temporary hack to include solution in grid
    config.kp = config_pendulum.kp
    config.kd = config_pendulum.kd

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Pendulum dependent dynamics and losses
    X_star = simulate(config_pendulum, dynamics_ideal, U_star)
    loss = PendulumErrorWithConstraint(X_star, config_pendulum)
    # loss = PendulumError(X_star, config_pendulum)

    model = ExactMultiTaskGP(config, loss.dim)

    x_safe = torch.tensor([[0, 0]])  # ,[0.5,0.5], [-0.5,-0.5], [-0.5,0.5],[0.5,-0.5]])
    trainer = Trainer(config, X_star)
    trainer.train(loss, model, x_safe)
    trainer.logger.save(config.save_file)

if __name__ == "__main__":
    app()
