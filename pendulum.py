import typer
import numpy as np
from src.trainer import Trainer
from src.pendulum.simulator import simulate
from src.pendulum.dynamics import dynamics_ideal, U_star
from src.config import Config
from src.pendulum.plot import PlotPendulum
import torch
from src.tools.logger import load
from src.tools.file import clearFiles, makeGIF
import matplotlib.pyplot as plt
from rich.progress import track
from src.losses.losses import PendulumError, PendulumErrorWithConstraint
from src.pendulum.config import Config as PendulumConfig
from src.models.GPModel import ExactGPModel, ExactMultiTaskGP 
from src.models.GPModel import ExactGPModel, ExactMultiTaskGP

# torch.set_default_dtype(torch.float64)
app = typer.Typer()

@app.command()
def make_gif():
    makeGIF()


@app.command()
def plot(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    clearFiles()
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    logger = load(config.save_file)

    X_star = simulate(config_pendulum, dynamics_ideal, U_star)

    plotter = PlotPendulum(X_star, config, config_pendulum)
    plotter.plot(logger)
    plt.show()



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
def train_constrained(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)

    # TODO temporary hack to include solution in grid
    config.kp = config_pendulum.kp
    config.kd = config_pendulum.kd

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    #Pendulum dependent dynamics and losses
    X_star = simulate(config_pendulum, dynamics_ideal, U_star)
    loss = PendulumErrorWithConstraint(X_star, config_pendulum)
    # loss = PendulumError(X_star, config_pendulum)

    model = ExactMultiTaskGP(config, loss.dim)

    x_safe = torch.tensor([[0,0]])#,[0.5,0.5], [-0.5,-0.5], [-0.5,0.5],[0.5,-0.5]])
    trainer = Trainer(config, X_star)
    trainer.train(loss, model, x_safe)
    trainer.logger.save(config.save_file)

if __name__ == "__main__":
    app()
