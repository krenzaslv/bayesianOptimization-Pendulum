import typer
import numpy as np
from bayopt.trainer import Trainer
from bayopt.pendulum.simulator import simulate, simulate_gym
from bayopt.pendulum.dynamics import U_pert, U_bo, dynamics_ideal, dynamics_real, U_star
from bayopt.config import Config
from bayopt.pendulum.plot import PlotPendulum
import torch
from bayopt.tools.logger import load
from bayopt.tools.file import clearFiles, makeGIF
import matplotlib.pyplot as plt
from bayopt.pendulum.losses import PendulumErrorWithConstraint, PendulumErrorWithConstraintRandomInit
from bayopt.pendulum.config import Config as PendulumConfig
from bayopt.models.GPModel import ExactMultiTaskGP, ExactGPModel
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
    def U_p(t, c): return U_pert(t, c, U_bo)  # Disturbance
    simulate_gym(config_pendulum, dynamics_real, U_p, render="human")

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
    endIdx = len(logger.X_buffer) -  1
    plotter.plotIdx(logger, endIdx)
    plt.show()

@app.command()
def train_gym(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    print("[green][Info][/green] Using: {}".format(config.aquisition))

    # TODO temporary hack to include solution in grid
    config.kp = config_pendulum.kp
    config.kd = config_pendulum.kd

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Pendulum dependent dynamics and losses
    loss = PendulumErrorWithConstraint(config_pendulum)

    model = ExactGPModel 

    x_safe = torch.tensor([[0.0, 0.0]])  # ,[0.5,0.5], [-0.5,-0.5], [-0.5,0.5],[0.5,-0.5]])
    trainer = Trainer(config)
    trainer.train(loss, model, x_safe)
    trainer.logger.save(config.save_file)

@app.command()
def train_gym_avg(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    config_path_pendulum: str = typer.Option("config_pendulum.txt", help="Path to config file"),
):
    config = Config(config_path)
    config_pendulum = PendulumConfig(config_path_pendulum)
    print("[green][Info][/green] Using: {}".format(config.aquisition))

    # TODO temporary hack to include solution in grid
    config.kp = config_pendulum.kp
    config.kd = config_pendulum.kd

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Pendulum dependent dynamics and losses
    loss = PendulumErrorWithConstraintRandomInit(config_pendulum)

    model = ExactGPModel 

    x_safe = torch.tensor([[0.0, 0.0], [0.2, 0.01]])  # ,[0.5,0.5], [-0.5,-0.5], [-0.5,0.5],[0.5,-0.5]])
    trainer = Trainer(config)
    trainer.train(loss, model, x_safe)
    trainer.logger.save(config.save_file)

if __name__ == "__main__":
    app()
