import typer
import numpy as np
from src.trainer import Trainer
from src.simulator import simulate
from src.dynamics import dynamics_ideal, U_star
from src.config import Config
import torch

app = typer.Typer()


@app.command()
def run():
    # print(simulate(config))
    print("test")


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


if __name__ == "__main__":
    app()
