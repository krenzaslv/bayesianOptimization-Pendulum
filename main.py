import typer
import numpy as np
from src.trainer import Trainer
from src.simulator import simulate
from src.dynamics import dynamics_ideal, U_star
from src.config import Config

app = typer.Typer()


@app.command()
def run(
    kd: float = typer.Option(..., help="k_d disturbance"),
    kp: float = typer.Option(..., help="k_p disturbance"),
    kd_bo: float = typer.Option(..., help="initial k_d_bo"),
    kp_bo: float = typer.Option(..., help="initial k_p_bo"),
    k1: float = typer.Option(..., help="Linear disturbance"),
    k2: float = typer.Option(..., help="Derivative disturbance"),
    m: float = typer.Option(..., help="mass"),
    l: float = typer.Option(..., help="length"),
    g: float = typer.Option(..., help="gravity"),
    pi: float = typer.Option(..., help="target x"),
    n_simulation: int = typer.Option(..., help="n simualtion steps"),
    n_iterations: int = typer.Option(..., help="iterations"),
    x0: float = typer.Option(..., help="x0"),
    x0_dot: float = typer.Option(..., help="x0_dot"),
):
    config = Config(
        m,
        l,
        g,
        k1,
        k2,
        kd,
        kp,
        kd_bo,
        kp_bo,
        pi,
        n_iterations,
        np.array([[x0, x0_dot]]),
        n_simulation,
    )
    print(simulate(config))


@app.command()
def learn(
    kd: float = typer.Option(..., help="k_d disturbance"),
    kp: float = typer.Option(..., help="k_p disturbance"),
    kd_bo: float = typer.Option(..., help="initial k_d_bo"),
    kp_bo: float = typer.Option(..., help="initial k_p_bo"),
    k1: float = typer.Option(..., help="Linear disturbance"),
    k2: float = typer.Option(..., help="Derivative disturbance"),
    m: float = typer.Option(..., help="mass"),
    l: float = typer.Option(..., help="length"),
    g: float = typer.Option(..., help="gravity"),
    pi: float = typer.Option(..., help="target x"),
    n_simulation: int = typer.Option(..., help="n simualtion steps"),
    n_iterations: int = typer.Option(..., help="iterations"),
    x0: float = typer.Option(..., help="x0"),
    x0_dot: float = typer.Option(..., help="x0_dot"),
):
    config = Config(
        m,
        l,
        g,
        k1,
        k2,
        kd,
        kp,
        kd_bo,
        kp_bo,
        pi,
        n_iterations,
        np.array([[x0, x0_dot]]),
        n_simulation,
    )

    X_star = simulate(config, dynamics_ideal, U_star)
    trainer = Trainer(config, X_star)
    trainer.train()


if __name__ == "__main__":
    app()
