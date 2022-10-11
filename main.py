import typer
import numpy as np
from src.dynamics import U
from src.trainer import train
from src.simulator import simulate

app = typer.Typer()


@app.command()
def run(
    kd: float = typer.Option(..., help="k_d constant in PD feedback"),
    kp: float = typer.Option(..., help="k_p constant in PD feedback"),
    m: float = typer.Option(..., help="mass"),
    l: float = typer.Option(..., help="length"),
    g: float = typer.Option(..., help="gravity"),
    pi: float = typer.Option(..., help="target x"),
):
    simulate()


@app.command()
def learn(
    kd: float = typer.Option(..., help="k_d constant in PD feedback"),
    kp: float = typer.Option(..., help="k_p constant in PD feedback"),
    m: float = typer.Option(..., help="mass"),
    l: float = typer.Option(..., help="length"),
    g: float = typer.Option(..., help="gravity"),
    pi: float = typer.Option(..., help="target x"),
    n_iterations: int = typer.Option(..., help="iterations"),
):
    train()


if __name__ == "__main__":
    app()
