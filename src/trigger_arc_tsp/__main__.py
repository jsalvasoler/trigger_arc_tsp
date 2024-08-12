from __future__ import annotations

import click

from trigger_arc_tsp.gurobi_model import gurobi_main


@click.group()
def cli() -> None:
    pass


@cli.command(name="gurobi", help="Evaluate the model")
@click.argument("instance", type=str)
@click.option("--time_limit_sec", type=int, default=60)
@click.option("--he", type=float, default=0.05)
def evaluate_command(instance: str | None, time_limit_sec: int = 60, he: float = 0.05) -> None:
    click.echo(f"Running Gurobi MIP model for instance: {instance}")
    gurobi_main(instance, time_limit_sec=time_limit_sec, heuristic_effort=he)


if __name__ == "__main__":
    cli()
