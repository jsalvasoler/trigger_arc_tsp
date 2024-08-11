from __future__ import annotations

import click

from trigger_arc_tsp.gurobi_model import gurobi_main


@click.group()
def cli() -> None:
    pass


@cli.command(name="gurobi", help="Evaluate the model")
@click.argument("instance", type=str)
def evaluate_command(instance: str | None) -> None:
    click.echo(f"Running Gurobi MIP model for instance: {instance}")
    gurobi_main(instance)


if __name__ == "__main__":
    cli()
