from __future__ import annotations

import click

from trigger_arc_tsp.gurobi_model import gurobi_main


@click.group()
def cli() -> None:
    pass


@cli.command(name="gurobi", help="Evaluate the model", context_settings={"show_default": True})
@click.argument("instance", type=str, required=True)
@click.option("--time_limit_sec", type=int, default=60)
@click.option("--he", type=float, default=0.05)
@click.option("--mip_start", type=bool, is_flag=True, default=False, help="Provide a MIP start")
@click.option(
    "--relax_obj_modeling", type=bool, is_flag=True, default=False, help="Relax integrality of objective modeling"
)
def evaluate_command(
    instance: str | None,
    time_limit_sec: int = 60,
    he: float = 0.05,
    *,
    mip_start: bool = False,
    relax_obj_modeling: bool = False,
) -> None:
    click.echo(f"Running Gurobi MIP model for instance: {instance}")
    gurobi_main(
        instance,
        time_limit_sec=time_limit_sec,
        heuristic_effort=he,
        mip_start=mip_start,
        relax_obj_modeling=relax_obj_modeling,
    )


if __name__ == "__main__":
    cli()
