from __future__ import annotations

import click

from trigger_arc_tsp.gurobi_model import gurobi_main
from trigger_arc_tsp.tsp_search import heuristic_search


@click.group()
def cli() -> None:
    pass


INSTANCES_SOLVED_TO_OPTIMALITY = [
    "instances_release_1/grf1.txt",
    "instances_release_1/grf2.txt",
    "instances_release_1/grf5.txt",
    "instances_release_1/grf6.txt",
    "instances_release_1/grf12.txt",
    "instances_release_1/grf19.txt",
]

INSTANCES_TO_IGNORE = INSTANCES_SOLVED_TO_OPTIMALITY


@cli.command(
    name="solver",
    help="Run the MIP model with a MIP solver",
    context_settings={"show_default": True},
)
@click.argument("instance", type=str, required=True)
@click.argument("solver", type=str, required=True)
@click.option("--time_limit_sec", type=int, default=60)
@click.option("--heuristic_effort", type=float, default=0.05)
@click.option("--presolve", type=int, default=-1)
@click.option("--mip_start", type=bool, is_flag=True, default=False, help="Provide a MIP start")
@click.option(
    "--relax_obj_modeling",
    type=bool,
    is_flag=True,
    default=False,
    help="Relax integrality of objective modeling",
)
@click.option(
    "--read_model", type=bool, is_flag=True, default=False, help="Try to read model from file"
)
def gb_main(
    instance: str,
    solver: str,
    time_limit_sec: int = 60,
    heuristic_effort: float = 0.05,
    presolve: int = -1,
    *,
    mip_start: bool = False,
    relax_obj_modeling: bool = False,
    read_model: bool = False,
) -> None:
    assert solver in ["gurobi"], "Solver must be either 'gurobi'"

    gurobi_main(
        instance,
        time_limit_sec=time_limit_sec,
        heuristic_effort=heuristic_effort,
        presolve=presolve,
        mip_start=mip_start,
        relax_obj_modeling=relax_obj_modeling,
        read_model=read_model,
    )


@cli.command(name="tsp_search", help="Run TSP search", context_settings={"show_default": True})
@click.argument("instance", type=str, required=True)
@click.option("--n_trials", type=int, default=60)
@click.option("--n_post_trials", type=int, default=10)
@click.option("--soft", type=bool, is_flag=True, default=False, help="Evaluate only the prior")
def tsp_main(
    instance: str, n_trials: int = 60, n_post_trials: int = 0, *, soft: bool = False
) -> None:
    if instance in INSTANCES_TO_IGNORE:
        click.echo(f"Instance {instance} has been solved to optimality. Skipping TSP search.")
        return
    heuristic_search(
        instance,
        search_type="randomized",
        n_trials=n_trials,
        n_post_trials=n_post_trials,
        soft=soft,
    )


@cli.command(
    name="swap_search", help="Run swap search", context_settings={"show_default": True}
)
@click.argument("instance", type=str, required=True)
@click.option("--n_post_trials", type=int, default=10)
@click.option("--k", type=int, default=2)
@click.option("--soft", type=bool, is_flag=True, default=False, help="Evaluate only the prior")
def swap_main_cli(
    instance: str, n_post_trials: int = 0, k: int = 2, *, soft: bool = False
) -> None:
    if instance in INSTANCES_TO_IGNORE:
        click.echo(f"Instance {instance} has been solved to optimality. Skipping.")
        return
    click.echo(f"Running swap search with k={k}")
    heuristic_search(
        instance, search_type=f"swap_{k}", n_trials=None, n_post_trials=n_post_trials, soft=soft
    )


@cli.command(
    name="delay_one",
    help="Run delay one neighborhood search",
    context_settings={"show_default": True},
)
@click.argument("instance", type=str, required=True)
@click.option("--n_post_trials", type=int, default=10)
@click.option("--soft", type=bool, is_flag=True, default=False, help="Evaluate only the prior")
def delay_one_cli(instance: str, n_post_trials: int = 0, *, soft: bool = False) -> None:
    if instance in INSTANCES_TO_IGNORE:
        click.echo(f"Instance {instance} has been solved to optimality. Skipping.")
        return
    click.echo("Running delay one neighborhood search")
    heuristic_search(
        instance, search_type="delay_1", n_trials=None, n_post_trials=n_post_trials, soft=soft
    )


@cli.command(name="grasp", help="Run GRASP", context_settings={"show_default": True})
@click.argument("instance", type=str, required=True)
@click.option("--n_trials", type=int, default=50)
def run_grasp(instance: str, n_trials: int = 50) -> None:
    if instance in INSTANCES_TO_IGNORE:
        click.echo(f"Instance {instance} has been solved to optimality. Skipping.")
        return
    click.echo("Running GRASP")
    heuristic_search(instance, search_type="grasp", n_trials=n_trials, n_post_trials=0)


if __name__ == "__main__":
    cli()
