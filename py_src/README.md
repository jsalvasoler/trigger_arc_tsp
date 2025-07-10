# Trigger Arc TSP

This is the code for the Trigger Arc TSP competition hosted by the [Metaheuristics Summer School](https://fourclicks.eu/fck/mess2024/frontend/#/home/dashboard) organizers.

In this [blog post](https://jsalvasoler.vercel.app/i-took-on-the-mess-2024-computational-optimization-challenge#1614370afad080cd977ddf6f8abf8ba0) I explain the approach I took to solve the problem.

-----

## Repo structure

- `instances/`: Contains the instances for the competition. They are not pushed, but they can be downloaded from the competition website.
- `solutions/`: A file for each instance with the list of solutions found during the competition.
- `py_src/trigger_arc_tsp/`: Contains the code for the Trigger Arc TSP competition.
- `tests/`: Contains the unit tests for the code.

## Getting started

This project uses Hatch for dependency management and packaging. Install hatch with the recommended `pipx install hatch`, or simply `pip install hatch`.
Hatch will sync the dependencies with the `pyproject.toml` file when you run the code, so we are all when it comes to dependencies.

Run the code with `hatch run trigger_arc_tsp <arguments>`. Run `hatch run trigger_arc_tsp --help` to see the arguments.

Feel free to check the `py_src/trigger_arc_tsp/__main__.py` file to see all the implemented algorithms.


An example command to run GRASP is:

```bash
export PYTHONPATH=$PYTHONPATH:py_src
uv run python -m trigger_arc_tsp grasp instances_release_2/grf134.txt --n_trials 500
```

An example command to solve an instance with the Gurobi solver is:

```bash
uv run python -m trigger_arc_tsp solver instances_release_2/grf103.txt gurobi \
    --time_limit_sec 300 \
    --heuristic_effort 1 \
    --mip_start \
    --presolve
```