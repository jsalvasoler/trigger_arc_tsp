# Trigger Arc TSP

This is the C++ implementation for the Trigger Arc TSP competition hosted by the [Metaheuristics Summer School](https://fourclicks.eu/fck/mess2024/frontend/#/home/dashboard) organizers.

In this [blog post](https://jsalvasoler.vercel.app/i-took-on-the-mess-2024-computational-optimization-challenge#1614370afad080cd977ddf6f8abf8ba0) we explain the approach we took to solve the problem.

This implementation uses C++20.

## Authors

- **Joan Salvà Soler** - [Website](https://jsalvasoler.vercel.app) - [Email](mailto:jsalvasoler@hotmail.com)
- **Gregoire deLmabertye** - [Website](https://gdelambertye.vercel.app/) - [Email](mailto:gregoire.delambertye@gmail.com)

-----

## Repo structure

- `instances/`: Contains the instances for the competition. They are not pushed, but they can be downloaded from the competition website.
- `solutions/`: A file for each instance with the list of solutions found.
- `src/`: Contains the C++ source code.
- `report/`: Contains the LaTeX source code for the report.
- `output/`: Contains the outputs of the experiments.
- `notebooks/`: Contains the notebooks used to generate the figures for the report and for the dataset generation and exploration. To run them, you can use the python environment defined in `pyproject.toml`.
- `benchmark/`: Contains scripts to reproduce experimental results.

## Getting started

### Dependencies

This project requires:
- A C++ compiler that supports C++20 (e.g., GCC 10+, Clang 11+).
- CMake (version 3.10 to 3.27).
- Boost libraries.
- Gurobi solver.

The paths for Boost and Gurobi might need to be adjusted in `src/CMakeLists.txt` depending on your system.

### Scripts

The `src` directory contains several useful scripts:

- `build.sh`: Compiles the project. Run this from the `src` directory.
- `test.sh`: Runs the tests. Run this from the `src` directory after building the project.
- `fmt.sh`: Formats the C++ code using `clang-format`.
- `benchmark.sh`: A script to run benchmarks. You might need to adapt it to your needs.

### Building and Running

1.  Place the problem instances in the `instances/` directory at the root of the repository.
2.  Navigate to the `src` directory: `cd src`
3.  Run the build script: `./build.sh`
4.  (Optional) Run the test script: `./test.sh`
5.  The executable `trigger_arc_tsp` will be created in `src/build/`.

You can then run the solver from `src/build` directory, for example:

```bash
./src/build/trigger_arc_tsp --method grasp --instance-file ../../instances/instances_release_1/grf1.txt --n-trials 10 --local-searches TwoOpt SwapTwo Relocate --constructive-heuristic SimpleRandomized --logs --alpha 0.1 --beta 3.0 --save-solutions
```

Run `./src/build/trigger_arc_tsp --help` to see all available options.

### GRASP Implementation

Our solver includes an implementation of the Greedy Randomized Adaptive Search Procedure (GRASP) metaheuristic. GRASP is an iterative process where each iteration consists of two phases: a construction phase and a local search phase.

#### Construction Phase

In the construction phase, a feasible solution is built using a randomized approach. We support the following constructive heuristics:

-   `RandomizedGreedy`: A simple randomized greedy construction.
-   `MIPRandomizedGreedyBias`: A more advanced construction that uses a MIP model on a restricted set of edges with a bias towards promising edges.
-   `MIPRandomizedGreedyRandom`: Similar to the above, but with a more random selection of edges.
-   `SimpleRandomized`: A very simple construction heuristic that randomly selects the next node from the neighbors of the current node. It's very fast but not very effective.

You can specify the constructive heuristic using the `--constructive-heuristic` flag.

#### Local Search Phase

Once a solution is constructed, a local search is applied to improve it until a local optimum is found. We support the following local search neighborhoods:

-   `TwoOpt`: The classic 2-opt neighborhood, which involves reversing a segment of the tour.
-   `SwapTwo`: Swaps two nodes in the tour.
-   `Relocate`: Moves a node to a different position in the tour.

You can specify one or more local searches to be applied using the `--local-searches` flag. They will be applied in the order they are provided. 

## Reproducing Results

To reproduce the results from our experiments, we provide a set of benchmark scripts in the `benchmark/` directory. These scripts automate the process of running the solver with different methods and configurations.

### Available Benchmark Scripts

- `grasp.sh`: Runs the solver using the GRASP metaheuristic.
- `gurobi.sh`: Runs the solver using the Gurobi MIP solver.
- `randomized_greedy.sh`: Runs the solver with the randomized greedy heuristic.
- `simple_randomized.sh`: Runs the solver with the simple randomized construction heuristic.
- `mip_randomized_construction_bias.sh`: Runs the solver with the MIP randomized construction (biased version).
- `mip_randomized_construction_random.sh`: Runs the solver with the MIP randomized construction (random version).

### How to Use

1. Make sure you have built the project as described above.
2. Place your problem instances in the `instances/` directory.
3. From the project root, run any of the scripts in the `benchmark/` directory. For example:
   ```bash
   ./benchmark/grasp.sh
   ```
   You may need to make the scripts executable:
   ```bash
   chmod +x benchmark/*.sh
   ```

Each script is pre-configured with recommended parameters, but you can edit them to suit your needs.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details. 
