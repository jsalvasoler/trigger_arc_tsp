# Trigger Arc TSP

This is the C++ implementation for the Trigger Arc TSP competition hosted by the [Metaheuristics Summer School](https://fourclicks.eu/fck/mess2024/frontend/#/home/dashboard) organizers.

In this [blog post](https://jsalvasoler.vercel.app/i-took-on-the-mess-2024-computational-optimization-challenge#1614370afad080cd977ddf6f8abf8ba0) we explain the approach we took to solve the problem.

This implementation uses C++20.

-----

## Repo structure

- `instances/`: Contains the instances for the competition. They are not pushed, but they can be downloaded from the competition website.
- `solutions/`: A file for each instance with the list of solutions found during the competition.
- `cpp_src/`: Contains the C++ source code.
- `py_src/`: Contains the legacy Python source code. See `py_src/README.md` for more information.

## Getting started

### Dependencies

This project requires:
- A C++ compiler that supports C++20 (e.g., GCC 10+, Clang 11+).
- CMake (version 3.10 to 3.27).
- Boost libraries.
- Gurobi solver.

The paths for Boost and Gurobi might need to be adjusted in `cpp_src/CMakeLists.txt` depending on your system.

### Scripts

The `cpp_src` directory contains several useful scripts:

- `build.sh`: Compiles the project. Run this from the `cpp_src` directory.
- `test.sh`: Runs the tests. Run this from the `cpp_src` directory after building the project.
- `fmt.sh`: Formats the C++ code using `clang-format`.
- `benchmark.sh`: A script to run benchmarks. You might need to adapt it to your needs.

### Building and Running

1.  Place the problem instances in the `instances/` directory at the root of the repository.
2.  Navigate to the `cpp_src` directory: `cd cpp_src`
3.  Run the build script: `./build.sh`
4.  The executable `trigger_arc_tsp` will be created in `cpp_src/build/bin/`.

You can then run the solver from `cpp_src/build` directory, for example:

```bash
./bin/trigger_arc_tsp <algorithm> <instance_path> [options]
``` 