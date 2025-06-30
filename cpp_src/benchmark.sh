#!/bin/bash

# Benchmark script
./cpp_src/build.sh

### Benchmark randomized greedy
alpha=0.5
n_trials=10

for instance in $(ls instances/instances_release_1/*.txt); do
    for trial in $(seq 1 $n_trials); do
        echo "Running trial $trial for instance $instance with alpha $alpha"
        cpp_src/build/trigger_arc_tsp --instance-file $instance --method randomized_greedy --alpha $alpha --time-limit 60 --heuristic-effort 0.05 --presolve -1 --mip-start --logs --output-dir output/randomized_greedy/$(basename $instance)_alpha_$alpha
    done
done