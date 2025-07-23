#!/bin/bash
cpp_src/build.sh

n_trials_rg=10
alpha_grid=(0 0.05 0.1 0.5 1)
timestamp=$(date +%Y%m%d_%H%M%S)
instance_set=instances_release_2
output_dir=output/randomized_greedy/$instance_set/$timestamp

echo "=== Starting Randomized Greedy Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    for alpha in "${alpha_grid[@]}"; do
        for trial in $(seq 1 $n_trials_rg); do
            echo "Running RG trial $trial for instance $instance with alpha $alpha"
            cpp_src/build/trigger_arc_tsp --instance-file $instance --method randomized_greedy --alpha $alpha --logs --output-dir $output_dir/$(basename $instance)
        done
    done
done