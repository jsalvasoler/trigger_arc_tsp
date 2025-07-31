#!/bin/bash
cpp_src/build.sh

# Check if instance_set is provided as first argument
if [ $# -eq 0 ]; then
    echo "Error: instance_set must be provided as first argument"
    echo "Usage: $0 <instance_set>"
    exit 1
fi

instance_set=$1

n_trials_rg=10
alpha_grid=(0 0.025 0.05 0.075 0.1 0.25 0.5 1)
timestamp=$(date +%Y%m%d_%H%M%S)
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

echo "--------------------------------"
echo "Benchmarking finished"
echo "Timestamp: $timestamp"
echo "--------------------------------"