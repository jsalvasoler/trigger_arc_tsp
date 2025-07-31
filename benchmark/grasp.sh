#!/bin/bash
cpp_src/build.sh


# Check if instance_set is provided as first argument
if [ $# -eq 0 ]; then
    echo "Error: instance_set must be provided as first argument"
    echo "Usage: $0 <instance_set> [local_searches]"
    exit 1
fi

instance_set=$1
local_searches=${2:-"TwoOpt SwapTwo Relocate"}

n_trials_grasp=10
alpha=0.1
beta=3.0
timestamp=$(date +%Y%m%d_%H%M%S)
time_limit_grasp=60    # time limit for GRASP

output_dir=output/grasp/$instance_set/$timestamp


echo "=== Starting GRASP Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    echo "Running GRASP for instance $instance with alpha $alpha and beta $beta"
    cpp_src/build/trigger_arc_tsp --instance-file $instance --method grasp --alpha $alpha --beta $beta --time-limit $time_limit_grasp --n-trials $n_trials_grasp --constructive-heuristic MIPRandomizedGreedyBias --local-searches $local_searches --logs --output-dir $output_dir/$(basename $instance)
done


echo "--------------------------------"
echo "Benchmarking finished"
echo "Timestamp: $timestamp"
echo "--------------------------------"