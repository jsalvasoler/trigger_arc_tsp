#!/bin/bash
cpp_src/build.sh

# Check if instance_set is provided as first argument
if [ $# -eq 0 ]; then
    echo "Error: instance_set must be provided as first argument"
    echo "Usage: $0 <instance_set>"
    exit 1
fi

instance_set=$1

n_trials_rg=5
alpha_grid=(0.05 0.1 1 3)
beta_grid=(0.05 0.1 1 3)
timestamp=$(date +%Y%m%d_%H%M%S)
time_limit_mip=2    # time limit for MIP TSP model

output_dir=output/mip_randomized_construction_bias/$instance_set/$timestamp

echo "=== Starting MIP Randomized Construction with Bias Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    for alpha in "${alpha_grid[@]}"; do
        for beta in "${beta_grid[@]}"; do
            for trial in $(seq 1 $n_trials_rg); do
                echo "Running SR trial $trial for instance $instance with alpha $alpha and beta $beta"
                cpp_src/build/trigger_arc_tsp --instance-file $instance --method mip_randomized_construction --alpha $alpha --beta $beta --time-limit $time_limit_mip --logs --output-dir $output_dir/$(basename $instance)
            done
        done
    done
done

echo "--------------------------------"
echo "Benchmarking finished"
echo "Timestamp: $timestamp"
echo "--------------------------------"