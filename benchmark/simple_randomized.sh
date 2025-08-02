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
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir=output/simple_randomized/$instance_set/$timestamp

echo "=== Starting Simple Construction Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    for trial in $(seq 1 $n_trials_rg); do
        echo "Running RG trial $trial for instance $instance"
        cpp_src/build/trigger_arc_tsp --instance-file $instance --method simple_randomized --logs --output-dir $output_dir/$(basename $instance)
    done
done

echo "--------------------------------"
echo "Benchmarking finished"
echo "Timestamp: $timestamp"
echo "--------------------------------"