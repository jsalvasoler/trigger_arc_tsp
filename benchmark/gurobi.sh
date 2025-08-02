#!/bin/bash
cpp_src/build.sh

# Check if instance_set is provided as first argument
if [ $# -eq 0 ]; then
    echo "Error: instance_set must be provided as first argument"
    echo "Usage: $0 <instance_set>"
    exit 1
fi

instance_set=$1

timestamp=$(date +%Y%m%d_%H%M%S)

timestamp=20250725_154430
output_dir=output/gurobi/$instance_set/$timestamp

# append output_dir to myrun.txt
echo $output_dir >> myrun.txt

echo "=== Starting Gurobi Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    echo "Running Gurobi for instance $instance"
    cpp_src/build/trigger_arc_tsp --instance-file $instance --method gurobi --time-limit 60 --logs --output-dir $output_dir/$(basename $instance)
done

echo "--------------------------------"
echo "Benchmarking finished"
echo "Timestamp: $timestamp"
echo "--------------------------------"