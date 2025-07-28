#!/bin/bash
cpp_src/build.sh

n_trials_rg=10
timestamp=$(date +%Y%m%d_%H%M%S)
instance_set=instances_generic
output_dir=output/simple_randomized/$instance_set/$timestamp

echo "=== Starting Simple Construction Benchmark ==="
for instance in $(ls instances/$instance_set/*.txt); do
    for trial in $(seq 1 $n_trials_rg); do
        echo "Running RG trial $trial for instance $instance"
        cpp_src/build/trigger_arc_tsp --instance-file $instance --method simple_randomized --logs --output-dir $output_dir/$(basename $instance)
    done
done