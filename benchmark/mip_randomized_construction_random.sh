#!/bin/bash
cpp_src/build.sh

n_trials_rg=5
alpha_grid=(0 0.1 1 10) # edge = edge + alpha * random_uniform(-1, 1)
beta_grid=(1.1 1.5 2 5) # edge = edge * beta * random_uniform(0, 1)
timestamp=$(date +%Y%m%d_%H%M%S)
time_limit_mip=2    # time limit for MIP TSP model
instance_set=instances_release_2

which=$1 # alpha or beta, read from script arguments
# raise error if which is not alpha or beta
if [ "$which" != "alpha" ] && [ "$which" != "beta" ]; then
    echo "Error: which must be alpha or beta"
    exit 1
fi

output_dir=output/mip_randomized_construction_random_${which}/$instance_set/$timestamp

if [ "$which" == "alpha" ]; then
    # alpha-randomization

    echo "=== Starting MIP Randomized Construction with Alpha-Randomization Benchmark ==="
    for instance in $(ls instances/$instance_set/*.txt); do
        for alpha in "${alpha_grid[@]}"; do
            for trial in $(seq 1 $n_trials_rg); do
                echo "Running SR trial $trial for instance $instance with alpha $alpha"
                cpp_src/build/trigger_arc_tsp --instance-file $instance --method mip_randomized_construction_random --alpha $alpha --time-limit $time_limit_mip --logs --output-dir $output_dir/$(basename $instance)
            done
        done
    done
fi

if [ "$which" == "beta" ]; then
    # beta-randomization

    echo "=== Starting MIP Randomized Construction with Beta-Randomization Benchmark ==="
    for instance in $(ls instances/$instance_set/*.txt); do
        for beta in "${beta_grid[@]}"; do
            for trial in $(seq 1 $n_trials_rg); do
                echo "Running SR trial $trial for instance $instance with beta $beta"
                cpp_src/build/trigger_arc_tsp --instance-file $instance --method mip_randomized_construction_random --beta $beta --time-limit $time_limit_mip --logs --output-dir $output_dir/$(basename $instance)
            done
        done
    done
fi
