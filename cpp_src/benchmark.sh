#!/bin/bash

# Benchmark script
./cpp_src/build.sh


# define what to benchmark with a map
to_benchmark=(
    # "randomized_greedy"
    # "mip_randomized_construction"
    "grasp"
)

if [[ " ${to_benchmark[@]} " =~ " randomized_greedy " ]]; then
    ### Benchmark randomized greedy
    alpha=0.5
    n_trials_rg=10

    echo "=== Starting Randomized Greedy Benchmark ==="
    for instance in $(ls instances/instances_release_1/*.txt); do
        for trial in $(seq 1 $n_trials_rg); do
            echo "Running RG trial $trial for instance $instance with alpha $alpha"
            cpp_src/build/trigger_arc_tsp --instance-file $instance --method randomized_greedy --alpha $alpha --time-limit 60 --heuristic-effort 0.05 --presolve -1 --mip-start --logs --output-dir output/randomized_greedy/$(basename $instance)_alpha_$alpha
        done
    done
fi

if [[ " ${to_benchmark[@]} " =~ " mip_randomized_construction " ]]; then

    ### Benchmark MIP randomized construction
    n_trials_mip=20
    time_limit_mip=10

    echo "=== Starting MIP Randomized Construction Benchmark ==="

    # MIP randomized construction benchmark
    echo "--- Benchmarking MIP Randomized Construction ---"
    for instance in $(ls instances/instances_release_1/*.txt); do
        for trial in $(seq 1 $n_trials_mip); do
            echo "Running MIP-RC trial $trial for instance $instance"
            cpp_src/build/trigger_arc_tsp --instance-file $instance --method mip_randomized_construction --time-limit $time_limit_mip --logs --output-dir output/mip_randomized_construction/$(basename $instance)
        done
    done
fi

if [[ " ${to_benchmark[@]} " =~ " grasp " ]]; then

    ### Benchmark GRASP
    n_trials_grasp=1000
    time_limit_grasp=120
    alpha_grasp=0.3
    beta_grasp=0.5
    local_searches_grasp="TwoOpt"

    constructive_heuristics_grasp=(
        # "RandomizedGreedy"
        "MIPRandomizedGreedyBias"
        # "MIPRandomizedGreedyRandom"
    )

    echo "=== Starting GRASP Benchmark ==="

    for instance in $(ls instances/instances_release_1/*.txt); do
        if [[ ! $instance == *"grf13.txt" ]]; then
            echo "Skipping instance $instance as it does not end with grf1.txt"
            continue
        fi
        for heuristic in "${constructive_heuristics_grasp[@]}"; do
            echo "Running GRASP for instance $instance with heuristic $heuristic"
            cpp_src/build/trigger_arc_tsp \
                --instance-file $instance \
                --method grasp \
                --time-limit $time_limit_grasp \
                --n-trials $n_trials_grasp \
                --alpha $alpha_grasp \
                --beta $beta_grasp \
                --constructive-heuristic $heuristic \
                --local-searches $local_searches_grasp \
                --logs \
                --output-dir output/grasp/$(basename $instance)_heuristic_${heuristic}
        done
    done
fi


echo "=== Benchmark Complete ==="