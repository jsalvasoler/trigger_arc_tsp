echo "Building..."
./cpp_src/build.sh

echo "Solving..."
outputDir="output/gurobi-solve"
mkdir -p $outputDir


to_skip=(
    # "grf10.txt"
)

# iterate over all files in instances/instances_release_1
for file in instances/instances_release_1/*; do
    for skip_file in "${to_skip[@]}"; do
        if [[ "$file" == *"$skip_file" ]]; then
            continue 2
        fi
    done

    echo "Solving $file"
    ./cpp_src/build/trigger_arc_tsp --instance-file $file --time-limit $((15*60)) --output-dir $outputDir --logs
done
