#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create build directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/build"

# Navigate to build directory
cd "${SCRIPT_DIR}/build"

# Run CMake and make
cmake "${SCRIPT_DIR}"
make -j$(sysctl -n hw.ncpu)

# Run the tests
echo "Running tests..."
"${SCRIPT_DIR}/build/instance_tests" --gtest_brief=1
"${SCRIPT_DIR}/build/tsp_model_tests" --gtest_brief=1 
"${SCRIPT_DIR}/build/model_tests" --gtest_brief=1 # --gtest_filter=${TEST_TO_RUN}
"${SCRIPT_DIR}/build/randomized_greedy_tests" --gtest_brief=1
"${SCRIPT_DIR}/build/simple_randomized_construction_tests" --gtest_brief=1
"${SCRIPT_DIR}/build/mip_randomized_construction_tests" --gtest_brief=1
"${SCRIPT_DIR}/build/grasp_tests" --gtest_brief=1