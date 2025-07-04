#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Default build type is Release
BUILD_TYPE=${1:-Release}

# Clean build directory
rm -rf "${SCRIPT_DIR}/build"
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

# Run CMake with Boost paths specified
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_GUROBI=OFF \
      -DBoost_INCLUDE_DIR=/usr/include \
      -DBoost_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu \
      "${SCRIPT_DIR}"

# Compile using all available CPU cores
make -j$(nproc)

echo "✅ Build completed successfully with build type: ${BUILD_TYPE}!"
