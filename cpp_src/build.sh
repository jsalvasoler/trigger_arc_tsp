#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for 'clean' argument
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${SCRIPT_DIR}/build"
    shift # Remove 'clean' from arguments
fi

# Default build type is Release
BUILD_TYPE=${1:-Release}

# Create build directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/build"

# Navigate to build directory
cd "${SCRIPT_DIR}/build"

# Run CMake and make
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} "${SCRIPT_DIR}"
make -j$2

echo "Build completed successfully with build type: ${BUILD_TYPE}!" 