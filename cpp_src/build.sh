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
make -j$(nproc)

echo "Build completed successfully!" 