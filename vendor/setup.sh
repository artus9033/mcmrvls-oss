#!/bin/bash

set -e

# Source venv
echo "Activating virtual environment..."
source ../venv/bin/activate

# Build AprilTag library
echo "Building AprilTag library..."
cd apriltag
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_INSTALL_PREFIX=$(realpath ../../../venv) ..
cmake --build . --target install
cd ../..

echo "Setup complete."
