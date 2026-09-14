#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/server/venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR" >&2
    echo "Create one with: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -r $REPO_ROOT/server/requirements.txt" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Building AprilTag from $SCRIPT_DIR/apriltag into $VENV_DIR ..."
mkdir -p "$SCRIPT_DIR/apriltag/build"
cd "$SCRIPT_DIR/apriltag/build"
cmake -DPython3_EXECUTABLE="$(which python3)" -DCMAKE_INSTALL_PREFIX="$(realpath "$VENV_DIR")" ..
cmake --build . --target install -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "AprilTag setup complete."
