#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "=== Building ${PROJECT_DIR} ==="
echo "  Build dir:  ${BUILD_DIR}"
echo "  Build type: ${BUILD_TYPE}"
echo "  Jobs:       ${JOBS}"

# ---- Configure ----
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_FLAGS="-Wall -O2 -Wno-unused-result" \
    -DCMAKE_BUILD_RPATH='$ORIGIN/lib'

# ---- Build ----
cmake --build . -j"${JOBS}"

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/yolov8_video"
ls -lh "${BUILD_DIR}/yolov8_video"

# ---- Release bundle (optional) ----
if [[ "${1}" == "--release" ]] || [[ "${1}" == "-r" ]]; then
    echo ""
    echo "=== Assembling release bundle ==="
    cmake --build . --target release
    echo "Release: ${BUILD_DIR}/release"
    ls -lh "${BUILD_DIR}/release/"
fi
