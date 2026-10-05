#!/usr/bin/env bash
# Launcher: uses the pyenv py312 environment that has onnxruntime/opencv/numpy.
cd "$(dirname "$0")"
export PYENV_VERSION=py312
exec python3 yolov8_cpu.py "$@"
