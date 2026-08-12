#!/bin/bash
set -e
BUILD_DIR="$1"
SRC_DIR="$2"
RELEASE="$BUILD_DIR/release"
ONNXQNN="$HOME/.pyenv/versions/3.12.13/envs/py312/lib/python3.12/site-packages/onnxruntime_qnn"
SYSLIB="/usr/lib/aarch64-linux-gnu"
MODEL_DIR="$SRC_DIR/../quantized_compiled_model"

echo "=== Assembling $RELEASE ==="
rm -rf "$RELEASE"
mkdir -p "$RELEASE/lib" "$RELEASE/model"

# Binary
cp "$BUILD_DIR/yolov8_video" "$RELEASE/"

# Video file
if [ -f "$SRC_DIR/test_video.mp4" ]; then
    cp "$SRC_DIR/test_video.mp4" "$RELEASE/"
fi

# Model
cp "$MODEL_DIR"/*.bin "$MODEL_DIR"/*_config.json "$RELEASE/model/" 2>/dev/null || true

# QNN libs
for lib in libQnnHtp.so libQnnHtpV68Stub.so libQnnHtpV68Skel.so \
           libQnnSystem.so libQnnCpu.so libQnnHtpNetRunExtensions.so \
           libQnnHtpPrepare.so libQnnIr.so libQnnSaver.so; do
    cp "$ONNXQNN/$lib" "$RELEASE/lib/"
done

# OpenCV libs — copy real .so.4.6.0 and create .so symlink
for name in core imgproc imgcodecs dnn highgui videoio; do
    real=$(readlink -f "$SYSLIB/libopencv_${name}.so")
    cp "$real" "$RELEASE/lib/"
    ln -sf "$(basename "$real")" "$RELEASE/lib/libopencv_${name}.so"
done

# run.sh
cat > "$RELEASE/run.sh" << 'RUNEOF'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export ADSP_LIBRARY_PATH="${DIR}/lib"
exec "${DIR}/yolov8_video" "$@"
RUNEOF
chmod +x "$RELEASE/run.sh"

echo "Release: $RELEASE"
ls -lh "$RELEASE/"
echo "Done."
