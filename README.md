# YOLOv8 Video Inference on Radxa Dragon Q6A NPU

Real-time YOLOv8 object detection on video files using the Qualcomm QCS6490
Hexagon V68 HTP (NPU).  Displays a live OpenCV window with bounding boxes,
FPS counter (smoothed + min/max), per-frame pipeline timing, and saves a
benchmark report on exit.

---

## Requirements

- Radxa Dragon Q6A running Ubuntu 24.04
- QNN SDK headers at `/home/radxa/qairt/include/`
- Compiled `.bin` model + `_config.json` in `../quantized_compiled_model/`

```bash
sudo apt install g++ cmake libopencv-dev
pip install onnxruntime-qnn
```

---

## Project structure

```
cpp_test_video/
├── main.cpp                   # Video loop, timing, display, report
├── preprocess.hpp / .cpp       # Letterbox preprocessor (image + cv::Mat)
├── inference.hpp / .cpp        # QNN HTP runtime
├── postprocess.hpp / .cpp      # Parse, NMS, draw boxes, debug overlay
├── CMakeLists.txt
├── build.sh                    # Build script (optional)
├── mkrelease.sh                # Release bundler (libs + model + binary)
├── README.md
├── test_video.mp4
└── build/                      # CMake build output + release bundle
    ├── yolov8_video            # Compiled binary
    └── release/                # Self-contained deployment bundle
        ├── yolov8_video        # Binary (RPATH=$ORIGIN/lib:)
        ├── run.sh              # Launcher
        ├── test_video.mp4
        ├── lib/                # QNN + OpenCV .so files
        ├── model/              # .bin + _config.json
        └── report.txt          # Generated on exit
```

---

## Build

### Quick build

```bash
cd cpp_test_video
bash build.sh
```

### Manual build

```bash
cd cpp_test_video
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Build + release bundle (for deployment)

```bash
bash build.sh --release
# or:
cmake --build build --target release
```

The release bundle at `build/release/` is self-contained — copy it to any
Radxa Q6A device and run `./run.sh`.

---

## Run

### From build directory (source tree)

```bash
cd build
./yolov8_video --video ../test_video.mp4
```

### From release bundle

```bash
cd build/release
./run.sh --video test_video.mp4
```

### Headless benchmark (no window, faster)

```bash
./yolov8_video --video test_video.mp4 --no-show
```

### RTSP camera stream

```bash
./yolov8_video --rtsp rtsp://192.168.1.100:8554/stream
```

---

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--video PATH` | `./test_video.mp4` | Input video file |
| `--rtsp URL` | — | RTSP stream URL (low-latency mode) |
| `--lib DIR` | `./lib` | Directory containing QNN .so files |
| `--conf FLOAT` | `0.25` | Confidence threshold |
| `--iou FLOAT` | `0.50` | NMS IoU threshold |
| `--no-show` | off | Disable display window (benchmark mode) |
| `--help` | | Show usage |

### Controls

| Key | Action |
|---|---|
| `ESC` | Quit and save report |
| `q` / `Q` | Quit and save report |

---

## On-screen overlay

```
FPS: 20.8  avg: 20.8  min: 18.2  max: 28.7  #795 det:13
pipe: 48.3 fps (20.7 ms)  pre:7.1  inf:12.6  post:1.1 ms
NPU: QCS6490 HTP | YOLOv8n  |  ESC/Q quit
```

- **FPS** — EMA-smoothed wall-clock FPS (dampened, low jitter)
- **avg** — cumulative average wall-clock FPS
- **min / max** — wall-clock FPS range (after 5-frame warmup)
- **pipe** — NPU pipeline FPS and ms breakdown (preprocess / inference / postprocess)
- Green text = > 25 FPS; yellow = slower
- Semi-transparent dark background behind text for readability on any scene

---

## Report

On exit, `report.txt` is saved next to the binary:

```
============================================================
  YOLOv8 Video Inference Report
============================================================
  Date:       Sat May 31 22:32:00 2026
  Video:      test_video.mp4
  Model:      .../model/job_xxx.bin
  Confidence: 0.25
  IoU:        0.50
------------------------------------------------------------
  Frames:     795
  Total time: 38.2 s
  Avg FPS:    20.8
  Min FPS:    18.2
  Max FPS:    28.7
------------------------------------------------------------
  Avg preprocess:   7.07 ms
  Avg inference:    12.59 ms
  Avg postprocess:  1.06 ms
  Avg total:        20.72 ms
============================================================
```

All FPS values (avg/min/max) measure the same wall-clock metric including
display overhead, so they are directly comparable.

---

## Performance

Tested on Radxa Dragon Q6A (QCS6490) with 640x640 input video:

| Metric | Time |
|---|---|
| Preprocess (letterbox + RGB + NCHW) | ~7 ms |
| **Inference (QNN HTP)** | **~13 ms** |
| Postprocess (parse + NMS + draw) | ~1 ms |
| **Pipeline total per frame** | **~21 ms** |
| **Wall-clock FPS (with display)** | **~21** |
| Pipeline FPS (headless, --no-show) | ~35 |

Display overhead (`cv::imshow` + `cv::waitKey`) is significant on the Q6A.
Use `--no-show` for accurate pipeline benchmarking.

---

## Deploy

Copy `build/release/` to any Radxa Q6A device:

```bash
scp -r build/release radxa@target:/home/radxa/yolov8_video/
```

On the target:

```bash
cd /home/radxa/yolov8_video
./run.sh --video my_video.mp4
```

---

## Troubleshooting

**`dlopen HTP: libQnnHtp.so: cannot open`**
Make sure `lib/` is next to the binary. The RPATH `$ORIGIN/lib:` should
find it automatically, or use `run.sh` which sets `ADSP_LIBRARY_PATH`.

**`contextCreateFromBinary` fails**
SDK version mismatch. The bundled libs must be from onnxruntime-qnn
(2.45.40), not the system QAIRT SDK (2.40.0).

**Flickering / double boxes**
Adjust IoU: `--iou 0.3` (more aggressive) or `--iou 0.7` (looser).

**Low FPS with display window**
Expected on the Q6A — OpenCV window rendering costs ~10-15 ms/frame.
Use `--no-show` for benchmarking or reduce window resolution in `main.cpp`
(`WINDOW_W` / `WINDOW_H` constants).
