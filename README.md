# YOLOv8 Video Inference on Radxa Dragon Q6A NPU

Real-time YOLOv8 object detection on video files using the Qualcomm QCS6490
Hexagon V68 HTP (NPU).  Displays a live OpenCV window with bounding boxes,
FPS counter (smoothed + min/max), per-frame pipeline timing, and saves a
benchmark report on exit.

![Screenshot](screenshot.png)
---

## Requirements

- Radxa Dragon Q6A running Ubuntu 24.04
- QNN SDK headers at `/home/radxa/qairt/include/`
- Compiled `.bin` model + `_config.json` in `../quantized_compiled_model/`

```bash
sudo apt install g++ cmake libopencv-dev
pip install onnxruntime-qnn
```

## Dataset

The image datasets (`data/train`, `data/valid`, `data/test`) are **not**
part of this repository.  Create them before running `--eval` / `--map`:

```
data/
└── <split>/
    ├── images/   image001.jpg …
    └── labels/   image001.txt …     # YOLO txt: "class cx cy w h" (normalized)
```

The compiled QNN model (`.bin` + `_config.json`) is expected in
`../quantized_compiled_model/` (or in `model/` next to the binary).

---

## Project structure

```
cpp_test_video/
├── main.cpp                   # Video loop, timing, display, report
├── preprocess.hpp / .cpp       # Letterbox preprocessor (image + cv::Mat)
├── inference.hpp / .cpp        # QNN HTP runtime
├── postprocess.hpp / .cpp      # Parse, NMS, draw boxes, debug overlay
├── evaluate.hpp / .cpp         # Static labeled-image accuracy evaluation
├── evaluate_metrics.py         # Ultralytics-exact mAP/P/R/F1 from predictions.csv
├── data/                       # Image datasets (YOLO label format)
│   ├── train/  test/  valid/
│   └── .../images  .../labels
├── CMakeLists.txt
├── build.sh                    # Build script (optional)
├── mkrelease.sh                # Release bundler (libs + model + binary)
├── README.md
├── test_video.mp4
└── build/                      # CMake build output + release bundle
    ├── yolov8_video            # Compiled binary
    └── release/                # Self-contained deployment bundle
        ├── yolov8_video        # Binary (RPATH=$ORIGIN/lib:)
        ├── evaluate_metrics.py # Ultralytics-exact metric evaluator
        ├── run.sh              # Launcher
        ├── test_video.mp4
        ├── data/               # Dataset (for --map / --eval)
        ├── lib/                # QNN + OpenCV .so files
        ├── model/              # .bin + _config.json
        ├── report.txt          # Video benchmark, generated on exit
        ├── predictions.csv     # Raw detections (--eval)
        ├── evaluation_results.csv  # Final metrics table (--eval)
        ├── eval_config.csv     # Evaluation parameters + audit info (--eval)
        ├── evaluated_images.txt    # Exact filenames evaluated (--eval)
        └── eval_report_*.txt   # Full evaluation reports
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

### Static labeled-image evaluation (object detection accuracy)

Runs the same QNN pipeline on every image of a labeled test set and
computes mAP / Precision / Recall / F1 against YOLO-format labels.

```bash
./yolov8_video --eval --images data/test/images --labels data/test/labels --no-show
# or just --images (labels dir defaults to ../labels):
./yolov8_video --eval --images data/test/images --no-show
# legacy dataset mode (images/ + labels/ inside one folder):
./yolov8_video --map data/test --classes 1 --conf 0.001 --iou 0.7
```

Dataset layout (images and labels are matched by filename stem):

```
test/
├── images/   image001.jpg  image002.jpg …
└── labels/   image001.txt  image002.txt …   # "class cx cy w h", normalized
```

An image may have no objects — the label file may be empty or absent;
both are handled as "no ground truth".  Missing/empty labels never crash
the evaluation.

#### Test-set evaluation — exact copy-paste command

This is the exact command used for the research-paper run on the 42-image
test split, with the `py312` environment (numpy/torch/ultralytics) active
for the metric evaluator:

```bash
cd build/release

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate py312

./yolov8_video --eval --images data/test/images --labels data/test/labels \
    --no-show --out ../results/test
```

It runs all 42 test images through the NPU (conf 0.001, NMS IoU 0.70,
max det 300, 5 warm-up images), saves `predictions.csv`,
`evaluation_results.csv`, `eval_config.csv`, `image_sizes.csv`,
`evaluated_images.txt` and `eval_report_test.txt` into `../results/test/`,
and prints the Ultralytics-exact metrics table.

To evaluate another split, replace `test` with `valid` or `train`
(both paths in `--images`, `--labels`, `--out` and the report name
follow automatically).

Evaluation settings default to **Ultralytics val.py values**:

- confidence retention threshold = `0.001`
- NMS IoU = `0.70`
- max detections per image = `300`

(The live video demo keeps its own `0.25` / `0.50` thresholds; the two
modes are fully independent.)

The pipeline per image is:

```
letterbox 640x640 → QNN/HTP inference → dequantization (scale+offset from
the .bin metadata) → YOLO decode (cxcywh→xyxy) → NMS → max-det cap →
reverse letterbox back to original image pixels
```

The reverse-letterbox uses the exact resize dimensions
(`(x - pad) * origW / resizedW`) so predicted boxes land in the original
image coordinate system with no sub-pixel error.  Boxes are not clipped to
image bounds, matching Ultralytics `scale_boxes` behaviour.

Every run writes, into the output directory (default: next to the binary,
override with `--out`):

| File | Content |
|---|---|
| `predictions.csv` | Raw detections after NMS + reverse letterbox: `image,class_id,confidence,x1,y1,x2,y2` (full float precision, nothing rounded) |
| `image_sizes.csv` | `image,width,height` for every evaluated image |
| `eval_config.csv` | Model path, backend, input resolution, thresholds, max_det, warm-up, dataset dirs, date |
| `evaluated_images.txt` | Exact filenames evaluated (verify all 42 test images were used) |
| `eval_report_<dataset>.txt` | Full report: metrics, per-IoU AP table, per-class table, timing, filename list |
| `evaluation_results.csv` | Final metrics table (written by evaluate_metrics.py) |

### Ultralytics-exact metrics (evaluate_metrics.py)

The C++ binary prints its own built-in COCO-style metrics, but for the
research comparison (FP32 PyTorch vs INT8 QNN HTP NPU) the canonical
numbers come from `evaluate_metrics.py`, which reproduces **Ultralytics
8.4.45 val.py exactly** (same `match_predictions`, `ap_per_class`,
`compute_ap`, `smooth`, and max-F1 operating point):

- mAP@0.5, mAP@0.5:0.95 from 101-point interpolated AP at IoU 0.50…0.95
- Precision / Recall / F1 at the smoothed max-F1 operating point
  (`i = argmax(smooth(mean F1 curve, 0.1))`), NOT at confidence 0.001
- all predictions ≥ 0.001 are used to build the precision-recall curves

It only needs `numpy` (no ultralytics/torch dependency).  `yolov8_video
--eval` runs it automatically when a `python3` with numpy is available
(set `QNN_EVAL_PYTHON=/path/to/python` or pass `--no-python-eval` to skip),
and it can always be rerun manually without re-running the NPU:

```bash
python3 evaluate_metrics.py \
    --predictions predictions.csv \
    --labels data/test/labels \
    --images data/test/images \
    --sizes image_sizes.csv \
    --config eval_config.csv \
    --output evaluation_results.csv
```

### Accuracy + latency timing in one run

While evaluating the images, per-stage timings are collected separately
for **preprocess**, **QNN/NPU inference**, and **postprocess** (plus the
total pipeline).  The first 5 images are warm-up: their predictions still
count for accuracy, but their timings are excluded from the statistics.
Reported per stage: mean, standard deviation (population), minimum and
maximum, plus total images, total processing duration and images/second.
"QNN inference latency" is reported separately from pre/postprocessing.

---

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--video PATH` | `./test_video.mp4` | Input video file |
| `--rtsp URL` | — | RTSP stream URL (low-latency mode) |
| `--lib DIR` | `./lib` | Directory containing QNN .so files |
| `--conf FLOAT` | `0.25` (video) / `0.001` (eval) | Confidence threshold |
| `--iou FLOAT` | `0.50` (video) / `0.70` (eval) | NMS IoU threshold |
| `--eval` | — | Static labeled-image evaluation mode |
| `--images DIR` | — | Test images directory (implies `--eval`) |
| `--labels DIR` | `../labels` next to images | YOLO txt labels directory |
| `--max-det N` | `300` | Max detections kept per image (evaluation) |
| `--warmup N` | `5` | Warm-up images excluded from timing stats |
| `--out DIR` | exe directory | Output dir for predictions.csv, reports, CSVs |
| `--no-python-eval` | off | Skip the automatic evaluate_metrics.py run |
| `--map [DATADIR]` | `data/valid` | Legacy: evaluate a dataset dir containing `images/` + `labels/` |
| `--classes N` | `1` | Number of model classes (evaluation) |
| `--no-show` | off | Disable display window (benchmark mode) |
| `--help` | | Show usage |

Environment variables:

| Variable | Description |
|---|---|
| `QNN_EVAL_PYTHON` | Python interpreter with numpy used for evaluate_metrics.py (default: `python3`) |

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

## Evaluation output

`--eval` writes everything into the output directory (default: next to the
binary, override with `--out`).  Console output ends with:

```
============================================================
QNN HTP NPU STATIC DETECTION EVALUATION
============================================================
Images              : 42
Ground-truth boxes  : 50
Predicted boxes     : 566

mAP@0.5              : 0.7004
mAP@0.5:0.95         : 0.4050
Precision            : 0.6364
Recall               : 0.7000
F1                   : 0.6667

Prediction threshold : 0.001
NMS IoU              : 0.70
Max detections       : 300
------------------------------------------------------------
--- Timing (5 warm-up excluded, 37 timed images) ---
                       mean       std       min       max
  preprocess  (ms)    4.429     0.776     3.512     6.379
  QNN inference(ms)   6.492     0.424     5.849     7.709
  postprocess (ms)    0.099     0.070     0.041     0.355
  total       (ms)   11.021     0.971     9.646    13.922
============================================================
```

followed by the canonical Ultralytics-exact metrics from
`evaluate_metrics.py` and `evaluation_results.csv`:

```
Runtime              mAP50    mAP50-95    Precision    Recall    F1
FP32 PyTorch         0.6833   0.3934      0.7063       0.6400    0.6715
INT8 QNN HTP NPU     <copy from evaluation_results.csv>
```

(With the bundled `yolov8_q6a.bin` on the 42-image test split the tool
reports mAP50 0.6962 / mAP50-95 0.3984 / P 0.6510 / R 0.7000 / F1 0.6746
via evaluate_metrics.py — but use the row printed for YOUR converted
model, not this one.)

(`evaluation_results.csv` has one header row + one data row:
`model/runtime,number_of_images,ground_truth_boxes,predicted_boxes,mAP50,
mAP50_95,precision,recall,F1,confidence_retention_threshold,nms_iou,max_det`.)

Notes on reading these numbers:

- Use the **evaluate_metrics.py** numbers for the paper — they reproduce
  Ultralytics val.py exactly, so the FP32-vs-INT8 comparison is
  apples-to-apples.  The C++ built-in table is a COCO-style cross-check.
- P/R/F1 are reported at the max-F1 operating point of the smoothed
  precision-recall curve, **not** at confidence 0.001.  The operating
  point (e.g. conf 0.473) is printed for audit.
- mAP matching is strict: a detection counts only if it overlaps its
  ground truth with IoU > threshold, and duplicate predictions count as
  false positives, so a few near-miss boxes measurably lower the score.
- The train → valid/test gap is the generalization gap — this is the
  number to watch, not the train score.

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
