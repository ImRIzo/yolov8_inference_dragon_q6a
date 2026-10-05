# YOLOv8 Video Inference on CPU (onnxruntime)

Real-time YOLOv8 object detection on video files using the CPU through
`onnxruntime`.  Python port of the Radxa Q6A NPU C++ tool — same pipeline,
same CLI, same reports — just running on the CPU instead of the QNN HTP.
Displays a live OpenCV window with bounding boxes, FPS counter
(smoothed + min/max), per-frame pipeline timing, and saves a benchmark
report on exit.

---

## Requirements

- Any aarch64 / x86_64 Linux (tested on Radxa Dragon Q6A, Ubuntu 24.04)
- Python 3.10+ with the `py312` pyenv environment (preinstalled with all
  dependencies), or any venv with the packages below
- A YOLOv8 `.onnx` model (e.g. `best.onnx`, Ultralytics export)

```bash
pip install -r requirements.txt
```

If the system Python is externally managed (PEP 668), use a venv or the
preconfigured pyenv environment — `run.sh` does this automatically.

---

## Project structure

```
inference_cpu/
├── yolov8_cpu.py               # Model load, video loop, timing, display, report
│                               #   preprocess  — letterbox (RGB, /255, NCHW, pad 114)
│                               #   infer       — onnxruntime CPUExecutionProvider
│                               #   postprocess — parse, NMS, draw boxes, debug overlay
│                               #   evaluate    — mAP evaluation (--map) on image datasets
├── run.sh                      # Launcher (activates the py312 pyenv environment)
├── requirements.txt            # pip dependencies (onnxruntime, numpy, opencv-python)
├── best.onnx                   # Ganoderma model — YOLOv8n, 1 class
│                               # (input [1,3,640,640], output [1,5,8400], float32)
├── test_video.mp4              # Demo video (same as the NPU repo)
├── data/                       # Image datasets for --map (YOLO label format)
│   ├── train/  test/  valid/
│   └── .../images  .../labels
├── report.txt                  # Video benchmark, generated on exit
├── eval_report_<dataset>.txt   # mAP reports (one per dataset)
├── result_<image>.jpg          # Annotated output of --image mode
└── README.md
```

---

## Setup

No build step — plain Python.  `run.sh` selects the `py312` pyenv
environment that already has `onnxruntime`, `opencv-python` and `numpy`
installed:

```bash
chmod +x run.sh
./run.sh --help
```

Or run directly with that interpreter:

```bash
PYENV_VERSION=py312 python3 yolov8_cpu.py --help
```

---

## Run

### Video file with display

```bash
./run.sh --video test_video.mp4
```

### Headless benchmark (no window, faster)

```bash
./run.sh --video test_video.mp4 --no-show
```

### RTSP camera stream

```bash
./run.sh --rtsp rtsp://192.168.1.100:8554/stream
```

### Single image

```bash
./run.sh --image photo.jpg
```

Detections are printed to the console and the annotated image is saved as
`result_photo.jpg`.

### mAP evaluation on image data

Runs the same CPU pipeline on every image of a dataset and computes
COCO/Ultralytics-style metrics against the YOLO-format labels.  One
dataset per run — run all three splits like this:

```bash
./run.sh --map data/train --names Ganoderma
./run.sh --map data/valid --names Ganoderma
./run.sh --map data/test --names Ganoderma --conf 0.001 --iou 0.7
```

(with `data/` being the ganoderma dataset from `Ganoderma Research/
genoderma_train/data/`, or a symlink to it.)

`--map` without a dataset defaults to `data/valid`.  The dataset folder must
contain `images/` (jpg/png/bmp/webp) and `labels/` (YOLO txt: one line per
box, `class cx cy w h`, normalized).  Evaluation defaults to `conf 0.001` /
`iou 0.7` / `max_det 300` (Ultralytics val.py defaults) unless `--conf` /
`--iou` are given.

Metrics (same conventions as Ultralytics):

- **mAP@0.5** — AP at IoU 0.5
- **mAP@0.5:0.95** — mean AP over 10 IoU thresholds (0.50…0.95, step 0.05)
- **Precision / Recall / F1** — at the max-F1 operating point
- AP uses 101-point interpolated precision-recall curves

The full report (per-IoU-threshold AP table, per-class table, pipeline
timing) is printed and saved to `eval_report_<dataset>.txt` next to the
script (e.g. `eval_report_valid.txt`).

The model's input size, output layout and class count are read from the
ONNX metadata automatically, so each model's own geometry is applied —
nothing to configure.  Only the class *names* may need `--names`: the
bundled model is 1-class (`Ganoderma`), so pass `--names Ganoderma` for
human-readable labels and eval tables (otherwise it shows `class_0`).

---

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--video PATH` | `./test_video.mp4` | Input video file (camera index like `0` also works) |
| `--rtsp URL` | — | RTSP stream URL (low-latency mode, auto-reconnect) |
| `--image PATH` | — | Run inference on a single image (saves `result_<name>.jpg`) |
| `--model PATH` | `./best.onnx` | ONNX model file |
| `--names FILE\|LIST` | COCO names if 80 classes, else `class_N` | Class names: a `.txt` file (one per line) or comma-separated list (e.g. `--names Ganoderma`) |
| `--conf FLOAT` | `0.25` | Confidence threshold |
| `--iou FLOAT` | `0.50` | NMS IoU threshold |
| `--map [DATADIR]` | `data/valid` | Run mAP evaluation on one image dataset (report saved as `eval_report_<dataset>.txt`) |
| `--classes N` | from ONNX | Number of model classes (auto-detected; can override with `--map`) |
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
FPS:  4.1  avg:  4.1  min:  2.6  max:  4.7  #795 det:0
pipe: 4.1 fps (243.9 ms)  pre:3.9  inf:239.6  post:0.4 ms
CPU: onnxruntime 1.26.0 | best | ESC/Q quit
```

- **FPS** — EMA-smoothed wall-clock FPS (dampened, low jitter)
- **avg** — cumulative average wall-clock FPS
- **min / max** — wall-clock FPS range (after 5-frame warmup)
- **pipe** — CPU pipeline FPS and ms breakdown (preprocess / inference / postprocess)
- Green text = > 25 FPS; yellow = slower
- Semi-transparent dark background behind text for readability on any scene

---

## Report

On exit, `report.txt` is saved next to the script:

```
============================================================
  YOLOv8 Video Inference Report (CPU)
============================================================
  Date:       Mon Oct  5 19:42:11 2026
  Video:      test_video.mp4
  Model:      /home/radxa/src/inference_cpu/best.onnx
  Backend:    onnxruntime 1.26.0 / CPUExecutionProvider
  Confidence: 0.25
  IoU:        0.50
------------------------------------------------------------
  Frames:     795
  Total time: 195.68 s
  Avg FPS:    4.1
  Min FPS:    2.6
  Max FPS:    4.7
------------------------------------------------------------
  Avg preprocess:   3.88 ms
  Avg inference:    239.62 ms
  Avg postprocess:  0.43 ms
  Avg total:        243.93 ms
============================================================
```

All FPS values (avg/min/max) measure the same wall-clock metric including
display overhead, so they are directly comparable.

---

## Evaluation report

`--map` writes `eval_report_<dataset>.txt` next to the script:

```
============================================================
  YOLOv8 Evaluation Report (COCO-style mAP) - CPU
============================================================
  Date:            2026-10-05 19:45:02
  Dataset:         data/valid
  Model:           /home/radxa/src/inference_cpu/best.onnx
  Images:          104
  GT boxes:        126
  Predictions:     1259
  Conf: 0.001   NMS IoU: 0.70   Classes: 1   Max det: 300
------------------------------------------------------------
  mAP@0.5          0.8188
  mAP@0.5:0.95     0.5271
  Precision        0.6918
  Recall           0.8016
  F1-score         0.7426
  Best conf        0.509   (operating point of P/R/F1)
------------------------------------------------------------
  AP per IoU threshold:
    IoU      AP
    0.50    0.8188
    0.55    0.8062
    0.60    0.7794
    0.65    0.7491
    0.70    0.7303
    0.75    0.6208
    0.80    0.4637
    0.85    0.2422
    0.90    0.0598
    0.95    0.0004
------------------------------------------------------------
  Per class:
    Class              Img    GT    AP50    AP50-95   Prec    Rec     F1
    Ganoderma          63    126   0.8188   0.5271   0.6918  0.8016  0.7426
------------------------------------------------------------
  Timing:
    Total:           25.9 s   (4.0 images/s)
    Avg preprocess:  2.4 ms
    Avg inference:   240.6 ms
    Avg postprocess: 0.1 ms
    Avg total:       243.1 ms
============================================================
```

### Measured results — CPU float32 vs NPU int8

Same `valid` split (104 images, 126 GT boxes), conf 0.001 / NMS IoU 0.7 /
max det 300:

| Backend | Weights | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| CPU (this repo) | float32 `.onnx` | **0.8188** | **0.5271** | **0.6918** | **0.8016** | **0.7426** |
| QNN HTP (NPU repo) | int8 `.bin` | 0.6980 | 0.3488 | 0.5652 | 0.7222 | 0.6341 |

The CPU runs the **float32** weights, the NPU repo runs the **int8**-
quantized weights.  Both use the same preprocessing, NMS and metrics, so
the numbers are directly comparable — the CPU scores higher because there
is no quantization rounding cost.  (The two models are also different
training checkpoints, so treat the gap as an upper bound of the
quantization cost, not an exact measure of it.)

mAP matching is strict: a detection counts only if it overlaps its ground
truth with IoU > threshold, and duplicate predictions count as false
positives, so a few near-miss boxes measurably lower the score.

---

## Performance

Tested on Radxa Dragon Q6A (QCS6490) with 640x640 input video,
`best.onnx` (YOLOv8n-scale, 1 class, float32) on the 8 Kryo CPU cores:

| Metric | Time |
|---|---|
| Preprocess (letterbox + RGB + NCHW) | ~3.9 ms |
| **Inference (CPU, float32)** | **~240 ms** |
| Postprocess (parse + NMS + draw) | ~0.4 ms |
| **Pipeline total per frame** | **~244 ms** |
| **Wall-clock FPS (headless)** | **~4.1** |
| Pipeline FPS (headless, --no-show) | ~4.1 |

For reference, the same pipeline on the QCS6490 NPU runs at ~13 ms
inference / ~21 FPS — the CPU is ~19x slower per frame, as expected for
float32 Conv-heavy workloads on ARM cores.  The CPU version is the
accuracy reference and the NPU fallback; use the NPU repo for real-time.

Display overhead (`cv2.imshow` + `cv2.waitKey`) is significant on the Q6A.
Use `--no-show` for accurate pipeline benchmarking.

---

## Deploy

Copy the folder to any Linux machine with Python 3.10+:

```bash
scp -r inference_cpu radxa@target:/home/radxa/
```

On the target (install `pip install -r requirements.txt` — or use a pyenv
environment named `py312` for `run.sh`):

```bash
cd /home/radxa/inference_cpu
./run.sh --video my_video.mp4
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'onnxruntime'`**
Run through `./run.sh` (uses the `py312` pyenv environment), or install
into your venv: `pip install -r requirements.txt`.

**`externally-managed-environment` error from pip**
Debian/Ubuntu PEP 668 protection.  Use a venv
(`python3 -m venv .venv && .venv/bin/pip install ...`) or the existing
pyenv environment via `run.sh`.

**Wrong class names in the boxes**
The bundled model has one class but the name is not stored in the ONNX
metadata, so the default label is `class_0`.  Pass `--names Ganoderma`
(or a `names.txt` file, one name per line) to label it properly.

**Qt font warnings when the window opens**
Harmless — OpenCV's Qt backend prints font directory warnings on Wayland.
The window still renders; install `fonts-dejavu-core` to silence them.

**Flickering / double boxes**
Adjust IoU: `--iou 0.3` (more aggressive) or `--iou 0.7` (looser).

**Low FPS**
Expected on CPU — inference is ~247 ms/frame float32.  Options:
smaller model (`yolov8n` or a P6/quantized variant), lower video
resolution, or `--no-show` for benchmarking.  For real-time, use the
Q6A NPU version of this repo.

**`Could not determine class count` on a custom export**
The script handles the standard Ultralytics layouts
(`[1, 4+NC, A]`, `[1, A, 4+NC]`, and two-output boxes+scores).  If yours
is dynamic-shaped, export again with a fixed imgsz:
`model.export(format="onnx", imgsz=640, dynamic=False)`.
