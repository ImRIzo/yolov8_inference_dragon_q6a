#!/usr/bin/env python3
"""
quantize_onnx.py — Quantize a YOLOv8 ONNX model to INT8 for CPU onnxruntime.

Two modes:

  static   (default): full QDQ INT8 (quantize_static) using calibration
                      images.  Best accuracy/speed balance — recommended.
  dynamic  : weights-only INT8 (quantize_dynamic).  No calibration data
             needed, activations stay float32.

The quantized model keeps float32 input/output, so it is a drop-in
replacement for best.onnx in yolov8_cpu.py: the same letterbox
preprocessing, postprocessing, NMS and evaluation pipeline all work
unchanged — just pass --model best_int8.onnx.

Only "standard" ONNX quantization is used (onnxruntime.quantization), which
runs on the CPUExecutionProvider.  This is NOT the QNN/NPU int8 format
(that conversion lives in the C++ NPU repo).

Usage examples:
    python3 quantize_onnx.py
    python3 quantize_onnx.py --mode static --num-calib 128
    python3 quantize_onnx.py --mode dynamic --output best_int8_dyn.onnx
    python3 quantize_onnx.py --calib-dir ../data/train/images --num-calib 256 \
                             --calib-method Entropy --no-verify

Default calibration images: ../data/train/images (never calibrate on the
split you evaluate on).

Afterwards, evaluate the INT8 model with the existing tool (no extra
evaluator needed):
    ./run.sh --eval --model best_int8.onnx --images ../data/test/images \
        --labels ../data/test/labels --out ../results/cpu_int8/test
"""

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # keep cv2 quiet

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

DEFAULT_MODEL = "best.onnx"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PAD_COLOR = (114, 114, 114)

CALIB_METHODS = {"minmax": CalibrationMethod.MinMax,
                 "entropy": CalibrationMethod.Entropy}


# ──────────────────────────────────────────────────────────────────────────────
# Calibration data: letterbox (same as yolov8_cpu.py) -> float32 NCHW blobs
# ──────────────────────────────────────────────────────────────────────────────

def letterbox_bgr_to_blob(bgr, imgsz):
    """RGB letterbox + /255 + NCHW float32 — identical to the inference tool."""
    h, w = bgr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_left = (imgsz - nw) // 2
    pad_top = (imgsz - nh) // 2
    padded = cv2.copyMakeBorder(resized, pad_top, imgsz - nh - pad_top,
                                pad_left, imgsz - nw - pad_left,
                                cv2.BORDER_CONSTANT, value=PAD_COLOR)
    return padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def load_calibration_blobs(img_dir, num, imgsz):
    """Letterbox the first `num` images of a directory to float32 blobs."""
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(img_dir, "*" + ext)) + \
                 glob.glob(os.path.join(img_dir, "*" + ext.upper()))
    files = sorted(set(files))[:num]
    blobs = []
    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            print(f"  WARN: cannot read {f} — skipped", file=sys.stderr)
            continue
        blobs.append(letterbox_bgr_to_blob(bgr, imgsz))
    if not blobs:
        print(f"ERROR: no usable calibration images in {img_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  Calibration blobs: {len(blobs)} of shape {blobs[0].shape} (float32)")
    return blobs, files


class BlobDataReader(CalibrationDataReader):
    """Feeds preprocessed float32 blobs to quantize_static."""

    def __init__(self, input_name, arrays):
        self.input_name = input_name
        self.data = arrays
        self.start = 0
        self.end = len(arrays)
        self.iter = None

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.data[self.start:self.end])
        try:
            arr = next(self.iter)
        except StopIteration:
            return None
        return {self.input_name: arr}

    def rewind(self):
        self.iter = None

    def set_range(self, start, end):
        self.start = max(0, start)
        self.end = min(len(self.data), end)
        self.rewind()


# ──────────────────────────────────────────────────────────────────────────────
# Quantization
# ──────────────────────────────────────────────────────────────────────────────

def quantize(args, input_name):
    print("--- Quantizing ---")
    if args.mode == "dynamic":
        print(f"  Mode: dynamic (weights-only, no calibration data)")
        t0 = time.perf_counter()
        quantize_dynamic(
            model_input=args.model,
            model_output=args.output,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            extra_options={"EnableSubgraph": False},
        )
        print(f"  Done in {time.perf_counter() - t0:.1f} s")
        return

    print(f"  Mode: static QDQ ({args.act_type} activations / {args.wt_type} weights, "
          f"per-channel={args.per_channel})")
    print(f"  Calibration method: {args.calib_method}")
    blobs, _ = load_calibration_blobs(args.calib_dir, args.num_calib, args.imgsz)
    reader = BlobDataReader(input_name, blobs)
    # YOLOv8 models carry a decode head (DFL boxes + sigmoid scores) baked
    # into the graph.  Those elementwise/latent ops mix values of very
    # different dynamic ranges (boxes 0-640 vs scores 0-1), so their outputs
    # must stay float32 — a single uint8 scale on the final Concat output
    # rounds the whole score channel to zero.  Conv weights/activations are
    # still fully quantized; only the output edges of these op types are kept
    # in float.
    extra = {
        "EnableSubgraph": False,
        "OpTypesToExcludeOutputQuantization": [
            "Concat", "Sigmoid", "Slice", "Sub", "Div", "Add", "Mul",
            "Reshape", "Split", "Transpose", "Resize", "MaxPool", "Softmax",
        ],
    }
    t0 = time.perf_counter()
    quantize_static(
        model_input=args.model,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8 if args.act_type == "QUInt8" else QuantType.QInt8,
        weight_type=QuantType.QInt8 if args.wt_type == "QInt8" else QuantType.QUInt8,
        per_channel=args.per_channel,
        reduce_range=False,
        calibrate_method=CALIB_METHODS[args.calib_method],
        calibration_providers=["CPUExecutionProvider"],
        extra_options=extra,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f} s")


# ──────────────────────────────────────────────────────────────────────────────
# Verification: run FP32 and INT8 models on the same images, compare outputs
# ──────────────────────────────────────────────────────────────────────────────

def make_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, os.cpu_count() or 4)
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def verify(args, input_name):
    print("--- Verifying (INT8 vs FP32) ---")
    if args.verify <= 0 or not args.calib_dir:
        print("  Skipped")
        return
    blobs, files = load_calibration_blobs(args.calib_dir, args.verify, args.imgsz)
    try:
        fp32 = make_session(args.model)
        int8 = make_session(args.output)
    except Exception as e:
        print(f"  WARN: could not load sessions for verification: {e}", file=sys.stderr)
        return

    # Same decode + NMS as the evaluation pipeline (handles any output layout)
    from yolov8_cpu import YOLOv8CPU, nms
    m32 = YOLOv8CPU(args.model)
    mi8 = YOLOv8CPU(args.output)

    out_fp = fp32.get_outputs()[0].name
    out_i8 = int8.get_outputs()[0].name
    cos, n32, n8, n_match = [], 0, 0, 0
    for i, blob in enumerate(blobs):
        y0 = np.asarray(fp32.run([out_fp], {input_name: blob})[0])
        y1 = np.asarray(int8.run([out_i8], {input_name: blob})[0])
        c = float(np.dot(y0.ravel(), y1.ravel()) /
                  (np.linalg.norm(y0) * np.linalg.norm(y1) + 1e-12))
        cos.append(c)
        # detection-level agreement: NMS'd boxes at conf 0.25, IoU > 0.5
        d32 = nms(m32.parse_outputs([y0], 0.25), 0.25, 0.7)
        d8 = nms(mi8.parse_outputs([y1], 0.25), 0.25, 0.7)
        n32 += len(d32); n8 += len(d8)
        agreed = 0
        for a in d32:
            for b in d8:
                if int(a[5]) != int(b[5]):
                    continue
                iou = box_iou([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]])
                if iou > 0.5:
                    agreed += 1
                    break
        n_match += agreed
        print(f"  [{i + 1:3d}/{len(blobs)}] {os.path.basename(files[i]):<48s} "
              f"cos={c:.6f}  dets fp32:{len(d32):3d} int8:{len(d8):3d} "
              f"matched:{agreed:3d}")

    print(f"\n  Summary over {len(blobs)} images:")
    print(f"    cosine similarity  : mean {np.mean(cos):.6f}  min {np.min(cos):.6f}")
    print(f"    detections (0.25)  : fp32 {n32}  int8 {n8}  "
          f"matched {n_match} ({100.0 * n_match / max(1, n32):.1f}% of fp32)")


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Quantize a YOLOv8 ONNX to INT8 (standard onnxruntime "
                    "quantization, runnable on CPUExecutionProvider).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Input float32 ONNX (default: ./{DEFAULT_MODEL})")
    ap.add_argument("--output", default=None,
                    help="Output INT8 ONNX path (default: <model>_int8.onnx "
                         "or <model>_int8_dyn.onnx for dynamic)")
    ap.add_argument("--mode", choices=["static", "dynamic"], default="static",
                    help="static: QDQ INT8 with calibration data (default); "
                         "dynamic: weights-only INT8, no calibration needed")
    ap.add_argument("--calib-dir", default=None,
                    help="Calibration images directory (default: "
                         "../data/train/images next to this script, falling back "
                         "to valid/images or ./images)")
    ap.add_argument("--num-calib", type=int, default=64,
                    help="Number of calibration images (static mode)")
    ap.add_argument("--calib-method", choices=["minmax", "entropy"],
                    default="minmax", help="Calibration method (static mode)")
    ap.add_argument("--wt-type", choices=["QInt8", "QUInt8"], default="QInt8",
                    help="Weight quantization type (static mode)")
    ap.add_argument("--act-type", choices=["QUInt8", "QInt8"], default="QUInt8",
                    help="Activation quantization type (static mode)")
    ap.add_argument("--per-channel", action="store_true", default=True,
                    help="Per-channel weight quantization (recommended, default on)")
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    ap.add_argument("--verify", type=int, default=16,
                    help="Compare INT8 vs FP32 outputs on N calibration images "
                         "(0 = skip)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.calib_dir is None:
        for cand in (os.path.join(script_dir, "..", "data", "train", "images"),
                     os.path.join(script_dir, "..", "data", "valid", "images"),
                     os.path.join(script_dir, "images")):
            if os.path.isdir(cand):
                args.calib_dir = cand
                break
    if args.calib_dir and not os.path.isdir(args.calib_dir):
        print(f"WARN: calibration dir not found: {args.calib_dir}", file=sys.stderr)
        if args.mode == "static":
            print("ERROR: static quantization needs calibration images. "
                  "Use --calib-dir or --mode dynamic.", file=sys.stderr)
            return 1
        args.calib_dir = None

    if args.output is None:
        stem = os.path.splitext(args.model)[0]
        suffix = "int8_dyn" if args.mode == "dynamic" else "int8"
        args.output = f"{stem}_{suffix}.onnx"

    # model geometry (input name + size) from the ONNX metadata
    m = onnx.load(args.model, load_external_data=False)
    inputs = [i for i in m.graph.input if i.name not in
              {o.name for o in m.graph.output}] or list(m.graph.input)
    inp = inputs[0]
    input_name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 3:
        print(f"ERROR: unexpected input shape {dims}, expected [1,3,H,W]",
              file=sys.stderr)
        return 1
    args.imgsz = int(dims[2])
    print(f"  Model:      {os.path.abspath(args.model)}")
    print(f"  Input:      '{input_name}' {dims}")
    print(f"  Output:     {args.output}")
    print(f"  Backend:    onnxruntime {ort.__version__} / CPUExecutionProvider\n")

    quantize(args, input_name)

    # ── Sanity check: the INT8 model must load and run on CPU ────────────────
    print("--- Sanity check ---")
    blob = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    sess = make_session(args.output)
    sess.run(None, {sess.get_inputs()[0].name: blob})
    print(f"  OK: {args.output} loads and runs on CPUExecutionProvider")
    print(f"  Size: {os.path.getsize(args.model) / 1e6:.1f} MB -> "
          f"{os.path.getsize(args.output) / 1e6:.1f} MB\n")

    verify(args, input_name)

    print("============================================================")
    print("  INT8 model ready. Evaluate it with the existing tool:")
    print("============================================================")
    print("  ./run.sh --eval \\")
    print(f"      --model {args.output} \\")
    print("      --images ../data/test/images --labels ../data/test/labels \\")
    print("      --out ../results/cpu_int8/test")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
