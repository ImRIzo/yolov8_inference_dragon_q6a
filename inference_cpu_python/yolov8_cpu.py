#!/usr/bin/env python3
"""
YOLOv8 ONNX Inference — CPU (onnxruntime)

Python port of the Radxa Q6A NPU C++ tool, running the same pipeline on CPU:

  * Video file / RTSP stream / single image / camera index
  * Letterbox preprocess (RGB, /255, NCHW, pad 114)
  * onnxruntime CPU inference on a YOLOv8 .onnx (e.g. best.onnx)
  * Parse + NMS + draw + live window with FPS / pipeline timing overlay
  * COCO-style mAP evaluation mode (--map) on YOLO-txt image datasets
  * Benchmark report saved to report.txt / eval_report_<dataset>.txt on exit

Handles the standard Ultralytics single output [1, 4+NC, A] as well as the
transposed [1, A, 4+NC] layout and the two-output QNN-style layout
(boxes [1, 4, A] + scores [1, NC, A]).

Usage examples:
    python3 yolov8_cpu.py --video test_video.mp4
    python3 yolov8_cpu.py --video test_video.mp4 --no-show
    python3 yolov8_cpu.py --rtsp rtsp://192.168.1.100:8554/stream
    python3 yolov8_cpu.py --image photo.jpg
    python3 yolov8_cpu.py --eval --images data/test/images --labels data/test/labels
    python3 yolov8_cpu.py --map data/valid
    python3 yolov8_cpu.py --map data/test --classes 1 --conf 0.001 --iou 0.7

Controls: ESC / q / Q quits and saves the report.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

# Quiet FFmpeg/OpenCV internal logging *before* cv2 is imported: on lossy
# RTSP links the bundled FFmpeg floods stderr with h264 decode warnings and
# OpenCV warns on every read timeout. Users can still override these.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")   # AV_LOG_FATAL and above
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")     # hide CV_LOG_WARNING

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL = "best.onnx"
DEFAULT_VIDEO = "test_video.mp4"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
WINDOW_W, WINDOW_H = 650, 650
WARMUP_FRAMES = 5
MAX_RECONNECT = 30
RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 3000
PAD_COLOR = (114, 114, 114)

# COCO class names (default for an 80-class YOLOv8 export)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class YOLOv8CPU:
    """Loads best.onnx, runs letterbox + inference, parses raw outputs."""

    def __init__(self, model_path):
        self.model_path = model_path
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 4)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) != 4 or inp.shape[1] != 3:
            raise ValueError(f"Unexpected input shape {inp.shape}, expected [1,3,H,W]")
        self.imgsz = int(inp.shape[2])
        if self.imgsz != int(inp.shape[3]):
            raise ValueError(f"Non-square input {inp.shape} not supported")

        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.num_classes = self._detect_num_classes()

    def _detect_num_classes(self):
        shapes = [list(o.shape) for o in self.sess.get_outputs()]
        if len(shapes) == 1:
            s = shapes[0]                       # [1, 4+NC, A] or [1, A, 4+NC]
            if len(s) == 3:
                _, d, a = s
            elif len(s) == 4:                   # [1, 1, 4+NC, A] style exports
                _, d, a = s[1], s[2], s[3]
            else:
                raise ValueError(f"Unexpected output shape {s}")
            d, a = int(d), int(a)
            # Anchor counts of YOLOv8/v5 heads at common imgsz values:
            # 320->2100, 416->3549, 512->5376, 640->8400, 768->18900,
            # 960->29400, 1280->33600 (YOLOv5 640->25200).
            known_anchors = {2100, 3549, 5376, 8400, 18900, 29400, 33600, 25200}
            if d in known_anchors and a not in known_anchors:
                nc = a - 4                      # [1, A, 4+NC]
            elif a in known_anchors and d not in known_anchors:
                nc = d - 4                      # [1, 4+NC, A]
            else:                               # fallback: larger dim = anchors
                nc = min(d, a) - 4
            return nc
        # Two outputs: boxes [1,4,A] + scores [1,NC,A]
        ncs = [int(s[1]) for s in shapes if int(s[1]) != 4]
        if not ncs:
            raise ValueError(f"Could not determine class count from outputs {shapes}")
        return ncs[0]

    def preprocess(self, bgr):
        """Letterbox -> RGB -> /255 -> NCHW. Mirrors the C++ preprocess."""
        h, w = bgr.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_left = (self.imgsz - nw) // 2
        pad_top = (self.imgsz - nh) // 2
        padded = cv2.copyMakeBorder(resized, pad_top, self.imgsz - nh - pad_top,
                                    pad_left, self.imgsz - nw - pad_left,
                                    cv2.BORDER_CONSTANT, value=PAD_COLOR)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, pad_left, pad_top

    def infer(self, blob):
        return self.sess.run(self.out_names, {self.input_name: blob})

    def parse_outputs(self, outs, conf_thresh):
        """Raw model outputs -> candidate detections [x1, y1, x2, y2, conf, cls].

        Detection = (x1, y1, x2, y2, confidence, class_id) in padded 640 space.
        """
        if len(outs) == 2:
            a, b = np.asarray(outs[0])[0], np.asarray(outs[1])[0]
            if a.shape[0] == 4:
                boxes, scores = a.T, b.T          # [A,4] [A,NC]
            else:
                boxes, scores = b.T, a.T
        else:
            p = np.asarray(outs[0])[0]            # [4+NC, A] or [A, 4+NC]
            # Anchor counts of YOLOv8/v5 heads at common imgsz values.
            known_anchors = {2100, 3549, 5376, 8400, 18900, 29400, 33600, 25200}
            if p.shape[0] in known_anchors and p.shape[1] not in known_anchors:
                pass                              # already [A, 4+NC]
            elif p.shape[1] in known_anchors and p.shape[0] not in known_anchors:
                p = p.T                          # [4+NC, A] -> [A, 4+NC]
            elif p.shape[0] <= p.shape[1]:
                p = p.T
            boxes, scores = p[:, :4], p[:, 4:]

        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]
        keep = confs >= conf_thresh
        if not keep.any():
            return []

        b = boxes[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]

        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        dets = list(zip(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                        confs, cls_ids))
        return dets


def nms(dets, conf_thresh, iou_thresh):
    """cv2 NMS, same as the C++ version."""
    if not dets:
        return []
    boxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in dets]
    scores = [float(d[4]) for d in dets]
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh, eta=1, top_k=0)
    keep = np.asarray(keep).ravel()
    return [dets[int(i)] for i in keep] if keep.size else []


def draw_detections(image, dets, names, scale, pad_left, pad_top):
    """Draw green boxes on a copy; dets are in padded-space, back-project first."""
    result = image.copy()
    for x1, y1, x2, y2, conf, cls in dets:
        label = names[int(cls)] if 0 <= int(cls) < len(names) else "?"
        px1 = int((x1 - pad_left) / scale)
        py1 = int((y1 - pad_top) / scale)
        px2 = int((x2 - pad_left) / scale)
        py2 = int((y2 - pad_top) / scale)
        cv2.rectangle(result, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(result, f"{conf:.2f} {label}", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return result


def resize_display(frame, max_w, max_h):
    scale = min(max_w / frame.shape[1], max_h / frame.shape[0])
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def draw_debug_overlay(image, pre_ms, inf_ms, post_ms, smooth_fps, avg_fps,
                       min_fps, max_fps, frame_count, num_dets, hw_line):
    total_ms = pre_ms + inf_ms + post_ms
    pipe_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    font, font_scale, thick, line_h = cv2.FONT_HERSHEY_DUPLEX, 0.55, 1, 20
    pad_x, pad_y = 8, 6
    y = pad_y + line_h
    yellow, green, grey = (0, 255, 255), (0, 255, 0), (180, 180, 180)

    def put(text, color):
        nonlocal y
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thick)
        strip_y = y - th - baseline
        bg = (pad_x - 2, strip_y - 2, tw + 8, th + baseline + 5)
        x0, y0 = max(0, bg[0]), max(0, bg[1])
        x1, y1 = min(image.shape[1], bg[0] + bg[2]), min(image.shape[0], bg[1] + bg[3])
        if x1 > x0 and y1 > y0:
            roi = image[y0:y1, x0:x1]
            overlay = np.full_like(roi, (20, 20, 30))
            cv2.addWeighted(overlay, 0.50, roi, 0.50, 0, roi)
        cv2.putText(image, text, (pad_x, y), font, font_scale, color, thick, cv2.LINE_AA)
        y += line_h

    fps_col = green if smooth_fps > 25.0 else yellow
    put(f"FPS:{smooth_fps:5.1f}  avg:{avg_fps:5.1f}  min:{min_fps:5.1f}  max:{max_fps:5.1f}  "
        f"#{frame_count} det:{num_dets}", fps_col)
    put(f"pipe: {pipe_fps:.1f} fps ({total_ms:.1f} ms)  pre:{pre_ms:.1f}  inf:{inf_ms:.1f}  "
        f"post:{post_ms:.1f} ms", yellow)
    put(hw_line, grey)


# ──────────────────────────────────────────────────────────────────────────────
# Video / image / RTSP inference
# ──────────────────────────────────────────────────────────────────────────────

def run_video(args, model, names, script_dir):
    video_path = args.rtsp if args.rtsp else args.video
    is_rtsp = bool(args.rtsp)

    cap = cv2.VideoCapture(video_path)
    if is_rtsp:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        print("  RTSP stream mode (low-latency)")
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}", file=sys.stderr)
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\n--- Video ---")
    print(f"  {video_path}")
    print(f"  {video_fps:.1f} fps, {total_frames} frames\n")

    if args.show:
        try:
            cv2.namedWindow("YOLOv8 - CPU (onnxruntime)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 - CPU (onnxruntime)", WINDOW_W, WINDOW_H)
        except cv2.error as e:
            print(f"WARN: cannot open display window ({e}) — running headless", file=sys.stderr)
            args.show = False

    total_pre = total_inf = total_post = 0.0
    frame_count = 0
    smooth_fps, min_fps, max_fps = 0.0, float("inf"), 0.0
    video_start = last_frame_time = time.perf_counter()

    reconnect_attempts = 0
    while True:
        ok, raw = cap.read()
        if is_rtsp and (not ok or raw is None):
            if reconnect_attempts >= MAX_RECONNECT:
                print("RTSP: max reconnects reached, exiting", file=sys.stderr)
                break
            reconnect_attempts += 1
            print(f"RTSP: frame dropped, reconnecting ({reconnect_attempts}/{MAX_RECONNECT})...",
                  file=sys.stderr)
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(video_path)
            continue
        reconnect_attempts = 0
        if not ok or raw is None:
            break

        # ── Preprocess ──
        t0 = time.perf_counter()
        blob, scale, pad_left, pad_top = model.preprocess(raw)
        t1 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000.0

        # ── Inference ──
        t2 = time.perf_counter()
        outs = model.infer(blob)
        t3 = time.perf_counter()
        inf_ms = (t3 - t2) * 1000.0

        # ── Postprocess ──
        t4 = time.perf_counter()
        dets = model.parse_outputs(outs, args.conf)
        dets = nms(dets, args.conf, args.iou)
        result = draw_detections(raw, dets, names, scale, pad_left, pad_top)
        t5 = time.perf_counter()
        post_ms = (t5 - t4) * 1000.0

        total_pre += pre_ms
        total_inf += inf_ms
        total_post += post_ms
        frame_count += 1

        elapsed = time.perf_counter() - video_start
        avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

        # Wall-clock frame-to-frame FPS (includes display overhead)
        now = time.perf_counter()
        wall_ms = (now - last_frame_time) * 1000.0
        last_frame_time = now
        wall_fps = 1000.0 / wall_ms if wall_ms > 0 else 0.0

        # EMA-smoothed FPS (dampened, low jitter)
        smooth_fps = wall_fps if smooth_fps <= 0 else 0.15 * wall_fps + 0.85 * smooth_fps
        if frame_count > WARMUP_FRAMES:
            min_fps = min(min_fps, wall_fps)
            max_fps = max(max_fps, wall_fps)

        # ── Display ──
        if args.show:
            display = resize_display(result, WINDOW_W, WINDOW_H)
            draw_debug_overlay(display, pre_ms, inf_ms, post_ms, smooth_fps, avg_fps,
                               min_fps, max_fps, frame_count, len(dets), args.hw_line)
            cv2.imshow("YOLOv8 - CPU (onnxruntime)", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    total_sec = time.perf_counter() - video_start
    avg_fps_final = frame_count / total_sec if total_sec > 0 else 0.0
    min_fps = min_fps if frame_count > WARMUP_FRAMES else avg_fps_final
    max_fps = max_fps if frame_count > WARMUP_FRAMES else avg_fps_final

    print("\n============================================================")
    print(f"  Processed {frame_count} frames in {total_sec:.1f} s")
    print(f"  Average FPS:        {avg_fps_final:.1f}")
    print(f"  Minimum FPS:        {min_fps:.1f}")
    print(f"  Maximum FPS:        {max_fps:.1f}")
    if frame_count:
        print(f"  Avg preprocess:     {total_pre / frame_count:.2f} ms")
        print(f"  Avg inference:      {total_inf / frame_count:.2f} ms")
        print(f"  Avg postprocess:    {total_post / frame_count:.2f} ms")
        print(f"  Avg total:          {(total_pre + total_inf + total_post) / frame_count:.2f} ms")
    print("============================================================")

    # ── Report ──
    lines = [
        "============================================================",
        "  YOLOv8 Video Inference Report (CPU)",
        "============================================================",
        f"  Date:       {time.ctime()}",
        f"  Video:      {video_path}",
        f"  Model:      {os.path.abspath(model.model_path)}",
        f"  Backend:    onnxruntime {ort.__version__} / CPUExecutionProvider",
        f"  Confidence: {args.conf:.2f}",
        f"  IoU:        {args.iou:.2f}",
        "------------------------------------------------------------",
        f"  Frames:     {frame_count}",
        f"  Total time: {total_sec:.2f} s",
        f"  Avg FPS:    {avg_fps_final:.1f}",
    ]
    if frame_count > WARMUP_FRAMES:
        lines += [f"  Min FPS:    {min_fps:.1f}", f"  Max FPS:    {max_fps:.1f}"]
    if frame_count:
        lines += [
            "------------------------------------------------------------",
            f"  Avg preprocess:   {total_pre / frame_count:.2f} ms",
            f"  Avg inference:    {total_inf / frame_count:.2f} ms",
            f"  Avg postprocess:  {total_post / frame_count:.2f} ms",
            f"  Avg total:        {(total_pre + total_inf + total_post) / frame_count:.2f} ms",
        ]
    lines.append("============================================================")
    report = "\n".join(lines) + "\n"
    report_path = os.path.join(script_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    return 0


def run_image(args, model, names, script_dir):
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    blob, scale, pad_left, pad_top = model.preprocess(img)
    outs = model.infer(blob)
    dets = nms(model.parse_outputs(outs, args.conf), args.conf, args.iou)
    t1 = time.perf_counter()

    result = draw_detections(img, dets, names, scale, pad_left, pad_top)
    print(f"  {img.shape[0]}x{img.shape[1]} -> {model.imgsz}x{model.imgsz}  "
          f"scale={scale:.4f}  pad=({pad_left},{pad_top})")
    print(f"  {len(dets)} detections in {(t1 - t0) * 1000:.1f} ms:")
    for x1, y1, x2, y2, conf, cls in dets:
        label = names[int(cls)] if 0 <= int(cls) < len(names) else "?"
        print(f"    {label:<20s} conf={conf:.3f}  "
              f"box=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    out_path = os.path.join(script_dir,
                            "result_" + os.path.basename(args.image))
    cv2.imwrite(out_path, result)
    print(f"  Annotated image saved: {out_path}")

    if args.show:
        try:
            display = resize_display(result, WINDOW_W, WINDOW_H)
            draw_debug_overlay(display, 0, (t1 - t0) * 1000, 0, 0, 0, 0, 0, 1,
                               len(dets), args.hw_line)
            cv2.imshow("YOLOv8 - CPU (onnxruntime)", display)
            print("  Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"WARN: cannot open display window ({e})", file=sys.stderr)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# COCO-style mAP evaluation (port of evaluate.cpp)
# ──────────────────────────────────────────────────────────────────────────────

NUM_IOU_THRESH = 10
NBINS = 1000
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compute_ap101(recall, precision):
    """101-point interpolated AP (np.trapz(np.interp(x, mrec, mpre)) convention)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 2, 0, -1):      # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renamed it
    return float(trapz(y, x))


def match_image(preds, gts, thr):
    """Greedy per-image matching: preds sorted by conf, best unused GT per class."""
    order = np.argsort(-np.array([p[4] for p in preds]), kind="stable")
    used = [False] * len(gts)
    out = []
    for oi in order:
        p = preds[int(oi)]
        best, best_iou = -1, thr
        for g, gt in enumerate(gts):
            if used[g] or gt[0] != int(p[5]):
                continue
            iou = box_iou(p, gt[1:5])       # gt is (cls, x1, y1, x2, y2)
            if iou > best_iou:
                best_iou, best = iou, g
        if best >= 0:
            used[best] = True
            out.append((float(p[4]), int(p[5]), True))
        else:
            out.append((float(p[4]), int(p[5]), False))
    return out


def gather_matches(images, thr, num_classes):
    per_class = [[] for _ in range(num_classes)]
    for img in images:
        for conf, cls, tp in match_image(img["preds"], img["gt"], thr):
            if 0 <= cls < num_classes:
                per_class[cls].append((conf, tp))
    for v in per_class:
        v.sort(key=lambda m: -m[0])
    return per_class


def ap_for_class(matches, num_gt):
    if num_gt <= 0 or not matches:
        return 0.0
    tp = fp = 0
    recall, precision = [], []
    for _, is_tp in matches:
        tp += is_tp
        fp += not is_tp
        recall.append(tp / num_gt)
        precision.append(tp / (tp + fp))
    return compute_ap101(recall, precision)


def prf_curves(matches, num_gt):
    """P/R/F1 sampled on NBINS confidence bins (Ultralytics-style)."""
    p, r, f = np.ones(NBINS), np.zeros(NBINS), np.zeros(NBINS)
    if num_gt <= 0:
        return p, r, f
    confs = np.array([m[0] for m in matches])
    tps = np.array([m[1] for m in matches], dtype=bool)
    for b in range(NBINS):
        cut = b / NBINS
        keep = confs >= cut                    # matches sorted desc
        tp, fp = int(tps[keep].sum()), int((~tps[keep]).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 1.0
        rec = tp / num_gt
        p[b], r[b] = prec, rec
        f[b] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return p, r, f


def smooth_box(y, frac):
    """Box filter with edge replication (Ultralytics' smooth(y, f))."""
    nf = int(round(len(y) * frac * 2.0))
    nf = nf // 2 * 2 + 1
    if nf < 1:
        nf = 1
    if nf >= len(y):
        return y.copy()
    half = nf // 2
    yp = np.concatenate((np.full(half, y[0]), y, np.full(half, y[-1])))
    kernel = np.ones(nf) / nf
    return np.convolve(yp, kernel, mode="valid")


def run_eval(args, model, names, script_dir):
    """Static labeled-image evaluation — same flow as the C++ yolov8_video --eval.

    Runs the ONNX model on CPU over every image of a dataset, saves the raw
    detections (after NMS + reverse letterbox) to predictions.csv together
    with audit files, computes the built-in COCO-style metrics, and hands the
    predictions to evaluate_metrics.py for the Ultralytics-exact numbers
    (evaluation_results.csv).
    """
    # ── Resolve dataset directories (--images/--labels or legacy --map) ─────
    if args.images:
        img_dir = args.images
        img_parent = os.path.dirname(img_dir.rstrip("/"))
        lbl_dir = args.labels or os.path.join(img_parent, "labels")
        ds_name = os.path.basename(img_parent) or "dataset"
    else:
        data_dir = args.map or "data/valid"
        img_dir = os.path.join(data_dir, "images")
        lbl_dir = os.path.join(data_dir, "labels")
        ds_name = os.path.basename(data_dir.rstrip("/")) or "dataset"

    out_dir = args.out or script_dir
    os.makedirs(out_dir, exist_ok=True)
    pred_csv  = os.path.join(out_dir, "predictions.csv")
    size_csv  = os.path.join(out_dir, "image_sizes.csv")
    cfg_csv   = os.path.join(out_dir, "eval_config.csv")
    img_list  = os.path.join(out_dir, "evaluated_images.txt")
    res_csv   = os.path.join(out_dir, "evaluation_results.csv")
    report_path = os.path.join(out_dir, f"eval_report_{ds_name}.txt")

    # ── Error checks ─────────────────────────────────────────────────────────
    if not os.path.isdir(img_dir):
        print(f"ERROR: image directory does not exist: {img_dir}", file=sys.stderr)
        return 1
    if not os.path.isdir(lbl_dir):
        print(f"ERROR: labels directory does not exist: {lbl_dir}", file=sys.stderr)
        return 1

    all_files = sorted(f for f in os.listdir(img_dir) if not f.startswith("."))
    image_files = [f for f in all_files if f.lower().endswith(IMG_EXTS)]
    unsupported = [f for f in all_files
                   if not f.lower().endswith(IMG_EXTS)
                   and os.path.isfile(os.path.join(img_dir, f))]
    for f in unsupported:
        print(f"WARN: unsupported image type in {img_dir} — skipped: {f}",
              file=sys.stderr)
    if not image_files:
        print(f"ERROR: no images (.jpg/.jpeg/.png/.bmp/.webp) found in {img_dir}",
              file=sys.stderr)
        return 1

    num_classes = args.classes if args.classes is not None else model.num_classes
    conf, iou = args.conf, args.iou
    max_det = args.max_det
    warmup = max(0, min(args.warmup, len(image_files) - 1))
    runtime = f"ONNX CPU (onnxruntime {ort.__version__}, CPUExecutionProvider)"

    print("\n============================================================")
    print("  ONNX CPU STATIC DETECTION EVALUATION")
    print("============================================================")
    print(f"  Runtime/backend    : {runtime}")
    print(f"  Model (onnx)       : {os.path.abspath(model.model_path)}")
    print(f"  Images directory   : {img_dir}   ({len(image_files)} images)")
    print(f"  Labels directory   : {lbl_dir}")
    print(f"  Input resolution   : {model.imgsz}x{model.imgsz}  "
          f"(tensor 1x3x{model.imgsz}x{model.imgsz})")
    print(f"  Classes            : {num_classes}")
    print(f"  Conf retention     : {conf:.3f}")
    print(f"  NMS IoU            : {iou:.2f}")
    print(f"  Max detections     : {max_det}")
    print(f"  Warm-up images     : {warmup} (excluded from timing, kept in accuracy)")
    print(f"  Output directory   : {out_dir}")
    print("------------------------------------------------------------")

    pcsv = open(pred_csv, "w", newline="")
    pw = csv.writer(pcsv)
    pw.writerow(["image", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    scsv = open(size_csv, "w", newline="")
    sw = csv.writer(scsv)
    sw.writerow(["image", "width", "height"])
    ilw = open(img_list, "w")

    images = []
    total_gt = total_preds = done = skipped = missing_labels = 0
    pre_times, inf_times, post_times, tot_times = [], [], [], []
    start_all = time.perf_counter()

    for idx, fname in enumerate(image_files, 1):
        raw = cv2.imread(os.path.join(img_dir, fname))
        if raw is None:
            print(f"WARN: cannot read image {fname} — skipped", file=sys.stderr)
            skipped += 1
            continue
        img = {"name": fname, "w": raw.shape[1], "h": raw.shape[0], "gt": []}

        # ── Ground truth (YOLO txt: "cls cx cy w h", normalized) ─────────────
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.isfile(lbl_path):
            missing_labels += 1       # valid empty-image case: no objects
        else:
            with open(lbl_path) as lf:
                for lineno, line in enumerate(lf, 1):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) < 5:
                        print(f"WARN: malformed label {lbl_path}:{lineno} "
                              f"({line.strip()!r}) — ignored", file=sys.stderr)
                        continue
                    try:
                        cls = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        print(f"WARN: malformed label {lbl_path}:{lineno} "
                              f"({line.strip()!r}) — ignored", file=sys.stderr)
                        continue
                    if cls < 0 or cls >= num_classes:
                        print(f"WARN: class id {cls} out of range [0,{num_classes}) "
                              f"in {lbl_path}:{lineno} — ignored", file=sys.stderr)
                        continue
                    if w <= 0 or h <= 0:
                        print(f"WARN: non-positive box size in {lbl_path}:{lineno} "
                              f"— ignored", file=sys.stderr)
                        continue
                    # normalized -> original pixel xyxy (no clipping, like val.py)
                    x1 = (cx - w / 2) * img["w"]
                    y1 = (cy - h / 2) * img["h"]
                    x2 = (cx + w / 2) * img["w"]
                    y2 = (cy + h / 2) * img["h"]
                    img["gt"].append((cls, x1, y1, x2, y2))
        total_gt += len(img["gt"])

        # ── Pipeline: preprocess / CPU inference / postprocess ───────────────
        t0 = time.perf_counter()
        blob, scale, pad_left, pad_top = model.preprocess(raw)
        t1 = time.perf_counter()
        outs = model.infer(blob)
        t2 = time.perf_counter()
        dets = nms(model.parse_outputs(outs, conf), conf, iou)
        t3 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0

        dets.sort(key=lambda d: -d[4])
        dets = dets[:max_det]               # max detections (COCO/Ultralytics)

        # ── Reverse letterbox with the EXACT resize dimensions ───────────────
        # (x - pad) * origW / resizedW — same as the C++ tool; no clipping,
        # matching Ultralytics scale_boxes.
        w_img, h_img = img["w"], img["h"]
        nw, nh = int(w_img * scale), int(h_img * scale)
        inv_x = (w_img / nw) if nw else 1.0 / scale
        inv_y = (h_img / nh) if nh else 1.0 / scale
        preds = []
        for x1, y1, x2, y2, conf_, cls in dets:
            px1 = (x1 - pad_left) * inv_x
            py1 = (y1 - pad_top) * inv_y
            px2 = (x2 - pad_left) * inv_x
            py2 = (y2 - pad_top) * inv_y
            preds.append((px1, py1, px2, py2, conf_, cls))
            pw.writerow([fname, cls, conf_, px1, py1, px2, py2])
        img["preds"] = preds
        total_preds += len(preds)
        sw.writerow([fname, img["w"], img["h"]])
        ilw.write(fname + "\n")

        # ── Timing (warm-up excluded) ────────────────────────────────────────
        timed = (idx - 1) >= warmup
        if timed:
            pre_times.append(pre_ms)
            inf_times.append(inf_ms)
            post_times.append(post_ms)
            tot_times.append(pre_ms + inf_ms + post_ms)
        done += 1

        print(f"  [{done:3d}/{len(image_files)}] {fname:<52s} gt:{len(img['gt']):2d} "
              f"det:{len(preds):3d}  {pre_ms:.1f}/{inf_ms:.1f}/{post_ms:.1f} ms"
              + ("" if timed else "  (warm-up)"))
        images.append(img)

    pcsv.close()
    scsv.close()
    ilw.close()
    wall_sec = time.perf_counter() - start_all

    if not images:
        print("ERROR: no images were processed", file=sys.stderr)
        return 1

    # ── Per-class GT stats ───────────────────────────────────────────────────
    gt_count = np.zeros(num_classes, dtype=int)
    img_count = np.zeros(num_classes, dtype=int)
    for img in images:
        seen = set()
        for gt in img["gt"]:
            gt_count[gt[0]] += 1
            seen.add(gt[0])
        for c in seen:
            img_count[c] += 1

    # ── AP per class per IoU threshold ───────────────────────────────────────
    ap = np.zeros((NUM_IOU_THRESH, num_classes))
    for t in range(NUM_IOU_THRESH):
        thr = 0.5 + 0.05 * t
        per_class = gather_matches(images, thr, num_classes)
        for c in range(num_classes):
            if gt_count[c] > 0:
                ap[t, c] = ap_for_class(per_class[c], gt_count[c])

    valid = [c for c in range(num_classes) if gt_count[c] > 0]
    map_at_iou = ap[:, valid].mean(axis=1) if valid else np.zeros(NUM_IOU_THRESH)
    map50 = float(map_at_iou[0])
    map5095 = float(map_at_iou.mean())
    ap50c = ap[0]
    ap5095c = ap.mean(axis=0)

    # ── P / R / F1 at IoU 0.5, max-F1 operating point (built-in) ─────────────
    per_class = gather_matches(images, 0.5, num_classes)
    pc, rc, fc = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    best_conf = 0.0
    if valid:
        pC, rC, fC = {}, {}, {}
        for c in valid:
            pC[c], rC[c], fC[c] = prf_curves(per_class[c], gt_count[c])
        mean_f = np.mean([fC[c] for c in valid], axis=0)
        sm_f = smooth_box(mean_f, 0.1)
        best = int(np.argmax(sm_f))
        best_conf = best / NBINS
        for c in valid:
            pc[c], rc[c], fc[c] = pC[c][best], rC[c][best], fC[c][best]
    P = float(pc[valid].mean()) if valid else 0.0
    R = float(rc[valid].mean()) if valid else 0.0
    F1 = float(fc[valid].mean()) if valid else 0.0

    # ── Timing statistics (warm-up excluded) ─────────────────────────────────
    def time_stats(v):
        if not v:
            return (0.0, 0.0, 0.0, 0.0)
        a = np.asarray(v)
        return (float(a.mean()), float(a.std()), float(a.min()), float(a.max()))

    s_pre, s_inf, s_post, s_tot = (time_stats(v) for v in
                                   (pre_times, inf_times, post_times, tot_times))
    timed_n = len(pre_times)
    img_per_s = done / wall_sec if wall_sec > 0 else 0.0

    def name_of(c):
        return names[c] if c < len(names) else f"class_{c}"

    # ── eval_config.csv (audit metadata for evaluate_metrics.py) ─────────────
    with open(cfg_csv, "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["key", "value"])
        for k, v in [
            ("model/runtime", runtime),
            ("model_path", os.path.abspath(model.model_path)),
            ("runtime/backend", runtime),
            ("input_resolution", f"{model.imgsz}x{model.imgsz}"),
            ("num_classes", str(num_classes)),
            ("num_images", str(done)),
            ("ground_truth_boxes", str(total_gt)),
            ("predicted_boxes", str(total_preds)),
            ("confidence_retention_threshold", f"{conf:.4f}"),
            ("nms_iou", f"{iou:.4f}"),
            ("max_det", str(max_det)),
            ("warmup_frames", str(warmup)),
            ("images_dir", img_dir),
            ("labels_dir", lbl_dir),
            ("date", time.strftime("%Y-%m-%d %H:%M:%S")),
        ]:
            cw.writerow([k, v])

    # ── Report ───────────────────────────────────────────────────────────────
    L = []
    L.append("============================================================")
    L.append("  YOLOv8 Evaluation Report (COCO-style mAP) - ONNX CPU")
    L.append("============================================================")
    L.append(f"  Date:            {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Runtime/backend: {runtime}")
    L.append(f"  Dataset:         {img_dir}")
    L.append(f"  Labels:          {lbl_dir}")
    L.append(f"  Model:           {os.path.abspath(model.model_path)}")
    L.append(f"  Input:           1x3x{model.imgsz}x{model.imgsz}")
    L.append(f"  Outputs:         {model.out_names}")
    L.append(f"  Images:          {done}")
    L.append(f"  GT boxes:        {total_gt}")
    L.append(f"  Predictions:     {total_preds}")
    L.append(f"  Conf: {conf:.3f}   NMS IoU: {iou:.2f}   "
             f"Classes: {num_classes}   Max det: {max_det}")
    L.append(f"  Warm-up images:  {warmup} (excluded from timing, kept in accuracy)")
    L.append("------------------------------------------------------------")
    L.append(f"  mAP@0.5          {map50:.4f}")
    L.append(f"  mAP@0.5:0.95     {map5095:.4f}")
    L.append(f"  Precision        {P:.4f}")
    L.append(f"  Recall           {R:.4f}")
    L.append(f"  F1-score         {F1:.4f}")
    L.append(f"  Best conf        {best_conf:.3f}   (operating point of P/R/F1)")
    L.append("------------------------------------------------------------")
    L.append("  AP per IoU threshold:")
    L.append("    IoU      AP")
    for t in range(NUM_IOU_THRESH):
        L.append(f"    {0.5 + 0.05 * t:.2f}    {map_at_iou[t]:.4f}")
    L.append("------------------------------------------------------------")
    L.append("  Per class:")
    L.append("    Class              Img    GT    AP50    AP50-95   Prec    Rec     F1")
    for c in range(num_classes):
        if gt_count[c] == 0:
            continue
        L.append(f"    {name_of(c):<16s} {img_count[c]:4d} {gt_count[c]:5d}   "
                 f"{ap50c[c]:6.4f}   {ap5095c[c]:6.4f}   "
                 f"{pc[c]:6.4f}  {rc[c]:6.4f}  {fc[c]:6.4f}")
    L.append("------------------------------------------------------------")
    L.append("  Timing (warm-up excluded; std = population stddev):")
    L.append(f"    Images timed:      {timed_n}")
    L.append(f"    Total:             {wall_sec:.3f} s   ({img_per_s:.1f} images/s, {done} images)")
    L.append("                        mean       std       min       max")
    L.append(f"    preprocess  (ms) {s_pre[0]:8.3f}  {s_pre[1]:8.3f}  {s_pre[2]:8.3f}  {s_pre[3]:8.3f}")
    L.append(f"    ONNX inference(ms){s_inf[0]:8.3f}  {s_inf[1]:8.3f}  {s_inf[2]:8.3f}  {s_inf[3]:8.3f}")
    L.append(f"    postprocess (ms) {s_post[0]:8.3f}  {s_post[1]:8.3f}  {s_post[2]:8.3f}  {s_post[3]:8.3f}")
    L.append(f"    total       (ms) {s_tot[0]:8.3f}  {s_tot[1]:8.3f}  {s_tot[2]:8.3f}  {s_tot[3]:8.3f}")
    if skipped:
        L.append(f"  Skipped (unreadable): {skipped}")
    if missing_labels:
        L.append(f"  Images without label file (treated as empty): {missing_labels}")
    L.append("------------------------------------------------------------")
    L.append(f"  Files: {pred_csv}")
    L.append(f"         {size_csv}")
    L.append(f"         {cfg_csv}")
    L.append(f"         {img_list}")
    L.append("------------------------------------------------------------")
    L.append("  Evaluated image files:")
    for img in images:
        L.append("    " + img["name"])
    L.append("============================================================")
    rep = "\n".join(L) + "\n"

    with open(report_path, "w") as f:
        f.write(rep)
    print(rep)
    print(f"Report saved: {report_path}")

    # ── Final console summary (same format as the C++ tool) ─────────────────
    print("\n============================================================")
    print("ONNX CPU STATIC DETECTION EVALUATION")
    print("============================================================")
    print(f"Images              : {done}")
    print(f"Ground-truth boxes  : {total_gt}")
    print(f"Predicted boxes     : {total_preds}")
    print()
    print(f"mAP@0.5              : {map50:.4f}")
    print(f"mAP@0.5:0.95         : {map5095:.4f}")
    print(f"Precision            : {P:.4f}")
    print(f"Recall               : {R:.4f}")
    print(f"F1                   : {F1:.4f}")
    print()
    print(f"Prediction threshold : {conf:.3f}")
    print(f"NMS IoU              : {iou:.2f}")
    print(f"Max detections       : {max_det}")
    print()
    print("(Built-in COCO 101-point metrics above; the canonical")
    print(" Ultralytics-exact numbers are printed by evaluate_metrics.py)")
    print()
    print(f"--- Timing ({warmup} warm-up excluded, {timed_n} timed images) ---")
    print("                       mean       std       min       max")
    print(f"  preprocess  (ms) {s_pre[0]:8.3f}  {s_pre[1]:8.3f}  {s_pre[2]:8.3f}  {s_pre[3]:8.3f}")
    print(f"  ONNX inference(ms){s_inf[0]:8.3f}  {s_inf[1]:8.3f}  {s_inf[2]:8.3f}  {s_inf[3]:8.3f}")
    print(f"  postprocess (ms) {s_post[0]:8.3f}  {s_post[1]:8.3f}  {s_post[2]:8.3f}  {s_post[3]:8.3f}")
    print(f"  total       (ms) {s_tot[0]:8.3f}  {s_tot[1]:8.3f}  {s_tot[2]:8.3f}  {s_tot[3]:8.3f}")
    print()
    print(f"  Total images      : {done}")
    print(f"  Total processing  : {wall_sec:.3f} s")
    print(f"  Images/second     : {img_per_s:.2f}")
    print("============================================================")

    print(f"\nSaved:\n  {pred_csv}\n  {size_csv}\n  {cfg_csv}\n  {img_list}\n  {report_path}")

    # ── Ultralytics-exact metrics via evaluate_metrics.py ────────────────────
    evaluator = os.path.join(script_dir, "evaluate_metrics.py")
    if not os.path.isfile(evaluator):
        evaluator = os.path.join(script_dir, "..", "evaluate_metrics.py")
    if os.path.isfile(evaluator):
        cmd = [sys.executable, evaluator,
               "--predictions", pred_csv,
               "--labels", lbl_dir,
               "--images", img_dir,
               "--sizes", size_csv,
               "--config", cfg_csv,
               "--output", res_csv]
        print("\n--- Ultralytics-exact metrics (evaluate_metrics.py) ---")
        rc = subprocess.call(cmd)
        if rc:
            print(f"\nNOTE: evaluate_metrics.py exited with code {rc} — rerun manually:\n  "
                  + " ".join(cmd))
    else:
        print("\nNOTE: evaluate_metrics.py not found — run it manually with "
              "--predictions/--labels/--images/--sizes/--config/--output")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def load_names(names_arg, num_classes):
    if names_arg:
        if os.path.isfile(names_arg):
            with open(names_arg) as f:
                names = [ln.strip() for ln in f
                         if ln.strip() and not ln.strip().startswith("#")]
        else:
            names = [n.strip() for n in names_arg.split(",") if n.strip()]
        if len(names) != num_classes:
            print(f"WARN: {len(names)} names given but model has {num_classes} classes",
                  file=sys.stderr)
        return names
    if num_classes == 80:
        return list(COCO_CLASSES)
    return [f"class_{c}" for c in range(num_classes)]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="YOLOv8 ONNX inference on CPU (onnxruntime) - Python port of "
                    "the Q6A NPU video tool.")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="Input video file "
                    "(default: ./test_video.mp4; camera index like '0' also works)")
    ap.add_argument("--rtsp", help="RTSP stream URL (low-latency mode)")
    ap.add_argument("--image", help="Run inference on a single image")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"ONNX model path (default: ./{DEFAULT_MODEL})")
    ap.add_argument("--names", help="Class names: path to a .txt (one per line) "
                    "or comma-separated list. Default: COCO names for 80 classes")
    ap.add_argument("--conf", type=float, default=None,
                    help=f"Confidence threshold (default: {DEFAULT_CONF:.2f})")
    ap.add_argument("--iou", type=float, default=None,
                    help=f"NMS IoU threshold (default: {DEFAULT_IOU:.2f})")
    ap.add_argument("--map", nargs="?", const="data/valid", default=None,
                    metavar="DATADIR",
                    help="Legacy: evaluate a dataset dir containing images/ + "
                         "labels/ (YOLO txt). Default dataset: data/valid")
    ap.add_argument("--eval", action="store_true",
                    help="Static labeled-image evaluation mode (saves "
                         "predictions.csv and runs evaluate_metrics.py)")
    ap.add_argument("--images", default=None,
                    help="Test images directory (implies --eval)")
    ap.add_argument("--labels", default=None,
                    help="YOLO txt labels directory (default: ../labels next "
                         "to the images directory)")
    ap.add_argument("--out", default=None,
                    help="Output directory for predictions.csv, reports, CSVs "
                         "(default: script directory)")
    ap.add_argument("--max-det", type=int, default=300,
                    help="Max detections kept per image (default: 300)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Warm-up images excluded from timing stats (default: 5)")
    ap.add_argument("--classes", type=int, default=None,
                    help="Number of model classes (default: auto from ONNX; "
                         "used with --eval/--map)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="Disable display window (benchmark mode)")
    ap.set_defaults(show=True)
    args = ap.parse_args(argv)

    print("============================================================")
    print("  YOLOv8 - CPU (onnxruntime) - Python")
    print("============================================================")

    model = YOLOv8CPU(args.model)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    args.hw_line = f"CPU: onnxruntime {ort.__version__} | {model_name} | ESC/Q quit"
    names = load_names(args.names, model.num_classes)

    conf_given = args.conf is not None
    iou_given = args.iou is not None
    args.conf = args.conf if conf_given else DEFAULT_CONF
    args.iou = args.iou if iou_given else DEFAULT_IOU
    eval_mode = bool(args.map) or args.eval or bool(args.images)
    if eval_mode:                               # Ultralytics val.py defaults
        args.conf = args.conf if conf_given else 0.001
        args.iou = args.iou if iou_given else 0.7

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n  Model:  {os.path.abspath(args.model)}")
    print(f"  Input:  1x3x{model.imgsz}x{model.imgsz}  Classes: {model.num_classes}")
    print(f"  Outputs: {model.out_names}")
    print(f"  Conf: {args.conf:.3f}  IoU: {args.iou:.2f}\n")

    if eval_mode:
        return run_eval(args, model, names, script_dir)
    if args.image:
        return run_image(args, model, names, script_dir)
    return run_video(args, model, names, script_dir)


if __name__ == "__main__":
    sys.exit(main())
