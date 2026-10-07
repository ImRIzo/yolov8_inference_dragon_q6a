#!/usr/bin/env python3
"""
evaluate_metrics.py — Ultralytics-exact detection metrics for QNN predictions.

Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall and F1 from the raw
predictions saved by `yolov8_video --eval` (predictions.csv) and the
YOLO-format ground-truth labels.  The computation mirrors Ultralytics
8.4.45 val.py EXACTLY (verified against the installed source):

  * per-image greedy matching   : BaseValidator.match_predictions()
  * AP via 101-point interp PR  : utils.metrics.compute_ap()
  * P/R curves on 1000 conf bins: utils.metrics.ap_per_class()
  * operating point             : i = argmax(smooth(mean F1 curve, 0.1))
  * final metrics               : Metric.mean_results() ->
      mp  = p[:, i].mean()
      mr  = r[:, i].mean()
      mAP50    = ap[:, 0].mean()
      mAP50-95 = ap.mean()
  * F1 reported as 2*P*R/(P+R) at the same operating point (the convention
    used for the FP32 PyTorch reference table).

Only numpy is required (no ultralytics/torch dependency).

Usage:
    python3 evaluate_metrics.py \
        --predictions predictions.csv \
        --labels data/test/labels \
        --images data/test/images \
        --sizes image_sizes.csv \
        --config eval_config.csv \
        --output evaluation_results.csv

Default --sizes is <predictions dir>/image_sizes.csv and default --output is
<predictions dir>/evaluation_results.csv.  --images is optional; it restricts
the ground-truth set to images that actually exist in the images directory
and allows reading image sizes with cv2 if image_sizes.csv is unavailable.
"""

import argparse
import csv
import os
import sys

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)          # IoU thresholds 0.50..0.95 (val.py iouv)
EPS = 1e-16

# ──────────────────────────────────────────────────────────────────────────────
# Metrics — mirror of ultralytics 8.4.45 (utils/metrics.py, engine/validator.py)
# ──────────────────────────────────────────────────────────────────────────────


def box_iou(box1, box2):
    """Pairwise IoU. box1 (M,4), box2 (N,4) xyxy -> (M,N)."""
    if box1.shape[0] == 0 or box2.shape[0] == 0:
        return np.zeros((box1.shape[0], box2.shape[0]))
    b1 = box1[:, None, :]                  # (M,1,4)
    b2 = box2[None, :, :]                  # (1,N,4)
    inter = (np.minimum(b1[..., 2:], b2[..., 2:]) -
             np.maximum(b1[..., :2], b2[..., :2]))
    inter = np.clip(inter, 0, None).prod(-1)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter
    return inter / (union + EPS)


def match_predictions(pred_classes, true_classes, iou):
    """Mirror of ultralytics 8.4.45 BaseValidator.match_predictions().

    pred_classes (N,), true_classes (M,), iou (M,N) -> correct (N,10) bool.
    """
    correct = np.zeros((pred_classes.shape[0], IOUV.shape[0]), dtype=bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes[None, :]
    iou = iou * correct_class               # zero out the wrong classes
    for i, threshold in enumerate(IOUV):
        matches = np.nonzero(iou >= threshold)      # IoU >= threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                # sort by IoU desc, keep first per detection, then per GT
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def compute_ap(recall, precision):
    """Mirror of ultralytics compute_ap() — 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))   # precision envelope
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ap = trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def smooth(y, f=0.05):
    """Mirror of ultralytics smooth() — box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Mirror of ultralytics 8.4.45 ap_per_class() (numpy, no plotting).

    Returns p, r, f1, ap, ap_class_index, p_curve, r_curve, f1_curve, x
    where p/r/f1 are at the max-F1 operating point (per class) and ap is
    (nc, 10) with one column per IoU threshold.
    """
    i = np.argsort(-conf)                   # sort by objectness
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x = np.linspace(0, 1, 1000)
    ap, p_curve, r_curve = (np.zeros((nc, tp.shape[1])),
                            np.zeros((nc, 1000)),
                            np.zeros((nc, 1000)))
    for ci, c in enumerate(unique_classes):
        m = pred_cls == c
        n_l = nt[ci]                        # number of labels
        n_p = m.sum()                       # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[m]).cumsum(0)         # accumulate FPs
        tpc = tp[m].cumsum(0)               # accumulate TPs

        recall = tpc / (n_l + EPS)          # recall curve
        r_curve[ci] = np.interp(-x, -conf[m], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)       # precision curve
        p_curve[ci] = np.interp(-x, -conf[m], precision[:, 0], left=1)

        for j in range(tp.shape[1]):        # AP per IoU threshold
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)

    i = smooth(f1_curve.mean(0), 0.1).argmax()   # max-F1 operating point
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    return p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def stem(name):
    base = os.path.basename(name)
    root, ext = os.path.splitext(base)
    return root if ext else base      # strip .jpg/.txt/.png … from both sides


def load_predictions(path):
    """predictions.csv -> {stem: [(x1,y1,x2,y2,conf,cls), ...]} in file order."""
    preds = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return preds
        expected = ["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        hdr = [h.strip().lower() for h in header]
        if hdr != expected:
            sys.exit(f"ERROR: unexpected predictions.csv header {header!r}; "
                     f"expected {expected}")
        for row in reader:
            if not row or len(row) < 7:
                continue
            image, cls, conf = row[0], int(float(row[1])), float(row[2])
            x1, y1, x2, y2 = (float(v) for v in row[3:7])
            preds.setdefault(stem(image), []).append((x1, y1, x2, y2, conf, cls))
    return preds


def load_sizes(path):
    """image_sizes.csv -> {stem: (w, h)}."""
    sizes = {}
    if not path or not os.path.isfile(path):
        return sizes
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            sizes[stem(row[0])] = (int(row[1]), int(row[2]))
    return sizes


def load_config(path):
    """eval_config.csv -> dict of key:value strings."""
    cfg = {}
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                cfg[row[0]] = row[1]
    return cfg


def read_image_size(image_path):
    """Read (w, h) from a jpg/png image file without OpenCV."""
    with open(image_path, "rb") as f:
        head = f.read(32)
    if len(head) < 24:
        return None
    if head[:2] == b"\xff\xd8":             # JPEG
        f.seek(2)
        while True:
            b = f.read(4)
            if len(b) < 4:
                return None
            if b[0] != 0xFF:
                continue
            marker = b[1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            seg_len = int.from_bytes(f.read(2), "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                data = f.read(5)
                if len(data) < 5:
                    return None
                h, w = int.from_bytes(data[1:3], "big"), int.from_bytes(data[3:5], "big")
                return w, h
            f.seek(seg_len - 2, 1)
    elif head[:8] == b"\x89PNG\r\n\x1a\n":  # PNG (IHDR width/height at bytes 16..24)
        w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        return w, h
    return None


def load_labels(labels_dir, images_dir, sizes):
    """YOLO txt labels -> {stem: (w, h, [(cls, x1, y1, x2, y2), ...])}.

    The ground-truth set is restricted to label files whose stem matches an
    image file in images_dir (when given).  Boxes are converted to original
    pixel coordinates using image_sizes.csv (or the image files themselves
    as a fallback).
    """
    if not os.path.isdir(labels_dir):
        sys.exit(f"ERROR: labels directory does not exist: {labels_dir}")

    available_images = set()
    if images_dir:
        if not os.path.isdir(images_dir):
            sys.exit(f"ERROR: images directory does not exist: {images_dir}")
        for name in os.listdir(images_dir):
            if not name.startswith("."):
                available_images.add(stem(name))

    gts = {}
    num_classes = 1
    for name in sorted(os.listdir(labels_dir)):
        if not name.lower().endswith(".txt"):
            continue
        s = stem(name)
        if images_dir and s not in available_images:
            continue
        w, h = sizes.get(s, (None, None))
        if w is None:
            img_file = None
            if images_dir:
                for cand in os.listdir(images_dir):
                    if stem(cand) == s:
                        img_file = os.path.join(images_dir, cand)
                        break
            if img_file:
                size = read_image_size(img_file)
                if size:
                    w, h = size
            if w is None:
                sys.exit(f"ERROR: no size information for image '{s}' — provide "
                         f"image_sizes.csv (or run yolov8_video --eval first)")
        boxes = []
        path = os.path.join(labels_dir, name)
        with open(path) as lf:
            for lineno, line in enumerate(lf, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"WARN: malformed label {name}:{lineno} ({line!r}) — ignored",
                          file=sys.stderr)
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = (float(p) for p in parts[1:5])
                except ValueError:
                    print(f"WARN: malformed label {name}:{lineno} ({line!r}) — ignored",
                          file=sys.stderr)
                    continue
                if cls < 0:
                    print(f"WARN: negative class id in {name}:{lineno} — ignored",
                          file=sys.stderr)
                    continue
                num_classes = max(num_classes, cls + 1)
                if bw <= 0 or bh <= 0:
                    print(f"WARN: non-positive box size in {name}:{lineno} — ignored",
                          file=sys.stderr)
                    continue
                # normalized -> original pixel xyxy (no clipping, like val.py)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append((cls, x1, y1, x2, y2))
        gts[s] = (int(w), int(h), boxes)
    return gts, num_classes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ultralytics-exact mAP/P/R/F1 from yolov8_video predictions.csv")
    ap.add_argument("--predictions", required=True, help="path to predictions.csv")
    ap.add_argument("--labels", required=True, help="directory with YOLO txt labels")
    ap.add_argument("--images", default=None,
                    help="optional images directory (restricts the GT set and "
                         "enables size fallback)")
    ap.add_argument("--sizes", default=None,
                    help="path to image_sizes.csv (default: <predictions dir>/image_sizes.csv)")
    ap.add_argument("--config", default=None,
                    help="path to eval_config.csv written by yolov8_video --eval")
    ap.add_argument("--output", default=None,
                    help="output CSV (default: <predictions dir>/evaluation_results.csv)")
    args = ap.parse_args(argv)

    pred_dir = os.path.dirname(os.path.abspath(args.predictions))
    sizes_path = args.sizes or os.path.join(pred_dir, "image_sizes.csv")
    out_path = args.output or os.path.join(pred_dir, "evaluation_results.csv")

    config = load_config(args.config)
    sizes = load_sizes(sizes_path)
    preds = load_predictions(args.predictions)
    gts, gt_num_classes = load_labels(args.labels, args.images, sizes)

    num_classes = int(config.get("num_classes", gt_num_classes))
    conf_thresh = float(config.get("confidence_retention_threshold", 0.001))
    nms_iou = float(config.get("nms_iou", 0.70))
    max_det = int(config.get("max_det", 300))
    runtime = config.get("model/runtime", "INT8 QNN HTP NPU (QCS6490)")

    # Predictions for images outside the ground-truth set: warn + exclude
    # (they would silently count as false positives otherwise).
    excluded = 0
    for s in list(preds.keys()):
        if s not in gts:
            print(f"WARN: predictions for unlabeled image '{s}' — excluded from metrics",
                  file=sys.stderr)
            del preds[s]
            excluded += 1

    # ── Build per-image stats exactly like val.py update_metrics() ──────────
    tp_parts, conf_parts, pred_cls_parts, target_cls_parts = [], [], [], []
    for s in sorted(gts):
        w, h, boxes = gts[s]
        p = preds.get(s, [])
        gboxes = np.array([b[1:] for b in boxes], dtype=np.float64).reshape(-1, 4) \
            if boxes else np.zeros((0, 4))
        gcls = np.array([b[0] for b in boxes], dtype=int) if boxes else np.zeros(0, dtype=int)
        target_cls_parts.append(gcls)
        if p:
            parr = np.array(p, dtype=np.float64)
            pboxes, pconf, pcls = parr[:, :4], parr[:, 4], parr[:, 5].astype(int)
            iou = box_iou(gboxes, pboxes)
            correct = match_predictions(pcls, gcls, iou)
            tp_parts.append(correct)
            conf_parts.append(pconf)
            pred_cls_parts.append(pcls)
        else:
            tp_parts.append(np.zeros((0, 10), dtype=bool))
            conf_parts.append(np.zeros(0))
            pred_cls_parts.append(np.zeros(0, dtype=int))

    tp = np.concatenate(tp_parts, 0) if tp_parts else np.zeros((0, 10), bool)
    conf = np.concatenate(conf_parts, 0) if conf_parts else np.zeros(0)
    pred_cls = np.concatenate(pred_cls_parts, 0) if pred_cls_parts else np.zeros(0, int)
    target_cls = np.concatenate(target_cls_parts, 0).astype(int) \
        if target_cls_parts else np.zeros(0, int)

    num_images = len(gts)
    num_gt = int(len(target_cls))
    num_preds = int(len(conf))

    # val.py:  if len(stats) and stats[0].any():  ap_per_class(...)
    if len(tp) and tp.any():
        p, r, f1, ap, ap_class, p_curve, r_curve, f1_curve, x = \
            ap_per_class(tp, conf, pred_cls, target_cls)
        map50 = float(ap[:, 0].mean()) if ap.size else 0.0
        map5095 = float(ap.mean()) if ap.size else 0.0
        P = float(p.mean()) if p.size else 0.0
        R = float(r.mean()) if r.size else 0.0
        # operating point: argmax of the smoothed mean-F1 curve (val.py)
        i_op = int(np.nanargmax(smooth(f1_curve.mean(0), 0.1)))
        best_conf = float(x[i_op])
    else:
        map50 = map5095 = P = R = best_conf = 0.0
        ap = np.zeros((0, 10))
        ap_class = np.zeros(0, int)
        i_op = 0
    F1 = 2.0 * P * R / (P + R) if (P + R) > 0 else 0.0

    # ── Per-class / per-IoU detail tables ────────────────────────────────────
    names = {0: "Ganoderma"} if num_classes == 1 else {c: f"class_{c}" for c in range(num_classes)}

    lines = []
    add = lines.append
    add("============================================================")
    add("QNN HTP NPU STATIC DETECTION EVALUATION (Ultralytics-exact)")
    add("============================================================")
    add(f"  Runtime/backend   : {runtime}")
    add(f"  Images            : {num_images}")
    add(f"  Ground-truth boxes: {num_gt}")
    add(f"  Predicted boxes   : {num_preds}")
    if excluded:
        add(f"  Excluded preds    : {excluded} rows (unlabeled images)")
    add("")
    add(f"  mAP@0.5            : {map50:.4f}")
    add(f"  mAP@0.5:0.95       : {map5095:.4f}")
    add(f"  Precision          : {P:.4f}")
    add(f"  Recall             : {R:.4f}")
    add(f"  F1                 : {F1:.4f}")
    add("")
    add(f"  Prediction threshold : {conf_thresh:.3f}")
    add(f"  NMS IoU              : {nms_iou:.2f}")
    add(f"  Max detections       : {max_det}")
    add(f"  P/R/F1 operating conf: {best_conf:.3f} (max-F1, smoothed f=0.1)")
    add("")
    add("  AP per IoU threshold:")
    add("    IoU      AP")
    for t in range(10):
        ap_i = float(ap[:, t].mean()) if ap.size else 0.0
        add(f"    {0.5 + 0.05 * t:.2f}    {ap_i:.4f}")
    add("")
    add("  Per class:")
    add("    Class              GT      AP50    AP50-95")
    for ci, c in enumerate(ap_class):
        n_c = int((target_cls == c).sum())
        add(f"    {names.get(int(c), f'class_{c}'):<16s} {n_c:5d}   "
            f"{ap[ci, 0]:6.4f}   {ap[ci].mean():6.4f}")
    add("============================================================")
    report = "\n".join(lines) + "\n"
    print(report)

    # ── evaluation_results.csv ───────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model/runtime", "number_of_images", "ground_truth_boxes",
                         "predicted_boxes", "mAP50", "mAP50_95", "precision",
                         "recall", "F1", "confidence_retention_threshold",
                         "nms_iou", "max_det"])
        writer.writerow([runtime, num_images, num_gt, num_preds,
                         f"{map50:.6f}", f"{map5095:.6f}", f"{P:.6f}", f"{R:.6f}",
                         f"{F1:.6f}", f"{conf_thresh:.4f}", f"{nms_iou:.4f}",
                         max_det])
    print(f"Evaluation results saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
