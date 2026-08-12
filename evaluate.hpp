#pragma once
#include <string>
#include <cstdint>

/**
 * Options for COCO-style mAP evaluation on an image dataset.
 */
struct EvalOptions {
    std::string dataDir = "data/valid";  // must contain images/ and labels/ subfolders
    int numClasses = 1;                  // number of classes in the model
    float confThresh = 0.001f;           // confidence threshold (Ultralytics-style low default)
    float iouThresh = 0.7f;              // NMS IoU threshold
    int maxDet = 300;                    // max detections kept per image (COCO convention)
    std::string reportPath = "eval_report.txt";
};

/**
 * Run mAP evaluation over all images in opt.dataDir/images using ground truth
 * from opt.dataDir/labels (YOLO txt format: "cls cx cy w h", normalized).
 *
 * Computes:
 *   - mAP@0.5
 *   - mAP@0.5:0.95  (mean AP over 10 IoU thresholds 0.50..0.95)
 *   - Precision / Recall / F1-score at the max-F1 operating point
 * plus per-class and per-IoU-threshold tables.
 *
 * AP is computed from 101-point interpolated precision-recall curves
 * (COCO / Ultralytics convention).  The full report is written to
 * opt.reportPath.
 *
 * Requires qnnInit() + loadModel() to have been called first.
 *
 * @param boxScale   dequantization scale for the boxes output
 * @param boxOffset  dequantization offset for the boxes output
 * @param scrScale   dequantization scale for the scores output
 * @param scrOffset  dequantization offset for the scores output
 * @param modelPath  model path (report header only)
 * @returns true on success
 */
bool runEvaluation(const EvalOptions& opt, float boxScale, int32_t boxOffset,
                   float scrScale, int32_t scrOffset,
                   const std::string& modelPath);
