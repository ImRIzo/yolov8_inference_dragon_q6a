#pragma once
#include <string>
#include <cstdint>

/**
 * Options for COCO/Ultralytics-style mAP evaluation on an image dataset.
 */
struct EvalOptions {
    std::string imagesDir;   // directory with test images (--images)
    std::string labelsDir;   // directory with YOLO txt labels (--labels)
    std::string dataDir;     // legacy --map: dataset dir containing images/ + labels/
    std::string outDir = ".";              // output dir: predictions.csv, eval_config.csv, reports
    std::string modelPath;                 // model/context binary path (report header only)
    std::string runtimeName = "INT8 QNN HTP NPU (QCS6490)";  // backend label in reports
    std::string pythonScript;              // path to evaluate_metrics.py (auto-run, optional)
    int numClasses = 1;                    // number of classes in the model
    float confThresh = 0.001f;             // confidence threshold for retaining predictions
    float iouThresh = 0.7f;                // NMS IoU threshold
    int maxDet = 300;                      // max detections kept per image (COCO/Ultralytics)
    int warmupFrames = 5;                  // warm-up inferences excluded from timing stats
    bool autoPython = true;                // auto-run evaluate_metrics.py when available
    std::string reportPath;                // optional explicit report path (default: <outDir>/eval_report_<ds>.txt)
};

/**
 * Run mAP evaluation over all images in the dataset using ground truth
 * YOLO txt labels ("cls cx cy w h", normalized).
 *
 * Computes (C++ built-in, COCO 101-point interpolation):
 *   - mAP@0.5
 *   - mAP@0.5:0.95  (mean AP over 10 IoU thresholds 0.50..0.95)
 *   - Precision / Recall / F1-score at the max-F1 operating point
 *
 * and writes the raw detections (after dequantization → decode → reverse
 * letterbox → NMS) to <outDir>/predictions.csv, plus audit files
 * (<outDir>/image_sizes.csv, <outDir>/eval_config.csv,
 * <outDir>/evaluated_images.txt) so evaluate_metrics.py can reproduce the
 * exact Ultralytics val.py metrics for research-paper results.
 *
 * Also collects per-stage timing (preprocess / QNN inference / postprocess)
 * with mean / std / min / max over all non-warm-up images.
 *
 * Requires qnnInit() + loadModel() to have been called first.
 *
 * @param boxScale   dequantization scale for the boxes output
 * @param boxOffset  dequantization offset for the boxes output
 * @param scrScale   dequantization scale for the scores output
 * @param scrOffset  dequantization offset for the scores output
 * @returns true on success
 */
bool runEvaluation(const EvalOptions& opt, float boxScale, int32_t boxOffset,
                   float scrScale, int32_t scrOffset);

