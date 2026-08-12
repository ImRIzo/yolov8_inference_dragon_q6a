#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection {
    int classId;
    float confidence;
    float x1, y1, x2, y2;
};

extern const char* COCO_CLASSES[80];

// Parse raw QNN output tensors into detections
std::vector<Detection> parseOutputs(const uint8_t* boxData, const uint8_t* scrData,
                                     const uint32_t boxDims[4], const uint32_t scrDims[4],
                                     float boxScale, float scrScale, float confThresh);

// Non-maximum suppression
std::vector<Detection> nms(const std::vector<Detection>& dets,
                            float confThresh, float iouThresh);

// Draw bounding boxes on image
cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& dets,
                        const char* const classes[80],
                        float scale, int padLeft, int padTop);

// Draw debug overlay: FPS, timing info, frame counter
// Designed to be called on the display-sized image (e.g. 1280x720)
// so text remains sharp and legible regardless of source video resolution.
void drawDebugOverlay(cv::Mat& image,
                       double preMs, double infMs, double postMs,
                       double smoothFps, double avgFps,
                       double minFps, double maxFps,
                       int frameCount, int numDetections);
