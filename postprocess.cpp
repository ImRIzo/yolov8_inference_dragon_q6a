#include "postprocess.hpp"

const char* COCO_CLASSES[80] = {
    "ganoderma",
};

std::vector<Detection> parseOutputs(const uint8_t* boxData, const uint8_t* scrData,
                                     const uint32_t boxDims[4], const uint32_t scrDims[4],
                                     float boxScale, int32_t boxOffset,
                                     float scrScale, int32_t scrOffset,
                                     float confThresh) {
    std::vector<Detection> detections;
    int NA = (int)scrDims[2];
    int NC = (int)scrDims[1];

    size_t boxCount = 1;
    for (int d = 0; d < 4 && boxDims[d] > 0; d++) boxCount *= boxDims[d];
    std::vector<float> boxesFloat(boxCount);
    for (size_t i = 0; i < boxCount; i++)
        boxesFloat[i] = (float)((int32_t)boxData[i] + boxOffset) * boxScale;

    size_t scrCount = 1;
    for (int d = 0; d < 4 && scrDims[d] > 0; d++) scrCount *= scrDims[d];
    std::vector<float> scoresFloat(scrCount);
    for (size_t i = 0; i < scrCount; i++)
        scoresFloat[i] = (float)((int32_t)scrData[i] + scrOffset) * scrScale;

    for (int a = 0; a < NA; a++) {
        float maxScore = -1.0f;
        int maxClass = -1;
        for (int c = 0; c < NC; c++) {
            float s = scoresFloat[c * NA + a];
            if (s > maxScore) { maxScore = s; maxClass = c; }
        }
        if (maxScore >= confThresh) {
            float cx = boxesFloat[0 * NA + a];
            float cy = boxesFloat[1 * NA + a];
            float w  = boxesFloat[2 * NA + a];
            float h  = boxesFloat[3 * NA + a];
            detections.push_back({maxClass, maxScore,
                                  cx - w / 2, cy - h / 2,
                                  cx + w / 2, cy + h / 2});
        }
    }
    return detections;
}

std::vector<Detection> nms(const std::vector<Detection>& dets,
                            float confThresh, float iouThresh) {
    if (dets.empty()) return {};
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    for (auto& d : dets) {
        boxes.push_back(cv::Rect2d(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
        scores.push_back(d.confidence);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThresh, iouThresh, keep);
    std::vector<Detection> result;
    for (int i : keep) result.push_back(dets[i]);
    return result;
}

cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& dets,
                        const char* const classes[80],
                        float scale, int padLeft, int padTop) {
    // Draw directly on a copy of the frame
    cv::Mat result = image.clone();
    for (auto& d : dets) {
        const char* label = (d.classId >= 0 && d.classId < 80) ? classes[d.classId] : "?";
        int x1 = (int)((d.x1 - padLeft) / scale);
        int y1 = (int)((d.y1 - padTop) / scale);
        int x2 = (int)((d.x2 - padLeft) / scale);
        int y2 = (int)((d.y2 - padTop) / scale);
        cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        char text[128];
        snprintf(text, sizeof(text), "%.2f %s", d.confidence, label);
        cv::putText(result, text, cv::Point(x1, y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    return result;
}

void drawDebugOverlay(cv::Mat& image,
                       double preMs, double infMs, double postMs,
                       double smoothFps, double avgFps,
                       double minFps, double maxFps,
                       int frameCount, int numDetections) {
    double totalMs = preMs + infMs + postMs;
    double pipelineFps = (totalMs > 0) ? (1000.0 / totalMs) : 0;

    // Compact, sharp overlay — sized for a 1280×720 display window
    const int font = cv::FONT_HERSHEY_DUPLEX;
    const double fontScale = 0.55;
    const int thick = 1;
    const int lineH = 20;
    const int padX = 8;
    const int padY = 6;
    int y = padY + lineH;

    const cv::Scalar yellow(0, 255, 255);
    const cv::Scalar green(0, 255, 0);
    const cv::Scalar grey(180, 180, 180);

    auto put = [&](const char* text, const cv::Scalar& col) {
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, font, fontScale, thick, &baseline);
        // Semi-transparent strip per line
        int stripY = y - ts.height - baseline;
        cv::Rect bgRect(padX - 2, stripY - 2, ts.width + 8, ts.height + baseline + 5);
        cv::Rect clip = bgRect & cv::Rect(0, 0, image.cols, image.rows);
        cv::Mat roi = image(clip);
        if (!roi.empty()) {
            cv::Mat overlay;
            roi.copyTo(overlay);
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(roi.cols, roi.rows),
                          cv::Scalar(20, 20, 30), cv::FILLED);
            cv::addWeighted(overlay, 0.50, roi, 0.50, 0, roi);
        }
        cv::putText(image, text, cv::Point(padX, y), font, fontScale, col, thick, cv::LINE_AA);
        y += lineH;
    };

    char buf[180];

    // ── FPS ────────────────────────────────────────────────────────────────
    auto fpsCol = (smoothFps > 25.0) ? green : yellow;
    snprintf(buf, sizeof(buf), "FPS:%5.1f  avg:%5.1f  min:%5.1f  max:%5.1f  #%d det:%d",
             smoothFps, avgFps, minFps, maxFps, frameCount, numDetections);
    put(buf, fpsCol);

    // ── Pipeline timing ────────────────────────────────────────────────────
    snprintf(buf, sizeof(buf), "pipe: %.1f fps (%.1f ms)  pre:%.1f  inf:%.1f  post:%.1f ms",
             pipelineFps, totalMs, preMs, infMs, postMs);
    put(buf, yellow);

    // ── Hardware ───────────────────────────────────────────────────────────
    put("NPU: QCS6490 HTP | YOLOv8n  |  ESC/Q quit", grey);
}
