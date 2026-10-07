#include "preprocess.hpp"

static constexpr int IMAGE_SIZE = 640;

static PreprocessResult preprocessImpl(const cv::Mat& bgr) {
    PreprocessResult r;
    r.original = bgr;  // no clone, reference only (valid for video frame lifetime)
    int h = bgr.rows, w = bgr.cols;
    float sc = std::min((float)IMAGE_SIZE / w, (float)IMAGE_SIZE / h);
    int nw = (int)(w * sc), nh = (int)(h * sc);

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    int dw = IMAGE_SIZE - nw, dh = IMAGE_SIZE - nh;
    r.padLeft = dw / 2;
    r.padTop  = dh / 2;
    r.resizedW = nw;   // exact resize target size — used for precise
    r.resizedH = nh;   // letterbox reversal in evaluation mode

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, r.padTop, dh - r.padTop,
                       r.padLeft, dw - r.padLeft,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    r.data.resize(3 * IMAGE_SIZE * IMAGE_SIZE);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < IMAGE_SIZE; y++)
            for (int x = 0; x < IMAGE_SIZE; x++)
                r.data[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                    padded.at<cv::Vec3b>(y, x)[c] / 255.0f;
    r.scale = sc;
    return r;
}

PreprocessResult preprocess(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        fprintf(stderr, "ERROR: cannot read %s\n", path.c_str());
        return {};
    }
    auto r = preprocessImpl(img);
    printf("  %dx%d letterbox -> %dx%d  scale=%.4f  pad=(%d,%d)\n",
           img.rows, img.cols, IMAGE_SIZE, IMAGE_SIZE, r.scale, r.padLeft, r.padTop);
    return r;
}

PreprocessResult preprocess(const cv::Mat& bgrFrame) {
    return preprocessImpl(bgrFrame);
}
