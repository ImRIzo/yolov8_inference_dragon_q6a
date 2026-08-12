#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct PreprocessResult {
    std::vector<float> data;   // NCHW float32 [0,1]
    cv::Mat original;          // original BGR image
    float scale;
    int padLeft, padTop;
};

// Preprocess from file path
PreprocessResult preprocess(const std::string& imagePath);

// Preprocess from in-memory BGR frame (video)
PreprocessResult preprocess(const cv::Mat& bgrFrame);
