#include "opencv2/mx/imgproc.hpp"
#include <opencv2/core.hpp>
#include <iostream>
int main() {
    cv::Mat src(32, 48, CV_8UC3); cv::randu(src, 0, 255);
    cv::mx::GpuMat a, b; a.upload(src); cv::mx::cvtColor(a, b, cv::COLOR_BGR2RGB);
    cv::Mat out; b.download(out); cv::Mat expected; cv::cvtColor(src, expected, cv::COLOR_BGR2RGB);
    if (out.type() != expected.type() || cv::norm(out, expected, cv::NORM_INF) != 0) return 1;
    std::cout << "cvtColor test passed\n"; return 0;
}
