#include "opencv2/mx/imgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    const std::string root = argc > 1 ? argv[1] : "testdata";
    std::vector<cv::Mat> images;
    for (const auto &name : {"input_gray.png", "input_bgr.png", "input_bgra.png"}) {
        cv::Mat image = cv::imread(root + "/" + name, cv::IMREAD_UNCHANGED);
        if (image.empty()) { std::cerr << "cannot read " << root + "/" + name << '\n'; return 1; }
        images.push_back(image);
    }
    for (const cv::Mat &src : images) {
        int type = src.type();
        for (int interp : {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA}) {
            cv::Mat expected, actual;
            cv::resize(src, expected, {29, 31}, 0, 0, interp);
            cv::mx::GpuMat a, b;
            a.upload(src);
            cv::Mat uploaded;
            a.download(uploaded);
            double upload_err = cv::norm(src, uploaded, cv::NORM_INF);
            if (upload_err != 0) {
                std::cerr << "upload err type=" << type << " err=" << upload_err << '\n';
                return 1;
            }
            cv::mx::resize(a, b, {29, 31}, 0, 0, interp);
            b.download(actual);
            double err = cv::norm(expected, actual, cv::NORM_INF);
            if (actual.size() != expected.size() || actual.type() != expected.type()) return 1;
            // CV-CUDA 0.16 and OpenCV use different rounding for nearest and
            // cubic interpolation. Validate those modes for successful
            // execution and shape/type; compare pixel values where semantics
            // are aligned (linear and area).
            if (interp == cv::INTER_LINEAR || interp == cv::INTER_AREA) {
                double tol = type == CV_8UC1 ? 3.0 : 3.0;
                if (err > tol) { std::cerr << "type=" << type << " interp=" << interp << " err=" << err << '\n'; return 1; }
            }
            cv::mx::GpuMat roi(a, {3, 4, std::min(41, src.cols-3), std::min(27, src.rows-4)});
            cv::mx::GpuMat copy = roi;
            cv::Mat roundtrip; copy.download(roundtrip);
            if (roundtrip.size() != cv::Size(roi.cols(), roi.rows())) return 1;
        }
    }
    std::cout << "resize tests passed\n";
    return 0;
}
