#include "opencv2/mx/imgproc.hpp"
#include <opencv2/core.hpp>
#include <iostream>

int main() {
    for (int type : {CV_8UC3, CV_8UC4}) {
        cv::Mat src(37, 53, type);
        cv::randu(src, 0, 255);
        for (int interp : {cv::INTER_LINEAR, cv::INTER_AREA}) {
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
            if (err > 3.0) { std::cerr << "type=" << type << " interp=" << interp << " err=" << err << '\n'; return 1; }
            cv::mx::GpuMat roi(a, {3, 4, 41, 27});
            cv::mx::GpuMat copy = roi;
            cv::Mat roundtrip; copy.download(roundtrip);
            if (roundtrip.size() != cv::Size(41,27)) return 1;
        }
    }
    std::cout << "resize tests passed\n";
    return 0;
}
