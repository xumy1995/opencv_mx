#include "opencv2/mx/gpumat.hpp"
#include <opencv2/core.hpp>
#include <iostream>

int main() {
    cv::Mat input(17, 23, CV_8UC3);
    cv::randu(input, 0, 255);
    cv::mx::GpuMat a;
    a.create(input.rows, input.cols, input.type());
    if (a.empty() || a.rows() != 17 || a.cols() != 23 || a.channels() != 3)
        return 1;
    void *original = a.data();
    a.create(input.rows, input.cols, input.type());
    if (a.data() != original)
        return 2;
    a.upload(input);
    cv::mx::GpuMat copy = a;
    cv::Size whole;
    cv::Point offset;
    cv::mx::GpuMat roi(copy, cv::Rect(3, 4, 10, 8));
    roi.locateROI(whole, offset);
    if (whole != cv::Size(23, 17) || offset != cv::Point(3, 4))
        return 3;
    cv::Mat output;
    roi.download(output);
    if (output.size() != cv::Size(10, 8) || output.type() != CV_8UC3)
        return 4;
    cv::mx::Stream stream;
    cv::mx::GpuMat async;
    async.upload(input, stream);
    stream.waitForCompletion();
    async.download(output, stream);
    stream.waitForCompletion();
    if (cv::norm(input, output, cv::NORM_INF) != 0)
        return 5;
    std::cout << "GpuMat tests passed\n";
    return 0;
}
