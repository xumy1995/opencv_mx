#include "opencv2/mx/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: mx_resize_example INPUT OUTPUT\n";
        return 2;
    }
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "cannot read " << argv[1] << '\n';
        return 1;
    }
    cv::mx::GpuMat mxSrc, mxDst;
    mxSrc.upload(src);
    cv::mx::resize(mxSrc, mxDst, cv::Size(src.cols / 2, src.rows / 2));
    cv::Mat dst;
    mxDst.download(dst);
    if (!cv::imwrite(argv[2], dst)) return 1;
    std::cout << src.cols << 'x' << src.rows << " -> "
              << dst.cols << 'x' << dst.rows << '\n';
}
