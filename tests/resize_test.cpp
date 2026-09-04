#include "opencv2/mx/imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct ResizeMode { int channels; int width; int height; };

cv::Mat makeInput()
{
    const int rows = 768, cols = 1024;
    cv::Mat image(rows, cols, CV_8UC1);
    cv::RNG rng(0x123456789abcdefULL);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double value;
            if (y < rows / 2) {
                value = x < cols / 2
                    ? (std::sin((x + 1) * CV_PI / 256.) *
                       std::sin((y + 1) * CV_PI / 256.) *
                       std::sin(7 * CV_PI / 8) + 1.) * 128.
                    : ((x / 128 + y / 128) % 2) * 250 + (y / 128) % 2;
            } else if (x < cols / 2) {
                value = (x / 128) * (85 - y / 256 * 40) * (y / 128 % 2) +
                        (7 - x / 128) * (85 - y / 256 * 40) *
                        ((y / 128 + 1) % 2);
            } else {
                value = static_cast<uchar>(rng);
            }
            image.at<uchar>(y, x) = static_cast<uchar>(value);
        }
    }
    return image;
}

cv::Mat makeChannels(const cv::Mat& gray, int channels)
{
    if (channels == 1) return gray;
    std::vector<cv::Mat> planes(channels, gray);
    cv::Mat result;
    cv::merge(planes, result);
    return result;
}

bool runLinearCase(const cv::Mat& source, const ResizeMode& mode, size_t index)
{
    cv::Mat reference;
    cv::resize(source, reference, cv::Size(mode.width, mode.height), 0, 0,
               cv::INTER_LINEAR_EXACT);
    cv::mx::GpuMat input, output;
    input.upload(source);
    cv::mx::resize(input, output, reference.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat actual;
    output.download(actual);
    const double error = cv::norm(reference, actual, cv::NORM_INF);
    std::cout << "case " << index << " CV_8UC" << mode.channels << " -> "
              << mode.width << 'x' << mode.height << " max_error=" << error << '\n';
    return error <= 1.0;
}

bool runNearestExactCases()
{
    const cv::Mat sources[] = {
        (cv::Mat_<uchar>(1, 6) << 0, 1, 2, 3, 4, 5),
        (cv::Mat_<uchar>(1, 5) << 0, 1, 2, 3, 4),
        (cv::Mat_<uchar>(1, 5) << 0, 1, 2, 3, 4),
        (cv::Mat_<uchar>(1, 5) << 0, 1, 2, 3, 4),
        (cv::Mat_<uchar>(3, 5) << 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14),
        (cv::Mat_<uchar>(2, 3) << 0,1,2, 3,4,5)};
    const cv::Mat expected[] = {
        (cv::Mat_<uchar>(1, 3) << 1,3,5), (cv::Mat_<uchar>(1,1) << 2),
        (cv::Mat_<uchar>(1, 3) << 0,2,4), (cv::Mat_<uchar>(1,2) << 1,3),
        (cv::Mat_<uchar>(5,7) << 0,1,1,2,3,3,4, 0,1,1,2,3,3,4, 5,6,6,7,8,8,9, 10,11,11,12,13,13,14, 10,11,11,12,13,13,14),
        (cv::Mat_<uchar>(4,6) << 0,0,1,1,2,2, 0,0,1,1,2,2, 3,3,4,4,5,5, 3,3,4,4,5,5)};
    for (int i = 0; i < 6; ++i) {
        cv::mx::GpuMat input, output; input.upload(sources[i]);
        cv::mx::resize(input, output, expected[i].size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat actual; output.download(actual);
        if (cv::norm(actual, expected[i], cv::NORM_INF) > 1.0) return false;
    }
    return true;
}

double compareNearestConventions()
{
    cv::Mat source(37, 53, CV_8UC3), cpuExact, actual;
    cv::randu(source, 0, 255);
    cv::resize(source, cpuExact, cv::Size(), 2, 1, cv::INTER_NEAREST_EXACT);
    cv::mx::GpuMat input, output; input.upload(source);
    cv::mx::resize(input, output, cpuExact.size(), 0, 0, cv::INTER_NEAREST);
    output.download(actual);
    return cv::norm(cpuExact, actual, cv::NORM_INF);
}

bool runLenaNearestTest()
{
    cv::Mat color = cv::imread("testdata/shared/lena.png", cv::IMREAD_COLOR);
    if (color.empty()) { std::cerr << "cannot read testdata/shared/lena.png\n"; return false; }
    cv::Mat gray; cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    for (const cv::Mat& source : {color, gray}) {
        cv::Mat exact; cv::resize(source, exact, cv::Size(), 2, 1, cv::INTER_NEAREST_EXACT);
        cv::mx::GpuMat input, output; input.upload(source);
        cv::mx::resize(input, output, exact.size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat actual; output.download(actual);
        const double error = cv::norm(exact, actual, cv::NORM_INF);
        std::cout << "Lena " << (source.channels() == 1 ? "gray" : "color")
                  << " nearest exact max_error=" << error << '\n';
        if (error > 1.0) return false;
    }
    return true;
}

} // namespace

int main()
{
    const ResizeMode modes[] = {
        {1,512,768},{3,512,768},{1,1024,384},{4,1024,384},
        {1,512,384},{2,512,384},{3,512,384},{4,512,384},
        {1,256,192},{2,256,192},{3,256,192},{4,256,192},
        {1,4,3},{2,4,3},{3,4,3},{4,4,3},{1,342,384},{1,342,256},
        {2,342,256},{3,342,256},{4,342,256},{1,512,256},{1,146,110},
        {3,146,110},{4,146,110},{1,931,698},{2,931,698},{3,931,698},
        {4,931,698},{1,853,640},{3,853,640},{4,853,640},
        {1,1004,753},{2,1004,753},{3,1004,753},{4,1004,753},
        {1,2048,1536},{2,2048,1536},{4,2048,1536},
        {1,3072,2304},{3,3072,2304},{1,7168,5376}};
    const cv::Mat gray = makeInput();
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        if (modes[i].channels == 2) continue;
        if (!runLinearCase(makeChannels(gray, modes[i].channels), modes[i], i)) return 1;
    }
    if (!runNearestExactCases()) return 1;
    std::cout << "NearestExact fixed cases passed (max error <= 1)\n";
    std::cout << "Nearest vs NearestExact max error=" << compareNearestConventions() << "\n";
    if (!runLenaNearestTest()) return 1;
    std::cout << "All OpenCV Resize_Bitexact Linear8U + NearestExact cases passed\n";
    return 0;
}
