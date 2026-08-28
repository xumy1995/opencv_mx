#include "opencv2/mx/imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    const std::string testdata = argc > 1 ? argv[1] : "testdata";
    const std::string golden = testdata + "/golden/maca3.8.0.10-cvcuda0.16.0";

    const char *types[] = {"gray", "bgr", "bgra"};
    const int interpolations[] = {
        cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA};

    for (const char *type : types)
    {
        const std::string inputPath = testdata + "/input_" + type + ".png";
        const cv::Mat input = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
        if (input.empty())
        {
            std::cerr << "cannot read " << inputPath << '\n';
            return 1;
        }

        for (int index = 0; index < 4; ++index)
        {
            const std::string expectedPath = golden + "/" + type + "_"
                                           + std::to_string(index) + ".png";
            const cv::Mat expected = cv::imread(expectedPath, cv::IMREAD_UNCHANGED);
            if (expected.empty())
            {
                std::cerr << "cannot read " << expectedPath << '\n';
                return 1;
            }

            cv::mx::GpuMat gpuInput, gpuOutput;
            gpuInput.upload(input);
            cv::mx::resize(gpuInput, gpuOutput, expected.size(), 0, 0,
                           interpolations[index]);

            cv::Mat actual;
            gpuOutput.download(actual);

            if (actual.size() != expected.size() || actual.type() != expected.type())
            {
                std::cerr << "shape/type mismatch for " << type
                          << " interpolation=" << index << '\n';
                return 1;
            }

            const double error = cv::norm(expected, actual, cv::NORM_INF);
            if (error != 0)
            {
                std::cerr << "golden mismatch for " << type
                          << " interpolation=" << index
                          << " max_error=" << error << '\n';
                return 1;
            }
        }
    }

    std::cout << "12 MX CV-CUDA resize golden cases passed\n";
    return 0;
}
