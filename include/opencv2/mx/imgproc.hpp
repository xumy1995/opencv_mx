#pragma once

#include "opencv2/mx/gpumat.hpp"
#include <opencv2/imgproc.hpp>

namespace cv::mx {

void resize(const GpuMat &src, GpuMat &dst, cv::Size dsize,
            double fx = 0, double fy = 0,
            int interpolation = cv::INTER_LINEAR,
            mcStream_t stream = nullptr);

inline void resize(const GpuMat &src, GpuMat &dst, cv::Size dsize,
                   int interpolation, const Stream &stream) {
    resize(src, dst, dsize, 0, 0, interpolation, stream.nativeHandle());
}

} // namespace cv::mx
