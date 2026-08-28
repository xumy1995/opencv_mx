#include "opencv2/mx/gpumat.hpp"

#include <stdexcept>
#include <utility>

namespace {
void check(mcError_t status, const char *operation) {
    if (status != mcSuccess)
        throw std::runtime_error(std::string(operation) + ": " + mcGetErrorString(status));
}
}

namespace cv::mx {

GpuMat::~GpuMat() { release(); }

GpuMat::GpuMat(GpuMat &&o) noexcept { *this = std::move(o); }

GpuMat &GpuMat::operator=(GpuMat &&o) noexcept {
    if (this != &o) {
        release();
        rows_ = std::exchange(o.rows_, 0);
        cols_ = std::exchange(o.cols_, 0);
        type_ = std::exchange(o.type_, -1);
        step_ = std::exchange(o.step_, 0);
        data_ = std::exchange(o.data_, nullptr);
    }
    return *this;
}

void GpuMat::create(int rows, int cols, int type) {
    CV_Assert(rows > 0 && cols > 0);
    if (rows_ == rows && cols_ == cols && type_ == type && data_) return;
    release();
    rows_ = rows;
    cols_ = cols;
    type_ = type;
    check(mcMallocPitch(&data_, &step_, cols * CV_ELEM_SIZE(type), rows), "mcMallocPitch");
}

void GpuMat::release() noexcept {
    if (data_) mcFree(data_);
    data_ = nullptr;
    rows_ = cols_ = 0;
    type_ = -1;
    step_ = 0;
}

void GpuMat::upload(cv::InputArray input, mcStream_t stream) {
    cv::Mat src = input.getMat();
    create(src.rows, src.cols, src.type());
    check(mcMemcpy2DAsync(data_, step_, src.data, src.step,
                            src.cols * src.elemSize(), src.rows,
                            mcMemcpyHostToDevice, stream), "mcMemcpy2DAsync H2D");
    if (!stream) check(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

void GpuMat::download(cv::OutputArray output, mcStream_t stream) const {
    CV_Assert(!empty());
    output.create(rows_, cols_, type_);
    cv::Mat dst = output.getMat();
    check(mcMemcpy2DAsync(dst.data, dst.step, data_, step_,
                            cols_ * dst.elemSize(), rows_,
                            mcMemcpyDeviceToHost, stream), "mcMemcpy2DAsync D2H");
    check(stream ? mcStreamSynchronize(stream) : mcDeviceSynchronize(), "download synchronize");
}

} // namespace cv::mx
