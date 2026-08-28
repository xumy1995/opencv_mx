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

Stream::Stream() { check(mcStreamCreate(&handle_), "mcStreamCreate"); }
Stream::~Stream() { if (handle_) mcStreamDestroy(handle_); }
Stream::Stream(Stream&& o) noexcept : handle_(std::exchange(o.handle_, nullptr)) {}
Stream& Stream::operator=(Stream&& o) noexcept { if (this != &o) { if (handle_) mcStreamDestroy(handle_); handle_=std::exchange(o.handle_, nullptr); } return *this; }
void Stream::waitForCompletion() const { check(mcStreamSynchronize(handle_), "mcStreamSynchronize"); }

GpuMat::~GpuMat() = default;

GpuMat::GpuMat(GpuMat &&o) noexcept { *this = std::move(o); }

GpuMat &GpuMat::operator=(GpuMat &&o) noexcept {
    if (this != &o) {
        release();
        rows_ = std::exchange(o.rows_, 0);
        cols_ = std::exchange(o.cols_, 0);
        type_ = std::exchange(o.type_, -1);
        step_ = std::exchange(o.step_, 0);
        data_ = std::exchange(o.data_, nullptr);
        owner_ = std::move(o.owner_);
    }
    return *this;
}

GpuMat::GpuMat(const GpuMat &p, cv::Rect roi) : rows_(roi.height), cols_(roi.width), type_(p.type_), step_(p.step_), owner_(p.owner_) {
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= p.cols_ && roi.y + roi.height <= p.rows_);
    data_ = static_cast<unsigned char*>(p.data_) + roi.y * p.step_ + roi.x * CV_ELEM_SIZE(type_);
}

void GpuMat::create(int rows, int cols, int type) {
    CV_Assert(rows > 0 && cols > 0);
    if (rows_ == rows && cols_ == cols && type_ == type && data_) return;
    release();
    rows_ = rows;
    cols_ = cols;
    type_ = type;
    check(mcMallocPitch(&data_, &step_, cols * CV_ELEM_SIZE(type), rows), "mcMallocPitch");
    owner_ = std::shared_ptr<void>(data_, [](void *p){ if (p) mcFree(p); });
}

void GpuMat::release() noexcept {
    owner_.reset();
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
