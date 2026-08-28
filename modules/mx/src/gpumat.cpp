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
bool Stream::queryIfComplete() const {
    mcError_t status = mcStreamQuery(handle_);
    if (status == mcSuccess) return true;
    if (status == mcErrorNotReady) return false;
    check(status, "mcStreamQuery");
    return false;
}

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
        wholeSize_ = std::exchange(o.wholeSize_, {});
        offset_ = std::exchange(o.offset_, {});
    }
    return *this;
}

GpuMat::GpuMat(const GpuMat &p, cv::Rect roi)
    : rows_(roi.height), cols_(roi.width), type_(p.type_), step_(p.step_),
      owner_(p.owner_), wholeSize_(p.wholeSize_), offset_(p.offset_ + roi.tl()) {
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= p.cols_ && roi.y + roi.height <= p.rows_);
    data_ = static_cast<unsigned char*>(p.data_) + roi.y * p.step_ + roi.x * CV_ELEM_SIZE(type_);
}

GpuMat::GpuMat(int rows, int cols, int type, void *deviceData, size_t step)
    : rows_(rows), cols_(cols), type_(type), step_(step), data_(deviceData),
      wholeSize_(cols, rows) {
    CV_Assert(rows > 0 && cols > 0 && deviceData != nullptr);
    CV_Assert(step >= cols * CV_ELEM_SIZE(type));
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
    wholeSize_ = cv::Size(cols, rows);
    offset_ = {};
}

void GpuMat::release() noexcept {
    owner_.reset();
    data_ = nullptr;
    rows_ = cols_ = 0;
    type_ = -1;
    step_ = 0;
    wholeSize_ = {};
    offset_ = {};
}

void GpuMat::upload(cv::InputArray src, const Stream &stream) {
    upload(src, stream.nativeHandle());
}

void GpuMat::upload(cv::InputArray input, mcStream_t stream) {
    cv::Mat src = input.getMat();
    create(src.rows, src.cols, src.type());
    check(mcMemcpy2DAsync(data_, step_, src.data, src.step,
                            src.cols * src.elemSize(), src.rows,
                            mcMemcpyHostToDevice, stream), "mcMemcpy2DAsync H2D");
    if (!stream) check(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

void GpuMat::download(cv::OutputArray dst, const Stream &stream) const {
    download(dst, stream.nativeHandle());
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

GpuMat GpuMat::clone() const {
    GpuMat out(rows_, cols_, type_);
    check(mcMemcpy2D(out.data_, out.step_, data_, step_, rowBytes(), rows_,
                     mcMemcpyDeviceToDevice), "mcMemcpy2D D2D");
    check(mcDeviceSynchronize(), "mcDeviceSynchronize clone");
    return out;
}

void GpuMat::locateROI(cv::Size &wholeSize, cv::Point &ofs) const {
    wholeSize = wholeSize_;
    ofs = offset_;
}

} // namespace cv::mx
