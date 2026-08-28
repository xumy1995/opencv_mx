#pragma once

#include <opencv2/core.hpp>
#include <mcr/mc_runtime_api.h>
#include <memory>

namespace cv::mx {

class Stream {
public:
    Stream();
    ~Stream();
    Stream(const Stream&) = delete;
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;
    void waitForCompletion() const;
    bool queryIfComplete() const;
    static mcStream_t Null() noexcept { return nullptr; }
    mcStream_t nativeHandle() const noexcept { return handle_; }
private:
    mcStream_t handle_ = nullptr;
};

class GpuMat {
public:
    GpuMat() = default;
    GpuMat(int rows, int cols, int type) { create(rows, cols, type); }
    ~GpuMat();

    GpuMat(const GpuMat &) = default;
    GpuMat &operator=(const GpuMat &) = default;
    GpuMat(GpuMat &&other) noexcept;
    GpuMat &operator=(GpuMat &&other) noexcept;
    GpuMat(const GpuMat &parent, cv::Rect roi);

    // Wrap device memory owned by the caller. The caller must keep it alive.
    GpuMat(int rows, int cols, int type, void *deviceData, size_t step);

    void create(int rows, int cols, int type);
    void release() noexcept;
    void upload(cv::InputArray src, mcStream_t stream = nullptr);
    void upload(cv::InputArray src, const Stream &stream);
    void download(cv::OutputArray dst, mcStream_t stream = nullptr) const;
    void download(cv::OutputArray dst, const Stream &stream) const;
    GpuMat clone() const;
    void locateROI(cv::Size &wholeSize, cv::Point &ofs) const;

    bool empty() const noexcept { return data_ == nullptr; }
    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    int type() const noexcept { return type_; }
    int channels() const noexcept { return type_ >= 0 ? CV_MAT_CN(type_) : 0; }
    size_t elemSize() const noexcept { return type_ >= 0 ? CV_ELEM_SIZE(type_) : 0; }
    size_t step() const noexcept { return step_; }
    size_t rowBytes() const noexcept { return cols_ > 0 ? cols_ * CV_ELEM_SIZE(type_) : 0; }
    bool isContinuous() const noexcept { return empty() || step_ == rowBytes(); }
    void *data() noexcept { return data_; }
    const void *data() const noexcept { return data_; }

private:
    int rows_ = 0;
    int cols_ = 0;
    int type_ = -1;
    size_t step_ = 0;
    void *data_ = nullptr;
    std::shared_ptr<void> owner_;
    cv::Size wholeSize_;
    cv::Point offset_;
};

} // namespace cv::mx
