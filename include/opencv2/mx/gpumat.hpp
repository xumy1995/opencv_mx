#pragma once

#include <opencv2/core.hpp>
#include <mcr/mc_runtime_api.h>

namespace cv::mx {

class GpuMat {
public:
    GpuMat() = default;
    GpuMat(int rows, int cols, int type) { create(rows, cols, type); }
    ~GpuMat();

    GpuMat(const GpuMat &) = delete;
    GpuMat &operator=(const GpuMat &) = delete;
    GpuMat(GpuMat &&other) noexcept;
    GpuMat &operator=(GpuMat &&other) noexcept;

    void create(int rows, int cols, int type);
    void release() noexcept;
    void upload(cv::InputArray src, mcStream_t stream = nullptr);
    void download(cv::OutputArray dst, mcStream_t stream = nullptr) const;

    bool empty() const noexcept { return data_ == nullptr; }
    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    int type() const noexcept { return type_; }
    size_t step() const noexcept { return step_; }
    void *data() noexcept { return data_; }
    const void *data() const noexcept { return data_; }

private:
    int rows_ = 0;
    int cols_ = 0;
    int type_ = -1;
    size_t step_ = 0;
    void *data_ = nullptr;
};

} // namespace cv::mx
