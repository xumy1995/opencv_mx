#include "opencv2/mx/imgproc.hpp"

#include <cvcuda/OpResize.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorData.h>
#include <stdexcept>

namespace {
void check(NVCVStatus status, const char *operation) {
    if (status != NVCV_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(status));
}

NVCVDataType dtypeFor(int cvType) {
    switch (cvType) {
    case CV_8UC1: return NVCV_DATA_TYPE_U8;
    // Channels are represented by HWC shape/layout. Using 3U8/4U8 here would
    // encode the channel count twice and makes the tensor descriptor invalid.
    case CV_8UC3: return NVCV_DATA_TYPE_U8;
    case CV_8UC4: return NVCV_DATA_TYPE_U8;
    default: throw std::invalid_argument("cv::mx::resize currently supports CV_8UC1/3/4");
    }
}

NVCVInterpolationType interpolationFor(int interpolation) {
    switch (interpolation) {
    case cv::INTER_NEAREST: return NVCV_INTERP_NEAREST;
    case cv::INTER_LINEAR: return NVCV_INTERP_LINEAR;
    case cv::INTER_CUBIC: return NVCV_INTERP_CUBIC;
    case cv::INTER_AREA: return NVCV_INTERP_AREA;
    default: throw std::invalid_argument("unsupported resize interpolation");
    }
}

class TensorHandle {
public:
    explicit TensorHandle(const cv::mx::GpuMat &mat) {
        NVCVTensorData data{};
        data.dtype = dtypeFor(mat.type());
        const int channels = CV_MAT_CN(mat.type());
        // Resize accepts HWC/NHWC tensors. Keep a singleton channel dimension
        // for grayscale images (HW is not a valid resize input in this API).
        data.layout = NVCV_TENSOR_HWC;
        data.rank = 3;
        data.shape[0] = mat.rows();
        data.shape[1] = mat.cols();
        data.shape[2] = channels;
        data.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        data.buffer.strided.basePtr = const_cast<NVCVByte *>(static_cast<const NVCVByte *>(mat.data()));
        data.buffer.strided.strides[0] = mat.step();
        data.buffer.strided.strides[1] = CV_ELEM_SIZE(mat.type());
        data.buffer.strided.strides[2] = CV_ELEM_SIZE1(mat.type());
        check(nvcvTensorWrapDataConstruct(&data, nullptr, nullptr, &handle_), "nvcvTensorWrapDataConstruct");
    }
    ~TensorHandle() { if (handle_) nvcvTensorDecRef(handle_, nullptr); }
    operator NVCVTensorHandle() const { return handle_; }
private:
    NVCVTensorHandle handle_{};
};
}

namespace cv::mx {

void resize(const GpuMat &src, GpuMat &dst, cv::Size dsize,
            double fx, double fy, int interpolation, mcStream_t stream) {
    CV_Assert(!src.empty());
    if (dsize.empty()) {
        CV_Assert(fx > 0 && fy > 0);
        dsize = cv::Size(cvRound(src.cols() * fx), cvRound(src.rows() * fy));
    }
    dst.create(dsize.height, dsize.width, src.type());
    TensorHandle srcTensor(src), dstTensor(dst);
    NVCVOperatorHandle op{};
    check(cvcudaResizeCreate(&op), "cvcudaResizeCreate");
    try {
        check(cvcudaResizeSubmit(op, stream, srcTensor, dstTensor,
                                 interpolationFor(interpolation)), "cvcudaResizeSubmit");
    } catch (...) {
        nvcvOperatorDestroy(op);
        throw;
    }
    nvcvOperatorDestroy(op);
}

} // namespace cv::mx
