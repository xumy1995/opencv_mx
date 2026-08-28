#include "opencv2/mx/imgproc.hpp"
#include <cvcuda/OpCvtColor.h>
#include <nvcv/TensorData.h>
#include <stdexcept>
#include <memory>

namespace {
void ncheck(NVCVStatus s, const char *op) {
    if (s != NVCV_SUCCESS) throw std::runtime_error(std::string(op) + " failed with status " + std::to_string(s));
}
void mcheck(mcError_t s, const char *op) {
    if (s != mcSuccess) throw std::runtime_error(std::string(op) + ": " + mcGetErrorString(s));
}
class Tensor {
public:
    explicit Tensor(const cv::mx::GpuMat &m) {
        NVCVTensorData d{}; d.dtype = NVCV_DATA_TYPE_U8; d.layout = NVCV_TENSOR_HWC;
        d.rank = 3; d.shape[0] = m.rows(); d.shape[1] = m.cols(); d.shape[2] = CV_MAT_CN(m.type());
        d.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        d.buffer.strided.basePtr = const_cast<NVCVByte *>(static_cast<const NVCVByte *>(m.data()));
        d.buffer.strided.strides[0] = m.step(); d.buffer.strided.strides[1] = CV_ELEM_SIZE(m.type());
        d.buffer.strided.strides[2] = CV_ELEM_SIZE1(m.type());
        ncheck(nvcvTensorWrapDataConstruct(&d, nullptr, nullptr, &h_), "nvcvTensorWrapDataConstruct");
    }
    ~Tensor() { if (h_) nvcvTensorDecRef(h_, nullptr); }
    operator NVCVTensorHandle() const { return h_; }
private: NVCVTensorHandle h_{};
};
struct Resources { Tensor in, out; NVCVOperatorHandle op{}; Resources(const cv::mx::GpuMat&a,const cv::mx::GpuMat&b):in(a),out(b){ncheck(cvcudaCvtColorCreate(&op),"cvcudaCvtColorCreate");} ~Resources(){if(op)nvcvOperatorDestroy(op);} };
void MC_CB destroy(void *p) { delete static_cast<Resources *>(p); }
int outputType(int code) {
    switch (code) {
    case cv::COLOR_BGR2GRAY: case cv::COLOR_BGRA2GRAY: case cv::COLOR_RGB2GRAY: case cv::COLOR_RGBA2GRAY: return CV_8UC1;
    case cv::COLOR_GRAY2BGR: return CV_8UC3;
    case cv::COLOR_GRAY2BGRA: return CV_8UC4;
    default: return CV_8UC3;
    }
}
}
namespace cv::mx {
void cvtColor(const GpuMat &src, GpuMat &dst, int code, mcStream_t stream) {
    CV_Assert(!src.empty());
    dst.create(src.rows(), src.cols(), outputType(code));
    auto r = std::make_unique<Resources>(src, dst);
    ncheck(cvcudaCvtColorSubmit(r->op, stream, r->in, r->out,
                                static_cast<NVCVColorConversionCode>(code)),
           "cvcudaCvtColorSubmit");
    if (stream) { mcheck(mcLaunchHostFunc(stream, destroy, r.get()), "mcLaunchHostFunc"); r.release(); }
    else { mcheck(mcDeviceSynchronize(), "cvtColor synchronize"); }
}
}
