#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "opencv2/mx/imgproc.hpp"

namespace py = pybind11;
using cv::mx::GpuMat;

static cv::Mat asMat(const py::array &a) {
    if (a.ndim() != 2 && a.ndim() != 3 || a.dtype().kind() != 'u' || a.dtype().itemsize() != 1)
        throw std::invalid_argument("expected uint8 HxW or HxWxC array");
    int channels = a.ndim() == 2 ? 1 : static_cast<int>(a.shape(2));
    if (channels != 1 && channels != 3 && channels != 4) throw std::invalid_argument("expected 1, 3 or 4 channels");
    int type = CV_MAKETYPE(CV_8U, channels);
    return cv::Mat(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)), type,
                   const_cast<void *>(a.data()), static_cast<size_t>(a.strides(0)));
}

PYBIND11_MODULE(opencv_mx, m) {
    py::class_<GpuMat, std::shared_ptr<GpuMat>>(m, "GpuMat")
        .def(py::init<>()).def("upload", [](GpuMat &g, py::array a) { g.upload(asMat(a)); })
        .def("download", [](const GpuMat &g) { py::MatAllocator *unused = nullptr; (void)unused; cv::Mat out; g.download(out); return py::array_t<uint8_t>({out.rows, out.cols, CV_MAT_CN(out.type())}, {out.step, static_cast<ssize_t>(out.elemSize()), 1}, out.data); })
        .def_property_readonly("shape", [](const GpuMat &g) { return py::make_tuple(g.rows(), g.cols(), g.channels()); });
    m.def("resize", [](const GpuMat &src, cv::Size size, int interpolation) { auto dst=std::make_shared<GpuMat>(); cv::mx::resize(src,*dst,size,0,0,interpolation); return dst; }, py::arg("src"), py::arg("size"), py::arg("interpolation")=cv::INTER_LINEAR);
    m.def("cvtColor", [](const GpuMat &src, int code) { auto dst=std::make_shared<GpuMat>(); cv::mx::cvtColor(src,*dst,code); return dst; });
}
