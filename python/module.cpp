#include <cstring>
#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "opencv2/mx/imgproc.hpp"

namespace py = pybind11;
using cv::mx::GpuMat;

namespace {

cv::Mat asMat(const py::array &array)
{
    if ((array.ndim() != 2 && array.ndim() != 3)
        || array.dtype().kind() != 'u' || array.dtype().itemsize() != 1)
    {
        throw std::invalid_argument("expected uint8 HxW or HxWxC array");
    }

    const int channels = array.ndim() == 2
        ? 1 : static_cast<int>(array.shape(2));
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::invalid_argument("expected 1, 3 or 4 channels");

    const int type = CV_MAKETYPE(CV_8U, channels);
    return cv::Mat(static_cast<int>(array.shape(0)),
                   static_cast<int>(array.shape(1)), type,
                   const_cast<void *>(array.data()),
                   static_cast<size_t>(array.strides(0)));
}

py::array download(const GpuMat &gpu)
{
    cv::Mat output;
    gpu.download(output);

    const int channels = CV_MAT_CN(output.type());
    py::array_t<uint8_t> result({output.rows, output.cols, channels});
    std::memcpy(result.mutable_data(), output.data,
                output.total() * output.elemSize());
    return result;
}

} // namespace

PYBIND11_MODULE(opencv_mx, module)
{
    py::class_<GpuMat, std::shared_ptr<GpuMat>>(module, "GpuMat")
        .def(py::init<>())
        .def("upload", [](GpuMat &gpu, const py::array &array) {
            gpu.upload(asMat(array));
        })
        .def("download", &download)
        .def_property_readonly("shape", [](const GpuMat &gpu) {
            return py::make_tuple(gpu.rows(), gpu.cols(), gpu.channels());
        });

    module.def("resize", [](const GpuMat &src, py::tuple size,
                             int interpolation) {
        if (size.size() != 2)
            throw std::invalid_argument("size must be (width, height)");
        const cv::Size outputSize(size[0].cast<int>(), size[1].cast<int>());
        auto dst = std::make_shared<GpuMat>();
        cv::mx::resize(src, *dst, outputSize, 0, 0, interpolation);
        return dst;
    }, py::arg("src"), py::arg("size"),
       py::arg("interpolation") = static_cast<int>(cv::INTER_LINEAR));

    module.def("cvtColor", [](const GpuMat &src, int code) {
        auto dst = std::make_shared<GpuMat>();
        cv::mx::cvtColor(src, *dst, code);
        return dst;
    }, py::arg("src"), py::arg("code"));
}
