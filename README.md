# OpenCV MX-C500 后端插件原型

本仓库提供 MX-C500 后端的 C++ 实现，以及可作为独立项目或 OpenCV extra module
使用的构建方式。实现基于 MACA Runtime 和沐曦 CV-CUDA，不依赖 NVIDIA GPU。

## 一、环境准备

### 1. 启动容器

```bash
sudo docker run -it --name opencv-mx-dev \
  --device=/dev/dri --device=/dev/mxcd --group-add video --shm-size=16g \
  -v /mnt/afs/xumengying/opencv_mx:/workspace/opencv_mx \
  -v /mnt/afs/xumengying/maca-cv-cuda-3.8.0.10:/opt/maca-cv-cuda:ro \
  -v /mnt/afs/xumengying/maca-sdk-3.8.2.6:/opt/maca-sdk:ro \
  cr.metax-tech.com/public-ai-release/maca/cv-cuda:0.16.0-maca.ai3.8.0.10-torch2.4-py312-ubuntu22.04-amd64 \
  /bin/bash
```

重新进入：

```bash
sudo docker start opencv-mx-dev
sudo docker exec -it opencv-mx-dev /bin/bash
```

### 2. 安装依赖

```bash
apt-get update
apt-get install -y build-essential cmake pkg-config libopencv-dev python3-dev
python -m pip install -U "pybind11>=2.12"
```

安装 CV-CUDA 和 MACA SDK 的 deb 包（从只读挂载目录复制到 `/tmp`，避免 `_apt` 权限警告）：

```bash
cp /opt/maca-cv-cuda/ai_deb/cvcuda-*.deb /tmp/
cp /opt/maca-sdk/deb/{mcruntime,mcanalyzer,commonlib,mccompiler,cu-bridge}_3.8.2.6.amd64.deb /tmp/
chmod 644 /tmp/*.deb
apt-get install -y /tmp/cvcuda-*.deb /tmp/{mcruntime,mcanalyzer,commonlib,mccompiler,cu-bridge}_3.8.2.6.amd64.deb
```

确认关键路径：`/opt/maca-ai/cvcuda0`、`/opt/maca-3.8.2`，以及 OpenCV 的
`/usr/lib/x86_64-linux-gnu/cmake/opencv4`。

## 二、编译和安装（二选一）

### 方案 A：Standalone 独立构建

适合快速验证本仓库。默认包含 Python 扩展（`BUILD_PYTHON=ON`）。

```bash
cd /workspace/opencv_mx
rm -rf build build-install
PYBIND11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR" \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
  -DCVCUDA_ROOT=/opt/maca-ai/cvcuda0 -DMACA_PATH=/opt/maca-3.8.2
cmake --build build -j"$(nproc)"
cmake --install build --prefix /opt/opencv-mx
```

### 方案 B：OpenCV contrib 插件构建

适合将本仓库作为独立 GitHub repo，与 OpenCV 一起编译。模块入口为
`modules/mx`，不要再配置本仓库顶层 CMake：

```bash
rm -rf /workspace/opencv/build-mx
cmake -S /workspace/opencv -B /workspace/opencv/build-mx \
  -DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv_mx/modules \
  -DBUILD_opencv_mx=ON -DBUILD_opencv_python3=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVCUDA_ROOT=/opt/maca-ai/cvcuda0 -DMACA_PATH=/opt/maca-3.8.2
cmake --build /workspace/opencv/build-mx -j"$(nproc)"
cmake --install /workspace/opencv/build-mx --prefix /opt/opencv-mx
```

插件模式由 OpenCV 主工程统一生成 `cv2` Python 模块；Standalone 的 pybind11
扩展名为 `opencv_mx`，两者不是同一个 Python API。

## 三、运行 test 模块

仅 Standalone 构建提供本仓库测试目标：

```bash
cd /workspace/opencv_mx
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

测试包括：

- `mx_resize_test`：C1/C3/C4 × 4 种插值，共 12 个 MX CV-CUDA golden 用例。
- `mx_gpumat_test`：尺寸/类型、内存复用、共享拷贝、ROI、异步 upload/download。
- `mx_cvtcolor_test`：MX-C500 上的 BGR→RGB 转换及像素一致性。

golden 文件位于 `testdata/golden/maca3.8.0.10-cvcuda0.16.0/`，测试不使用
`cv::resize()` 作为参考，也不依赖 Python、Torch 或 NVIDIA CUDA。

## 四、Example（C++）

Standalone 编译后运行：

```bash
cd /workspace/opencv_mx
./build/mx_resize_example input.jpg output.jpg
```

可选参数为插值方式、读取模式和输出尺寸：

```bash
./build/mx_resize_example input.jpg output.jpg 1 1 320 240
```

基本 C++ 调用：

```cpp
#include <opencv2/imgcodecs.hpp>
#include <opencv2/mx/imgproc.hpp>

cv::Mat input = cv::imread("input.jpg", cv::IMREAD_COLOR);
cv::mx::GpuMat src, dst;
src.upload(input);
cv::mx::resize(src, dst, cv::Size(320, 240));
cv::Mat output;
dst.download(output);
cv::imwrite("output.jpg", output);
```

支持 `CV_8UC1/3/4` packed HWC、四种 resize 插值、基础 8-bit `cvtColor`、
同步和 Stream 异步接口。异步操作完成前，源/目标 `GpuMat` 必须保持有效且不得重新分配。
