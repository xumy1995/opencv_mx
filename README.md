# OpenCV MX-C500 后端原型

## 下载镜像和 SDK

### 1. CV-CUDA 镜像

- 镜像：`cv-cuda:0.16.0-maca.ai3.8.0.10-torch2.4-py312-ubuntu22.04-amd64`
- 下载地址：https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=cv-cuda&frame_version=0.16.0&python_version=3.12&system=ubuntu

### 2. MACA SDK

- SDK 名称：`maca-sdk-3.8.2.6-deb-x86_64.tar.xz`
- 下载地址：https://developer.metax-tech.com/softnova/download?dimension=metax&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&package_kind=SDK&chip_name=

### 3. MACA CV-CUDA

- SDK 名称：`maca-cv-cuda-0.16.0-py312-3.8.0.10-linux-x86_64.tar.xz`
- 下载地址：https://developer.metax-tech.com/softnova/ai-download/cv-cuda?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=cv-cuda&ai_label=CV-CUDA

## 加载镜像

```bash
sudo docker run -it --name opencv-mx-dev \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --group-add video \
  --shm-size=16g \
  -v /mnt/afs/xumengying/opencv_mx:/workspace/opencv_mx \
  -v /mnt/afs/xumengying/maca-cv-cuda-3.8.0.10:/opt/maca-cv-cuda:ro \
  -v /mnt/afs/xumengying/maca-sdk-3.8.2.6:/opt/maca-sdk:ro \
  cr.metax-tech.com/public-ai-release/maca/cv-cuda:0.16.0-maca.ai3.8.0.10-torch2.4-py312-ubuntu22.04-amd64 \
  /bin/bash
```

退出容器后再次进入：

```bash
sudo docker start opencv-mx-dev
sudo docker exec -it opencv-mx-dev /bin/bash
```

## 容器内构建和运行

本项目实现了第一版 C++ API：

```cpp
cv::mx::GpuMat
cv::mx::GpuMat::upload()
cv::mx::GpuMat::download()
cv::mx::resize()
```

OpenCV 负责 CPU 图像读写，MACA Runtime 负责 MX-C500 显存和数据搬运，沐曦版 CV-CUDA 负责 GPU resize。

以下步骤从已经进入 `opencv-mx-dev` 容器后开始。

## 1. 确认挂载目录

容器启动时应已挂载：

```text
/workspace/opencv_mx   项目源码
/opt/maca-cv-cuda     maca-cv-cuda 3.8.0.10 开发包
/opt/maca-sdk         MACA SDK 3.8.2.6 开发包
```

容器内确认：

```bash
test -f /workspace/opencv_mx/CMakeLists.txt
test -f /opt/maca-cv-cuda/ai_deb/cvcuda-dev-0.16.0-cuda11-x86_64-linux.deb
test -f /opt/maca-sdk/deb/mcruntime_3.8.2.6.amd64.deb
echo "mounts OK"
```

## 2. 安装系统构建依赖

```bash
apt-get update
apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev
```

本项目使用 Ubuntu 提供的 OpenCV 4.5.4 C++ 开发包。Python 的 `opencv-python` wheel 不包含 C++ 头文件和 `OpenCVConfig.cmake`，不能替代 `libopencv-dev`。

## 3. 安装 CV-CUDA C++ SDK

将只读挂载目录中的 deb 复制到 `/tmp`，避免 APT 的 `_apt` 读取权限警告：

```bash
cp /opt/maca-cv-cuda/ai_deb/cvcuda-lib-0.16.0-cuda11-x86_64-linux.deb /tmp/
cp /opt/maca-cv-cuda/ai_deb/cvcuda-dev-0.16.0-cuda11-x86_64-linux.deb /tmp/
chmod 644 /tmp/cvcuda-*.deb

apt-get install -y \
  /tmp/cvcuda-lib-0.16.0-cuda11-x86_64-linux.deb \
  /tmp/cvcuda-dev-0.16.0-cuda11-x86_64-linux.deb
```

安装后得到：

```text
/opt/maca-ai/cvcuda0/include/cvcuda
/opt/maca-ai/cvcuda0/include/nvcv
/opt/maca-ai/cvcuda0/lib/x86_64-linux-gnu
```

## 4. 安装最小 MACA C++ 开发组件

当前项目需要以下五个包：

```bash
cp /opt/maca-sdk/deb/mcruntime_3.8.2.6.amd64.deb /tmp/
cp /opt/maca-sdk/deb/mcanalyzer_3.8.2.6.amd64.deb /tmp/
cp /opt/maca-sdk/deb/commonlib_3.8.2.6.amd64.deb /tmp/
cp /opt/maca-sdk/deb/mccompiler_3.8.2.6.amd64.deb /tmp/
cp /opt/maca-sdk/deb/cu-bridge_3.8.2.6.amd64.deb /tmp/
chmod 644 /tmp/*.amd64.deb

apt-get install -y \
  /tmp/mcruntime_3.8.2.6.amd64.deb \
  /tmp/mcanalyzer_3.8.2.6.amd64.deb \
  /tmp/commonlib_3.8.2.6.amd64.deb \
  /tmp/mccompiler_3.8.2.6.amd64.deb \
  /tmp/cu-bridge_3.8.2.6.amd64.deb
```

这些包分别提供 MACA Runtime、公共导出头、基础类型、编译器公共头以及 CV-CUDA 所需的 CUDA 兼容头和 `libruntime_cu.so`。

## 5. 验证开发环境

```bash
test -f /usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake
test -f /opt/maca-ai/cvcuda0/include/cvcuda/OpResize.h
test -f /opt/maca-ai/cvcuda0/include/nvcv/Tensor.h
test -f /opt/maca-3.8.2/include/mcr/mc_runtime_api.h
test -f /opt/maca-3.8.2/tools/cu-bridge/include/cuda_runtime.h
test -f /opt/maca-3.8.2/lib/libmcruntime.so
test -f /opt/maca-3.8.2/lib/libruntime_cu.so
echo "development environment OK"
```

## 6. 配置和编译

```bash
cd /workspace/opencv_mx
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
  -DCVCUDA_ROOT=/opt/maca-ai/cvcuda0 \
  -DMACA_PATH=/opt/maca-3.8.2

cmake --build build -j"$(nproc)"
```

成功时应看到：

```text
[100%] Built target mx_resize_example
```

## 7. 运行 resize 示例

准备一张输入图片，例如：

```text
/workspace/opencv_mx/input.jpg
```

运行：

```bash
cd /workspace/opencv_mx
./build/mx_resize_example input.jpg output.jpg
```

对于 `640 × 480` 输入，预期输出信息为：

```text
640x480 -> 320x240
```


## 8. C++ 使用方式

```cpp
#include <opencv2/imgcodecs.hpp>
#include <opencv2/mx/imgproc.hpp>

cv::Mat src = cv::imread("input.jpg", cv::IMREAD_COLOR);

cv::mx::GpuMat mxSrc;
cv::mx::GpuMat mxDst;

mxSrc.upload(src);
cv::mx::resize(mxSrc, mxDst, cv::Size(320, 240));

cv::Mat dst;
mxDst.download(dst);
cv::imwrite("output.jpg", dst);
```

当前支持：

- `CV_8UC3`、`CV_8UC4`（packed HWC）
- `INTER_LINEAR`、`INTER_AREA`
- 同步 upload、resize、download
- `cv::mx::Stream` 异步接口
- `GpuMat` 拷贝共享所有权和 ROI 视图

## 9. 运行自动化测试

```bash
cd /workspace/opencv_mx
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

测试覆盖：

- `CV_8UC3`、`CV_8UC4`
- `INTER_LINEAR`、`INTER_AREA`
- CPU↔MX-C500 upload/download 数据一致性
- ROI 视图和共享拷贝
- 输出尺寸、类型及 CPU OpenCV 结果误差

在当前 CV-CUDA 0.16 外部 Tensor 路径中，`INTER_NEAREST` 和 `INTER_CUBIC`
与 OpenCV 的舍入语义存在差异，因此暂未纳入一致性测试。

当前尚未实现 `CV_8UC1` resize、完整引用计数语义、显存池和 Python `cv2.mx` 绑定。
