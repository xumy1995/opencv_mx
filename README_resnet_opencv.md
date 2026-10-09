# ResNet-50 OpenCV 预处理精度与性能对比

本实验在 MetaX C500 上，使用同一个 Hugging Face ResNet-50 和 ImageNet-1k
validation，对比三种预处理后端：

- `hf`：Hugging Face `AutoImageProcessor`。Transformers 5.12.1 实际使用
  torchvision backend；processor 接收整个 batch，输出一个 batch Tensor，再一次性
  上传到 C500，不逐图上传；
- `opencv_cpu`：官方 CPU OpenCV resize；
- `opencv_mx`：本仓库 standalone Python 扩展提供的 MX OpenCV resize。

三路从 Hugging Face datasets 解码得到的相同 PIL RGB 图像开始，因此本次不比较
JPEG 解码。OpenCV 两路只替换 resize，中心裁剪、归一化和 HWC 到 NCHW 共用代码。
参数为：短边 resize 到 256、中心裁剪 224x224、BICUBIC、ImageNet mean/std、FP32。

## 环境

```text
image: cr.metax-tech.com/public-ai-release/maca/cv-cuda:0.16.0-maca.ai3.8.0.10-torch2.4-py312-ubuntu22.04-amd64
cv2: 5.0.0
datasets: 5.0.1
pybind11: 3.1.0
torch: 2.4.0+metax3.8.0.7
transformers: 5.12.1
device: MetaX C500
```

在主机创建容器：

```bash
sudo docker run -it --name opencv-mx-eval \
  --device=/dev/dri --device=/dev/mxcd --group-add video --shm-size=16g \
  -v /mnt/afs/xumengying/opencv_mx:/workspace/opencv_mx \
  -v /mnt/afs/xumengying/models_and_datasets:/mnt/afs/xumengying/models_and_datasets:ro \
  -v /mnt/afs/xumengying/maca-cv-cuda-3.8.0.10:/opt/maca-cv-cuda:ro \
  -v /mnt/afs/xumengying/maca-sdk-3.8.2.6:/opt/maca-sdk:ro \
  cr.metax-tech.com/public-ai-release/maca/cv-cuda:0.16.0-maca.ai3.8.0.10-torch2.4-py312-ubuntu22.04-amd64 \
  /bin/bash
```

重新进入已有容器：

```bash
sudo docker start opencv-mx-eval
sudo docker exec -it opencv-mx-eval /bin/bash
```

以下命令均在容器中执行。安装依赖：

```bash
apt-get update
apt-get install -y build-essential cmake pkg-config libopencv-dev python3-dev
python -m pip install -U pybind11 opencv-python-headless datasets transformers Pillow tqdm

cp /opt/maca-cv-cuda/ai_deb/cvcuda-*.deb /tmp/
cp /opt/maca-sdk/deb/{mcruntime,mcanalyzer,commonlib,mccompiler,cu-bridge}_3.8.2.6.amd64.deb /tmp/
chmod 644 /tmp/*.deb
apt-get install -y /tmp/cvcuda-*.deb \
  /tmp/{mcruntime,mcanalyzer,commonlib,mccompiler,cu-bridge}_3.8.2.6.amd64.deb
```

不需要安装 `cvcuda-python3.12`，评测使用本仓库的 pybind11 扩展。如果出现
`No module named 'cv2'`，执行 `python -m pip install -U opencv-python-headless`。

## 编译

```bash
cd /workspace/opencv_mx
PYBIND11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
cmake -S . -B build-eval -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR" \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
  -DCVCUDA_ROOT=/opt/maca-ai/cvcuda0 -DMACA_PATH=/opt/maca-3.8.2
cmake --build build-eval -j"$(nproc)"
ctest --test-dir build-eval --output-on-failure
PYTHONPATH=/workspace/opencv_mx/build-eval \
  python -m unittest -v test_eval_resnet_opencv
```

本次实测 CTest `3/3`、Python 单测 `4/4` 通过。

## 运行

为公平比较串行预处理，三路统一使用 `--num-workers 0`。完整评测命令：

```bash
cd /workspace/opencv_mx
mkdir -p logs
set -o pipefail
PYTHONPATH=/workspace/opencv_mx/build-eval python eval_resnet_opencv.py \
  --model-path /mnt/afs/xumengying/models_and_datasets/resnet-50 \
  --dataset-path /mnt/afs/xumengying/models_and_datasets/imagenet-1k \
  --cache-dir /tmp/hf_datasets_cache \
  --mx-python-path /workspace/opencv_mx/build-eval \
  --batch-size 64 --num-workers 0 --warmup-batches 5 \
  --backends hf opencv_cpu opencv_mx \
  --check-parity --tolerance-pp 0.1 \
  2>&1 | tee logs/resnet50-opencv-workers0.log
```

快速检查可在上述命令中加入 `--max-samples 100`，但小样本不适合使用严格的
`0.1 pp` 精度阈值。只运行一路可使用例如 `--backends hf`。

## 计时口径

前六项与 `/mnt/afs/xumengying/model-verify/mx-c500/eval_resnet_mx.py` 完全一致：

- `Preprocess time`：已准备好的 batch Tensor 和标签上传 C500 并同步；这个名称沿用
  旧脚本，**不包含图像 resize/crop/normalize**；
- `Inference time`：模型 forward 并同步；
- `Measured wall`：H2D 加 forward；
- `Throughput`：按 `Measured wall` 计算；
- `Infer throughput`：按 `Inference time` 计算；
- `Avg infer latency`：平均每张图 forward 时间。

为了不隐去真实预处理耗时，脚本另外输出：

- `Image prep work`：collate 内图像转换、resize、crop、normalize 的工作时间；
- `Data wait`：主进程从请求 batch 到 DataLoader 返回的实际墙钟时间，包含数据读取、
  解码和图像预处理；
- `End-to-end wall/rate`：从请求 batch 到完成推理的墙钟时间和吞吐。

当前脚本还会打印 measured batches 的分段 breakdown。`num-workers=0` 时各项可直接
解释为主进程累计时间：

- `dataset_getitem`：datasets 取样、parquet/Arrow 字段访问和图像对象解码；
- `rgb_convert`：HF 路径为 PIL `.convert("RGB")`；OpenCV 路径为图像对象转连续
  RGB NumPy；
- `image_processor`：仅 HF backend，统计批量 torchvision resize、crop、rescale、
  normalize 和 batch Tensor 输出；
- `resize`：CPU/MX backend 的逐图 resize；`upload`、`download`：仅 MX backend 的逐图 GPU 往返；
- `postprocess`：center crop、float 转换、归一化和 HWC 到 NCHW；
- `stack`：把单图 Tensor 组装成 batch；
- `collate`：整个 collate 函数耗时，包含上述 collate 内部步骤。

这些分段时间之和不一定严格等于 `Data wait`：`Data wait` 还包含 DataLoader/Python
调度和计时边界开销，而 `collate` 已包含其内部各步骤。

设置多个 DataLoader worker 时，`Image prep work` 是各 worker 时间之和，不能直接作为
墙钟时间；`Data wait` 和 `End-to-end wall` 仍是墙钟时间。正式三路对比使用 worker 0。

## 50,000 张完整结果

以下结果由上面的命令直接生成，三路均为 `batch-size=64`、`warmup-batches=5`、
`num-workers=0`、FP32。表中的前六个时间/吞吐字段与旧脚本口径一致；后四个字段
反映真实图像预处理和端到端时间。

| Backend | Top-1 | Top-5 | Preprocess* | Inference | Measured wall | Throughput | Infer throughput | Image prep work | End-to-end wall | End-to-end rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF/torchvision | 80.13% | 94.48% | 1.774s | 17.665s | 19.440s | 2555.55/s | 2812.36/s | 60.398s | 206.962s | 240.04/s |
| OpenCV CPU | 79.22% | 94.11% | 1.481s | 17.580s | 19.062s | 2606.20/s | 2826.00/s | 70.068s | 211.021s | 235.43/s |
| OpenCV MX | 79.21% | 94.11% | 1.477s | 17.580s | 19.059s | 2606.66/s | 2825.90/s | 95.879s | 238.643s | 208.18/s |

`*Preprocess` 沿用旧脚本名称，实际只表示已生成 batch Tensor 的 H2D 传输和同步，
不表示图像 resize/crop/normalize 时间。

MX 相对 CPU OpenCV 的精度差异为 Top-1 `0.0120 pp`、Top-5 `0.0020 pp`，低于
`0.1 pp` 阈值，检查通过。日志位于 `logs/resnet50-opencv-workers0.log`。该历史日志
是在分段 breakdown 加入前生成的，因此没有上述各子项；重新运行命令后即可获得分段数据。

完整 50,000 张 HF 单后端分段复测记录在 `logs/resnet50-hf-breakdown.log`：

```text
Data wait        : 184.857s
dataset_getitem  : 123.925s
collate          : 58.791s
rgb_convert      : 7.352s
image_processor  : 51.397s
```

`rgb_convert + image_processor = 58.749s`，与 `collate=58.791s` 的约 `0.042s`
差值来自 label Tensor 构造、字典访问和计时/Python 开销。HF 的 ImageProcessor 在
CPU 上按 batch 执行 torchvision resize、crop、rescale 和 normalize；完成后整个
batch Tensor 才一次性上传 C500。

## 性能限制

本次端到端时间显示，CPU OpenCV 为 `211.021s`，MX OpenCV 为 `238.643s`，HF 为
`206.962s`。因此当前 MX 端到端约比 CPU OpenCV 慢 `13.1%`，原因是 Python 原型对
每张图分别执行 `GpuMat.upload -> resize -> download`：

1. 没有 batch resize API，50,000 张图会产生 50,000 次 upload/resize/download 调用；
2. 每张图都发生 CPU 与 C500 间的数据往返，抵消了单次 MX resize kernel 的优势；
3. MX 为保证 GpuMat 上下文安全只能使用 `num-workers=0`，无法像 CPU/HF 那样通过
   多个 DataLoader worker 并行准备数据；
4. 当前实现没有把 MX 预处理与模型推理异步流水线重叠。

这是真实的当前 Python 绑定和调度开销，不能据此推断 MX resize kernel 本身慢。旧脚本
口径下三路推理吞吐均约 `2,826 samples/s`，说明主要差异发生在图像预处理路径。若要
评估 kernel 性能，需要使用 batch/device-side API，或另做去除逐图传输的 benchmark。
