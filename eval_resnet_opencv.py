#!/usr/bin/env python3
"""Compare ResNet-50 preprocessing backends on the same ImageNet samples.

The model and labels are shared by all backends.  The OpenCV backends start
from the RGB pixels returned by the datasets library, so JPEG decoding is not
part of this first operator comparison.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

# Keep older onnx/protobuf installations importable through transformers.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEFAULT_MODEL_PATH = "/mnt/afs/xumengying/models_and_datasets/resnet-50"
DEFAULT_DATASET_PATH = "/mnt/afs/xumengying/models_and_datasets/imagenet-1k"
DEFAULT_CACHE_DIR = "/mnt/afs/xumengying/hf_datasets_cache"
DEFAULT_MX_PYTHON_PATH = (
    "/workspace/opencv_mx/build"
    if os.path.isdir("/workspace/opencv_mx/build")
    else "/mnt/afs/xumengying/opencv_mx/build"
)

IMAGE_SIZE = 224
CROP_PCT = 0.875
RESIZE_SHORT_EDGE = int(IMAGE_SIZE / CROP_PCT)
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF, CPU OpenCV and MX OpenCV ResNet-50 preprocessing."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate. Use 'auto' for validation/val/test fallback.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("hf", "opencv_cpu", "opencv_mx"),
        default=["hf", "opencv_cpu", "opencv_mx"],
        help="Preprocessing backends to evaluate in the given order.",
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples. 0 means the whole selected split.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Warmup batches excluded from timing (accuracy still includes them).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 for the model input/model on CUDA.",
    )
    parser.add_argument(
        "--mx-python-path",
        default=DEFAULT_MX_PYTHON_PATH,
        help="Directory containing the standalone opencv_mx Python extension.",
    )
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=0.1,
        help="Allowed absolute accuracy difference from opencv_cpu, in percentage points.",
    )
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="Exit nonzero when an OpenCV backend exceeds --tolerance-pp.",
    )
    return parser.parse_args()


def load_imagenet_dataset(
    dataset_path: str, split: str, cache_dir: str
) -> tuple[str, Any]:
    """Load only the requested local parquet split when it is available."""
    from datasets import DatasetDict, load_dataset
    from datasets.exceptions import DatasetGenerationError

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    try:
        if split != "auto":
            parquet_pattern = os.path.join(dataset_path, "data", f"{split}-*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))
            if parquet_files:
                return split, load_dataset(
                    "parquet",
                    data_files={split: parquet_files},
                    split=split,
                    cache_dir=cache_dir,
                )
            return split, load_dataset(dataset_path, split=split, cache_dir=cache_dir)

        for candidate in ("validation", "val", "test"):
            parquet_pattern = os.path.join(dataset_path, "data", f"{candidate}-*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))
            if parquet_files:
                return candidate, load_dataset(
                    "parquet",
                    data_files={candidate: parquet_files},
                    split=candidate,
                    cache_dir=cache_dir,
                )
            try:
                return candidate, load_dataset(
                    dataset_path, split=candidate, cache_dir=cache_dir
                )
            except (DatasetGenerationError, ValueError, KeyError):
                continue
        dataset = load_dataset(dataset_path, cache_dir=cache_dir)
    except DatasetGenerationError as exc:
        raise RuntimeError(
            "Failed to load ImageNet-1k. The dataset looks incomplete or a parquet "
            f"shard is still downloading under {dataset_path}."
        ) from exc

    if not isinstance(dataset, DatasetDict):
        return "default", dataset
    for candidate in ("validation", "val", "test"):
        if candidate in dataset:
            return candidate, dataset[candidate]
    available = list(dataset.keys())
    if not available:
        raise RuntimeError(f"No dataset splits found under {dataset_path}")
    return available[0], dataset[available[0]]


def _rgb_array(image: Any) -> np.ndarray:
    """Return an owned uint8 HWC RGB array for one dataset image."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB HWC image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    # A contiguous owned array is required by the pybind11 upload wrapper.
    return np.ascontiguousarray(array)


def resize_output_size(height: int, width: int) -> tuple[int, int]:
    """Return ``(width, height)`` using ConvNextImageProcessor's calculation."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape {(height, width)}")
    short, long = (width, height) if width <= height else (height, width)
    new_short = RESIZE_SHORT_EDGE
    new_long = int(new_short * long / short)
    if width <= height:
        return new_short, new_long
    return new_long, new_short


def center_crop(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    top = (height - size) // 2
    left = (width - size) // 2
    if top < 0 or left < 0:
        raise ValueError(f"image {(height, width)} is smaller than crop {size}")
    return image[top : top + size, left : left + size]


def normalize_to_nchw(image: np.ndarray) -> torch.Tensor:
    image_f32 = image.astype(np.float32) / 255.0
    image_f32 = (image_f32 - IMAGE_MEAN) / IMAGE_STD
    # copy() ensures torch.from_numpy never receives a negative-stride view.
    return torch.from_numpy(image_f32.transpose(2, 0, 1).copy())


def preprocess_opencv_cpu(images: list[Any]) -> torch.Tensor:
    import cv2

    output = []
    for image in images:
        rgb = _rgb_array(image)
        width, height = resize_output_size(rgb.shape[0], rgb.shape[1])
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        output.append(normalize_to_nchw(center_crop(resized)))
    return torch.stack(output, dim=0)


def import_mx(mx_python_path: str) -> Any:
    if mx_python_path and mx_python_path not in sys.path:
        sys.path.insert(0, mx_python_path)
    try:
        import opencv_mx as mx
    except ImportError as exc:
        raise RuntimeError(
            "Could not import standalone opencv_mx. Build it first and pass its "
            "directory with --mx-python-path."
        ) from exc
    return mx


def preprocess_opencv_mx(images: list[Any], mx: Any) -> torch.Tensor:
    import cv2

    output = []
    for image in images:
        rgb = _rgb_array(image)
        width, height = resize_output_size(rgb.shape[0], rgb.shape[1])
        source = mx.GpuMat()
        source.upload(rgb)
        resized = mx.resize(source, (width, height), int(cv2.INTER_CUBIC))
        # The standalone binding synchronizes download and returns uint8 HWC.
        resized_rgb = np.asarray(resized.download())
        if resized_rgb.ndim == 2:
            resized_rgb = resized_rgb[..., None]
        output.append(normalize_to_nchw(center_crop(resized_rgb)))
    return torch.stack(output, dim=0)


@dataclass
class Backend:
    name: str
    collate: Callable[[list[dict[str, Any]]], tuple[torch.Tensor, torch.Tensor, float]]


def make_backend(
    name: str, processor: Any | None, mx: Any | None
) -> Backend:
    if name == "hf":
        if processor is None:
            raise ValueError("HF backend requires an image processor")

        def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, float]:
            start = time.perf_counter()
            images = [item["image"].convert("RGB") for item in batch]
            labels = torch.tensor(
                [item.get("label", -1) for item in batch], dtype=torch.long
            )
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            return pixels, labels, time.perf_counter() - start

    elif name == "opencv_cpu":

        def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, float]:
            start = time.perf_counter()
            images = [item["image"] for item in batch]
            labels = torch.tensor(
                [item.get("label", -1) for item in batch], dtype=torch.long
            )
            pixels = preprocess_opencv_cpu(images)
            return pixels, labels, time.perf_counter() - start

    elif name == "opencv_mx":
        if mx is None:
            raise ValueError("MX backend requires the opencv_mx module")

        def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, float]:
            start = time.perf_counter()
            images = [item["image"] for item in batch]
            labels = torch.tensor(
                [item.get("label", -1) for item in batch], dtype=torch.long
            )
            pixels = preprocess_opencv_mx(images, mx)
            return pixels, labels, time.perf_counter() - start

    else:
        raise ValueError(f"unknown backend {name}")
    return Backend(name, collate)


def compute_correct(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int]:
    valid = labels >= 0
    if not bool(valid.any()):
        return 0, 0, 0
    valid_logits = logits[valid]
    valid_labels = labels[valid]
    top1 = valid_logits.argmax(dim=1)
    top1_correct = int((top1 == valid_labels).sum().item())
    top5 = valid_logits.topk(min(5, valid_logits.shape[1]), dim=1).indices
    top5_correct = int(
        (top5 == valid_labels.unsqueeze(1)).any(dim=1).sum().item()
    )
    return top1_correct, top5_correct, int(valid_labels.numel())


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(
    args: argparse.Namespace,
    backend: Backend,
    split_name: str,
    dataset: Any,
    model: Any,
    device: torch.device,
) -> dict[str, float | int | str]:
    workers = args.num_workers
    if backend.name == "opencv_mx" and workers:
        # CUDA/MACA contexts must not be initialized in forked DataLoader workers.
        print("opencv_mx: forcing --num-workers 0 for GpuMat safety")
        workers = 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=backend.collate,
    )

    total_seen = measured_seen = labeled_seen = 0
    top1_correct = top5_correct = 0
    preprocess_time = transfer_time = inference_time = measured_wall_time = 0.0
    data_wait_time = end_to_end_time = 0.0
    progress = tqdm(
        range(len(loader)),
        total=len(loader),
        desc=f"{device} {split_name} {backend.name}",
        unit="batch",
        dynamic_ncols=True,
    )
    loader_iterator = iter(loader)
    with torch.inference_mode():
        for batch_index in progress:
            end_to_end_start = time.perf_counter()
            pixel_values, labels, batch_prep_time = next(loader_iterator)
            data_ready = time.perf_counter()
            measured = batch_index >= args.warmup_batches
            batch_size = labels.numel()
            batch_start = time.perf_counter()
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.fp16 and device.type == "cuda":
                pixel_values = pixel_values.half()
            sync_if_cuda(device)
            transfer_elapsed = time.perf_counter() - batch_start
            infer_start = time.perf_counter()
            logits = model(pixel_values=pixel_values).logits
            sync_if_cuda(device)
            infer_elapsed = time.perf_counter() - infer_start
            if measured:
                measured_seen += batch_size
                preprocess_time += batch_prep_time
                data_wait_time += data_ready - end_to_end_start
                transfer_time += transfer_elapsed
                inference_time += infer_elapsed
                measured_wall_time += time.perf_counter() - batch_start
                end_to_end_time += time.perf_counter() - end_to_end_start
            correct1, correct5, valid_count = compute_correct(logits, labels)
            top1_correct += correct1
            top5_correct += correct5
            labeled_seen += valid_count
            total_seen += batch_size
            progress.set_postfix(
                samples=total_seen,
                top1=(f"{100.0 * top1_correct / labeled_seen:.2f}%" if labeled_seen else "N/A"),
            )

    top1 = 100.0 * top1_correct / labeled_seen if labeled_seen else float("nan")
    top5 = 100.0 * top5_correct / labeled_seen if labeled_seen else float("nan")
    return {
        "backend": backend.name,
        "samples": total_seen,
        "measured_samples": measured_seen,
        "labeled": labeled_seen,
        "top1": top1,
        "top5": top5,
        "preprocess_seconds": preprocess_time,
        "data_wait_seconds": data_wait_time,
        "transfer_seconds": transfer_time,
        "inference_seconds": inference_time,
        "wall_seconds": measured_wall_time,
        "end_to_end_seconds": end_to_end_time,
        "samples_per_second": measured_seen / measured_wall_time if measured_wall_time else 0.0,
        "inference_samples_per_second": measured_seen / inference_time if inference_time else 0.0,
        "end_to_end_samples_per_second": measured_seen / end_to_end_time if end_to_end_time else 0.0,
    }


def print_result(
    args: argparse.Namespace,
    split_name: str,
    model_load_time: float,
    result: dict[str, float | int | str],
) -> None:
    measured_samples = int(result["measured_samples"])
    inference_time = float(result["inference_seconds"])
    avg_latency_ms = 1000.0 * inference_time / measured_samples if measured_samples else 0.0
    print(f"\nBackend          : {result['backend']}")
    print(f"Model path       : {args.model_path}")
    print(f"Dataset path     : {args.dataset_path}")
    print(f"Split            : {split_name}")
    print(f"Samples          : {result['samples']}")
    print(f"Labeled samples  : {result['labeled']}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Warmup batches   : {args.warmup_batches}")
    print(f"Precision        : {'fp16' if args.fp16 and args.device == 'cuda' else 'fp32'}")
    print(f"Model load time  : {model_load_time:.3f}s")
    # These six fields intentionally match eval_resnet_mx.py's definitions.
    print(f"Preprocess time  : {result['transfer_seconds']:.3f}s")
    print(f"Inference time   : {result['inference_seconds']:.3f}s")
    print(f"Measured wall    : {result['wall_seconds']:.3f}s")
    print(f"Throughput       : {result['samples_per_second']:.2f} samples/s")
    print(f"Infer throughput : {result['inference_samples_per_second']:.2f} samples/s")
    print(f"Avg infer latency: {avg_latency_ms:.3f} ms/sample")
    print(f"Top-1 accuracy   : {result['top1']:.2f}%")
    print(f"Top-5 accuracy   : {result['top5']:.2f}%")
    # The old script excludes image preprocessing; report it explicitly as well.
    print(f"Image prep work  : {result['preprocess_seconds']:.3f}s")
    print(f"Data wait        : {result['data_wait_seconds']:.3f}s")
    print(f"End-to-end wall  : {result['end_to_end_seconds']:.3f}s")
    print(f"End-to-end rate  : {result['end_to_end_samples_per_second']:.2f} samples/s")


def main() -> int:
    args = parse_args()
    if args.max_samples < 0 or args.warmup_batches < 0:
        raise ValueError("--max-samples and --warmup-batches must be non-negative")
    if args.check_parity and "opencv_cpu" not in args.backends:
        raise ValueError("--check-parity requires opencv_cpu in --backends")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/C500 is not available; use --device cpu for a smoke test")
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    split_name, dataset = load_imagenet_dataset(args.dataset_path, args.split, args.cache_dir)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if not len(dataset):
        raise RuntimeError("Dataset split is empty")

    if "hf" in args.backends:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(args.model_path)
    else:
        processor = None
    mx = import_mx(args.mx_python_path) if "opencv_mx" in args.backends else None
    device = torch.device(args.device)
    from transformers import ResNetForImageClassification

    model_load_start = time.perf_counter()
    model = ResNetForImageClassification.from_pretrained(args.model_path).eval().to(device)
    if args.fp16 and device.type == "cuda":
        model.half()
    sync_if_cuda(device)
    model_load_time = time.perf_counter() - model_load_start

    print(f"Loaded split '{split_name}' with {len(dataset)} samples.")
    print(
        f"Preprocess: shortest edge {RESIZE_SHORT_EDGE}, center crop {IMAGE_SIZE}, "
        "OpenCV interpolation INTER_CUBIC"
    )
    results = []
    for name in args.backends:
        result = run_backend(
            args,
            make_backend(name, processor, mx),
            split_name,
            dataset,
            model,
            device,
        )
        results.append(result)
        print_result(args, split_name, model_load_time, result)

    reference = next((r for r in results if r["backend"] == "opencv_cpu"), None)
    if args.check_parity and reference is not None:
        failures = []
        for result in results:
            if result["backend"] in ("opencv_cpu", "hf"):
                continue
            top1_delta = abs(float(result["top1"]) - float(reference["top1"]))
            top5_delta = abs(float(result["top5"]) - float(reference["top5"]))
            print(
                f"parity {result['backend']} vs opencv_cpu: "
                f"top1_delta={top1_delta:.4f}pp top5_delta={top5_delta:.4f}pp"
            )
            if top1_delta > args.tolerance_pp or top5_delta > args.tolerance_pp:
                failures.append(result["backend"])
        if failures:
            print(
                f"Parity check failed for {', '.join(str(x) for x in failures)} "
                f"(tolerance {args.tolerance_pp:.4f}pp).",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
