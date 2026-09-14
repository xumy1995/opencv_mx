#!/usr/bin/env python3
"""Visualize OpenCV CPU versus OpenCV CUDA resize results.

The CUDA build from the validation record is selected automatically.  Use
``--opencv-cuda-build`` or ``OPENCV_CUDA_BUILD`` to select another build.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CUDA_BUILD = Path(
    "/data/xumengying/tool-verify/opencv-verify/build-cuda"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opencv-cuda-build",
        type=Path,
        default=None,
        help=(
            "OpenCV CUDA build directory. Defaults to OPENCV_CUDA_BUILD, "
            f"{VALIDATION_CUDA_BUILD}, or ROOT/build-cuda."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: testdata/resize_visualization_cuda_512x384).",
    )
    return parser.parse_args()


def select_cuda_build(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("OPENCV_CUDA_BUILD"):
        candidates.append(Path(os.environ["OPENCV_CUDA_BUILD"]))
    candidates.extend((VALIDATION_CUDA_BUILD, ROOT / "build-cuda"))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "python_loader" / "cv2").is_dir():
            return candidate
    return None


def import_opencv(cuda_build: Path | None):
    """Import cv2, preferring the selected CUDA build's generated loader."""
    if cuda_build is not None:
        # The generated loader imports the extension by the top-level ``cv2``
        # name.  Remove an already-imported system cv2 so a prior import in a
        # host shell cannot silently select the non-CUDA wheel.
        sys.modules.pop("cv2", None)
        sys.path.insert(0, str(cuda_build / "python_loader"))
        sys.path.insert(1, str(cuda_build / "lib" / "python3"))
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        hint = str(cuda_build) if cuda_build else "an OpenCV CUDA build"
        raise RuntimeError(
            f"Unable to import {hint}. Set --opencv-cuda-build or "
            "OPENCV_CUDA_BUILD to a build containing python_loader/cv2."
        ) from exc
    return cv2


def require_cuda(cv2, cuda_build: Path | None) -> None:
    if not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "resize"):
        raise RuntimeError(
            "The imported OpenCV does not provide cv2.cuda.resize. Use a "
            "CUDA-enabled OpenCV Python build."
        )
    try:
        device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "OpenCV CUDA is present, but querying CUDA devices failed. Check "
            "the NVIDIA driver/device and the selected OpenCV build."
        ) from exc
    if device_count < 1:
        hint = f" ({cuda_build})" if cuda_build else ""
        raise RuntimeError(
            "No CUDA-enabled device is available for the selected OpenCV build"
            f"{hint}. Run this visualization on the CUDA validation host."
        )
    requested = int(os.environ.get("CUDA_DEVICE", "0"))
    if requested < 0 or requested >= device_count:
        raise RuntimeError(
            f"CUDA_DEVICE={requested} is outside the available device range "
            f"[0, {device_count - 1}]."
        )
    cv2.cuda.setDevice(requested)
    print(f"opencv_cuda_device={requested}/{device_count}")


def make_input(cv2, out: Path) -> np.ndarray:
    """Create the deterministic bitexact-style input used by the MX script."""
    width, height = 1024, 768
    base = np.empty((height, width), np.uint8)
    rng = np.random.default_rng(0x123456789ABCDEF & 0xFFFFFFFF)
    for y in range(height):
        for x in range(width):
            if y < height // 2:
                value = (
                    (np.sin((x + 1) * np.pi / 256)
                     * np.sin((y + 1) * np.pi / 256)
                     * np.sin(7 * np.pi / 8) + 1) * 128
                    if x < width // 2
                    else ((x // 128 + y // 128) % 2) * 250 + (y // 128) % 2
                )
            elif x < width // 2:
                value = ((x // 128) * (85 - y // 256 * 40) * (y // 128 % 2)
                         + (7 - x // 128) * (85 - y // 256 * 40)
                         * ((y // 128 + 1) % 2))
            else:
                value = int(rng.integers(0, 256))
            base[y, x] = np.uint8(value)
    src = np.repeat(base[:, :, None], 3, axis=2)
    cv2.imwrite(str(out / "input.png"), src)
    return src


def resize_cuda(cv2, src: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    gpu_src = cv2.cuda_GpuMat()
    gpu_src.upload(src)
    result = cv2.cuda.resize(gpu_src, size, interpolation=interpolation).download()
    if result is None or result.shape[:2] != (size[1], size[0]):
        shape = None if result is None else result.shape
        raise RuntimeError(f"cv2.cuda.resize returned an invalid result for {size}: {shape}")
    return result


def make_tile(cv2, image: np.ndarray, label: str, heat_scale: int | None = None) -> np.ndarray:
    header, legend_width = 38, 64
    tile = np.full((image.shape[0] + header, image.shape[1] + legend_width, 3), 255, np.uint8)
    tile[header:, :image.shape[1]] = image
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, header - 1), (32, 32, 32), -1)
    cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    if heat_scale is not None:
        bar_x, bar_h = image.shape[1] + 10, image.shape[0] - 20
        for y in range(bar_h):
            value = int(255 * (bar_h - 1 - y) / max(1, bar_h - 1))
            tile[header + 10 + y, bar_x:bar_x + 18] = cv2.applyColorMap(np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
        cv2.putText(tile, str(heat_scale), (bar_x + 22, header + 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, "0", (bar_x + 22, header + bar_h + 8), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, "low -> high", (bar_x - 2, header + bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def make_grid(cv2, rows) -> np.ndarray:
    gap, label_width = 24, 150
    tile_h, tile_w = rows[0][1][0].shape[:2]
    grid = np.full(
        (len(rows) * tile_h + (len(rows) - 1) * gap,
         label_width + 3 * tile_w + 2 * gap, 3),
        255,
        np.uint8,
    )
    for row, (name, tiles) in enumerate(rows):
        y = row * (tile_h + gap)
        cv2.putText(grid, name, (12, y + tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 0, 0), 2, cv2.LINE_AA)
        for col, tile in enumerate(tiles):
            x = label_width + col * (tile_w + gap)
            grid[y:y + tile_h, x:x + tile_w] = tile
    return grid


def make_constructed_grid(cv2, src: np.ndarray, out: Path, size: tuple[int, int]) -> None:
    print("constructed input (1024x768 deterministic pattern):")
    rows = []
    for name, interpolation in (("nearest", 0), ("linear", 1), ("cubic", 2), ("area", 3)):
        cpu = cv2.resize(src, size, interpolation=interpolation)
        cuda = resize_cuda(cv2, src, size, interpolation)
        diff = np.abs(cpu.astype(np.int16) - cuda.astype(np.int16)).astype(np.uint8)
        heat = cv2.applyColorMap(np.max(diff, axis=2), cv2.COLORMAP_JET)
        maximum = int(diff.max())
        rows.append((name.upper(), [
            make_tile(cv2, cpu, "OpenCV CPU"),
            make_tile(cv2, cuda, "OpenCV CUDA"),
            make_tile(cv2, heat, f"Absolute Diff (max={maximum})", 255),
        ]))
        print(f"{name}: max={maximum} mean={diff.mean():.6f} differing_pixels={np.any(diff != 0, axis=2).sum()}")
    cv2.imwrite(str(out / "comparison_grid.png"), make_grid(cv2, rows))


def make_lena_grid(cv2, out: Path, size: tuple[int, int]) -> None:
    lena = cv2.imread(str(ROOT / "testdata/shared/lena.png"), cv2.IMREAD_COLOR)
    if lena is None:
        return
    cv2.imwrite(str(out / "lena_input.png"), lena)
    rows, heat_scale = [], 125
    for name, interpolation in (("nearest", 0), ("linear", 1), ("cubic", 2), ("area", 3)):
        cpu = cv2.resize(lena, size, interpolation=interpolation)
        cuda = resize_cuda(cv2, lena, size, interpolation)
        diff = np.abs(cpu.astype(np.int16) - cuda.astype(np.int16)).astype(np.uint8)
        print(
            f"lena {name}: max={int(diff.max())} mean={diff.mean():.6f} "
            f"differing_pixels={np.any(diff != 0, axis=2).sum()}"
        )
        values = np.minimum(np.max(diff, axis=2), heat_scale).astype(np.uint8)
        heat = cv2.applyColorMap((values.astype(np.float32) * 255 / heat_scale).astype(np.uint8), cv2.COLORMAP_JET)
        maximum = int(diff.max())
        rows.append((name.upper(), [
            make_tile(cv2, cpu, "OpenCV CPU"),
            make_tile(cv2, cuda, "OpenCV CUDA"),
            make_tile(
                cv2,
                heat,
                f"Absolute Diff (max={maximum}, scale={heat_scale})",
                heat_scale,
            ),
        ]))
    cv2.imwrite(str(out / "lena_comparison_grid.png"), make_grid(cv2, rows))


def remove_intermediates(out: Path) -> None:
    for pattern in ("cpu_*.png", "cuda_*.png", "diff_*.png"):
        for path in out.glob(pattern):
            path.unlink()


def main() -> None:
    args = parse_args()
    cuda_build = select_cuda_build(args.opencv_cuda_build)
    cv2 = import_opencv(cuda_build)
    require_cuda(cv2, cuda_build)
    size = (512, 384)
    out = (args.output or ROOT / "testdata/resize_visualization_cuda_512x384").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = make_input(cv2, out)
    print(f"input={src.shape[1]}x{src.shape[0]}, output={size[0]}x{size[1]}")
    make_constructed_grid(cv2, src, out, size)
    make_lena_grid(cv2, out, size)
    remove_intermediates(out)
    print(f"outputs: {out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
