#!/usr/bin/env python3
"""Visualize CPU OpenCV vs MX resize for one constructed bitexact case."""
from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata/resize_visualization_512x384"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 512, 384
rows, cols = 768, 1024
base = np.empty((rows, cols), np.uint8)
rng = np.random.default_rng(0x123456789abcdef & 0xffffffff)
for y in range(rows):
    for x in range(cols):
        if y < rows // 2:
            value = ((np.sin((x + 1) * np.pi / 256) * np.sin((y + 1) * np.pi / 256) * np.sin(7 * np.pi / 8) + 1) * 128
                     if x < cols // 2 else ((x // 128 + y // 128) % 2) * 250 + (y // 128) % 2)
        elif x < cols // 2:
            value = (x // 128) * (85 - y // 256 * 40) * (y // 128 % 2) + (7 - x // 128) * (85 - y // 256 * 40) * ((y // 128 + 1) % 2)
        else:
            value = int(rng.integers(0, 256))
        base[y, x] = np.uint8(value)
src = np.repeat(base[:, :, None], 3, axis=2)
cv2.imwrite(str(OUT / "input.png"), src)
rows_for_grid = []
for name, interpolation in (("nearest", 0), ("linear", 1), ("cubic", 2), ("area", 3)):
    cpu = cv2.resize(src, (W, H), interpolation=interpolation)
    cpu_path = OUT / f"cpu_{name}.png"
    mx_path = OUT / f"mx_{name}.png"
    cv2.imwrite(str(cpu_path), cpu)
    subprocess.run([str(ROOT / "build/mx_resize_example"), str(OUT / "input.png"), str(mx_path), str(interpolation), "1", str(W), str(H)], check=True, stdout=subprocess.DEVNULL)
    mx = cv2.imread(str(mx_path), cv2.IMREAD_COLOR)
    diff = np.abs(cpu.astype(np.int16) - mx.astype(np.int16)).astype(np.uint8)
    # Use one fixed scale for every interpolation: blue=0, red=255.
    heat = cv2.applyColorMap(np.max(diff, axis=2), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / f"diff_{name}.png"), heat)
    max_diff = int(diff.max())
    labels = ["OpenCV CPU", "OpenCV MX", f"Absolute Diff (max={max_diff})"]
    tiles = []
    for image, label in zip((cpu, mx, heat), labels):
        header = 38
        legend_width = 64
        tile = np.full((image.shape[0] + header, image.shape[1] + legend_width, 3), 255, dtype=np.uint8)
        tile[header:, :image.shape[1]] = image
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, header - 1), (32, 32, 32), -1)
        cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if image is heat:
            bar_x = image.shape[1] + 10
            bar_h = image.shape[0] - 20
            for y in range(bar_h):
                value = int(255 * (bar_h - 1 - y) / max(1, bar_h - 1))
                tile[header + 10 + y, bar_x:bar_x + 18] = cv2.applyColorMap(
                    np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
            cv2.putText(tile, "255", (bar_x + 22, header + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, "0", (bar_x + 22, header + bar_h + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, "low -> high", (bar_x - 2, header + bar_h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows_for_grid.append((name.upper(), tiles))
    print(f"{name}: max={diff.max()} mean={diff.mean():.6f} differing_pixels={np.any(diff != 0, axis=2).sum()}")
gap = 24
label_width = 150
tile_h, tile_w = rows_for_grid[0][1][0].shape[:2]
grid = np.full((len(rows_for_grid) * tile_h + (len(rows_for_grid) - 1) * gap,
                label_width + 3 * tile_w + 2 * gap, 3), 255, dtype=np.uint8)
for row, (name, tiles) in enumerate(rows_for_grid):
    y = row * (tile_h + gap)
    cv2.putText(grid, name, (12, y + tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (0, 0, 0), 2, cv2.LINE_AA)
    for col, tile in enumerate(tiles):
        x = label_width + col * (tile_w + gap)
        grid[y:y + tile_h, x:x + tile_w] = tile
cv2.imwrite(str(OUT / "comparison_grid.png"), grid)
# grid for OpenCV's canonical Lena test image.
lena = cv2.imread(str(ROOT / "testdata/shared/lena.png"), cv2.IMREAD_COLOR)
if lena is not None:
    cv2.imwrite(str(OUT / "lena_input.png"), lena)
    lena_dir = OUT / "lena"
    lena_dir.mkdir(exist_ok=True)
    lena_rows = []
    for name, interpolation in (("nearest", 0), ("linear", 1), ("cubic", 2), ("area", 3)):
        cpu = cv2.resize(lena, (W, H), interpolation=interpolation)
        mx_path = lena_dir / f"mx_{name}.png"
        cv2.imwrite(str(lena_dir / f"cpu_{name}.png"), cpu)
        subprocess.run([str(ROOT / "build/mx_resize_example"), str(ROOT / "testdata/shared/lena.png"), str(mx_path), str(interpolation), "1", str(W), str(H)], check=True, stdout=subprocess.DEVNULL)
        mx = cv2.imread(str(mx_path), cv2.IMREAD_COLOR)
        diff = np.abs(cpu.astype(np.int16) - mx.astype(np.int16)).astype(np.uint8)
        # Lena comparison uses a fixed 0~125 scale; larger errors saturate red.
        lena_heat_scale = 125
        heat_values = np.minimum(np.max(diff, axis=2), lena_heat_scale).astype(np.uint8)
        heat = cv2.applyColorMap((heat_values.astype(np.float32) * 255 / lena_heat_scale).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(lena_dir / f"diff_{name}.png"), heat)
        lena_rows.append((name.upper(), cpu, mx, heat, int(diff.max()), lena_heat_scale))
    canvas = np.full((len(lena_rows) * (H + 62), 150 + 3 * (W + 64) + 48, 3), 255, np.uint8)
    for row, (name, cpu, mx, heat, maximum, heat_scale) in enumerate(lena_rows):
        y = row * (H + 62); cv2.putText(canvas, name, (12, y + H // 2), cv2.FONT_HERSHEY_SIMPLEX, .72, (0,0,0), 2)
        for col, (image, label) in enumerate(((cpu,"OpenCV CPU"),(mx,"OpenCV MX"),(heat,f"Absolute Diff (max={maximum}, scale=125)"))):
            x = 150 + col * (W + 64); canvas[y+38:y+38+H, x:x+W] = image
            cv2.putText(canvas, label, (x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,0,0), 1)
            if col == 2:
                bar_x = x + W + 10
                for by in range(H - 20):
                    value = int(255 * (H - 21 - by) / max(1, H - 21))
                    canvas[y + 38 + 10 + by, bar_x:bar_x + 18] = cv2.applyColorMap(
                        np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
                cv2.putText(canvas, "125", (bar_x + 22, y + 38 + 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1)
                cv2.putText(canvas, "0", (bar_x + 22, y + 38 + H - 12), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1)
                cv2.putText(canvas, "low -> high", (bar_x - 2, y + 38 + H + 8), cv2.FONT_HERSHEY_SIMPLEX, .34, (0,0,0), 1)
    cv2.imwrite(str(OUT / "lena_comparison_grid.png"), canvas)
print(f"outputs: {OUT}")

# Keep only the two summary images; all intermediate CPU/MX/diff files are
# temporary implementation details.
for path in OUT.glob("cpu_*.png"):
    path.unlink()
for path in OUT.glob("mx_*.png"):
    path.unlink()
for path in OUT.glob("diff_*.png"):
    path.unlink()
# Keep input.png and lena_input.png as the two source images for reference.
import shutil
shutil.rmtree(OUT / "lena", ignore_errors=True)
