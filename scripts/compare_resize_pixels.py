#!/usr/bin/env python3
"""Compare CPU OpenCV, OpenCV CUDA and OpenCV-MX resize pixel values.

Run inside the opencv-mx-dev container, from the repository root:
  python3 scripts/compare_resize_pixels.py
"""
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "testdata/input_bgr.png"
OUT = ROOT / "testdata/resize_compare"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (512, 384) if "--opencv-bitexact" in sys.argv else (320, 240)

if "--all-bitexact" in sys.argv:
    # All CV_8U Linear8U modes from OpenCV test_resize_bitexact.cpp.
    modes = [(1,512,768),(3,512,768),(1,1024,384),(4,1024,384),
             (1,512,384),(2,512,384),(3,512,384),(4,512,384),
             (1,256,192),(2,256,192),(3,256,192),(4,256,192),
             (1,4,3),(2,4,3),(3,4,3),(4,4,3),(1,342,384),(1,342,256),
             (2,342,256),(3,342,256),(4,342,256),(1,512,256),(1,146,110),
             (3,146,110),(4,146,110),(1,931,698),(2,931,698),(3,931,698),
             (4,931,698),(1,853,640),(3,853,640),(4,853,640),
             (1,1004,753),(2,1004,753),(3,1004,753),(4,1004,753),
             (1,2048,1536),(2,2048,1536),(4,2048,1536),
             (1,3072,2304),(3,3072,2304),(1,7168,5376)]
    rows, cols = 768, 1024
    rng = np.random.default_rng(0x123456789abcdef & 0xffffffff)
    base = np.empty((rows, cols), np.uint8)
    for j in range(rows):
        for i in range(cols):
            if j < rows//2:
                val = ((np.sin((i+1)*np.pi/256)*np.sin((j+1)*np.pi/256)*np.sin(7*np.pi/8)+1)*128
                       if i < cols//2 else ((i//128+j//128)%2)*250+(j//128)%2)
            elif i < cols//2:
                val = (i//128)*(85-j//256*40)*(j//128%2)+(7-i//128)*(85-j//256*40)*((j//128+1)%2)
            else: val = int(rng.integers(0,256))
            base[j,i] = np.uint8(val)
    total = 0
    for idx,(cn,w,h) in enumerate(modes):
        if cn == 2:
            print(f"[{idx:02d}] CV_8UC2 1024x768->{w}x{h}: skipped (PNG/imread cannot represent 2-channel input)")
            continue
        src = np.repeat(base[:, :, None], cn, axis=2) if cn > 1 else base
        ip = OUT / f"bitexact_input_{idx}.png"; op = OUT / f"bitexact_mx_{idx}.png"
        cv2.imwrite(str(ip), src)
        cpu = cv2.resize(src, (w,h), interpolation=cv2.INTER_LINEAR_EXACT)
        subprocess.run([str(ROOT/"build/mx_resize_example"),str(ip),str(op),"1","-1",str(w),str(h)],check=True,stdout=subprocess.DEVNULL)
        mx = cv2.imread(str(op), cv2.IMREAD_UNCHANGED)
        d=np.abs(cpu.astype(np.int16)-mx.astype(np.int16)); nz=np.any(d!=0,axis=2) if d.ndim==3 else d!=0
        total += int(nz.sum()); print(f"[{idx:02d}] CV_8UC{cn} 1024x768->{w}x{h}: diff={nz.sum()}/{nz.size} max={d.max()} mean={d.mean():.6f}")
    print(f"all Linear8U modes: {len(modes)} cases, differing pixels={total}")
    sys.exit(0)
elif "--opencv-bitexact" in sys.argv:
    # Same deterministic 1024x768 CV_8UC3 pattern as OpenCV's
    # modules/imgproc/test/test_resize_bitexact.cpp (512x384 case).
    rows, cols, cn = 768, 1024, 3
    src = np.empty((rows, cols, cn), np.uint8)
    rng = np.random.default_rng(0x123456789abcdef & 0xffffffff)
    for j in range(rows):
        for i in range(cols):
            if j < rows // 2:
                if i < cols // 2:
                    val = (np.sin((i + 1)*np.pi/256)*np.sin((j + 1)*np.pi/256)*np.sin((cn + 4)*np.pi/8) + 1)*128
                else:
                    val = ((i//128 + j//128) % 2) * 250 + (j//128) % 2
            elif i < cols // 2:
                val = (i//128) * (85 - j//256 * 40) * (j//128 % 2) + (7 - i//128) * (85 - j//256 * 40) * ((j//128 + 1) % 2)
            else:
                val = int(rng.integers(0, 256))
            src[j, i, :] = np.uint8(val)
    cv2.imwrite(str(OUT / "opencv_bitexact_input.png"), src)
    input_path = OUT / "opencv_bitexact_input.png"
else:
    src = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    input_path = INPUT
if src is None:
    raise RuntimeError(f"cannot read {INPUT}")

interpolation = cv2.INTER_LINEAR_EXACT if "--opencv-bitexact" in sys.argv else cv2.INTER_LINEAR
cpu = cv2.resize(src, SIZE, interpolation=interpolation)
cv2.imwrite(str(OUT / "opencv_cpu.png"), cpu)

cuda_status = "unavailable"
cuda = None
if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    gpu = cv2.cuda_GpuMat(); gpu.upload(src)
    cuda = cv2.cuda.resize(gpu, SIZE, interpolation=cv2.INTER_LINEAR).download()
    cv2.imwrite(str(OUT / "opencv_cuda.png"), cuda)
    cuda_status = "ok"

mx_path = OUT / "opencv_mx.png"
subprocess.run([str(ROOT / "build/mx_resize_example"), str(input_path), str(mx_path),
                str(cv2.INTER_LINEAR), str(cv2.IMREAD_COLOR), str(SIZE[0]), str(SIZE[1])],
               check=True)
mx = cv2.imread(str(mx_path), cv2.IMREAD_COLOR)

def stats(name, a, b):
    d = a.astype(np.int16) - b.astype(np.int16)
    ad = np.abs(d)
    nz = np.any(d != 0, axis=2)
    print(f"{name}: differing_pixels={nz.sum()}/{nz.size} ({nz.mean()*100:.4f}%), "
          f"max_abs={ad.max()}, mean_abs={ad.mean():.6f}, sum_abs={ad.sum()}")
    ys, xs = np.where(nz)
    for y, x in list(zip(ys, xs))[:10]:
        print(f"  [{x},{y}] cpu={a[y,x].tolist()} other={b[y,x].tolist()} diff={d[y,x].tolist()}")

print(f"input={src.shape[1]}x{src.shape[0]}, output={SIZE[0]}x{SIZE[1]}, interpolation={interpolation}")
print(f"opencv_cuda={cuda_status}")
stats("opencv_mx-vs-cpu", cpu, mx)
if cuda is not None:
    stats("opencv_cuda-vs-cpu", cpu, cuda)
print(f"outputs: {OUT}")
