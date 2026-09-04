#!/usr/bin/env python3
import cv2, numpy as np, subprocess

modes = [(1,512,768),(3,512,768),(1,1024,384),(4,1024,384),(1,512,384),(2,512,384),(3,512,384),(4,512,384),(1,256,192),(2,256,192),(3,256,192),(4,256,192),(1,4,3),(2,4,3),(3,4,3),(4,4,3),(1,342,384),(1,342,256),(2,342,256),(3,342,256),(4,342,256),(1,512,256),(1,146,110),(3,146,110),(4,146,110),(1,931,698),(2,931,698),(3,931,698),(4,931,698),(1,853,640),(3,853,640),(4,853,640),(1,1004,753),(2,1004,753),(3,1004,753),(4,1004,753),(1,2048,1536),(2,2048,1536),(4,2048,1536),(1,3072,2304),(3,3072,2304),(1,7168,5376)]
rows, cols = 768, 1024; base=np.empty((rows,cols),np.uint8); rng=np.random.default_rng(0x123456789abcdef & 0xffffffff)
for y in range(rows):
 for x in range(cols):
  if y<rows//2: v=((np.sin((x+1)*np.pi/256)*np.sin((y+1)*np.pi/256)*np.sin(7*np.pi/8)+1)*128 if x<cols//2 else ((x//128+y//128)%2)*250+(y//128)%2)
  elif x<cols//2: v=(x//128)*(85-y//256*40)*(y//128%2)+(7-x//128)*(85-y//256*40)*((y//128+1)%2)
  else: v=int(rng.integers(0,256))
  base[y,x]=np.uint8(v)
for cn,w,h in modes:
 if cn==2: continue
 src=base if cn==1 else np.repeat(base[:,:,None],cn,2)
 for name,interp in (("NEAREST",0),("LINEAR",1),("CUBIC",2),("AREA",3)):
  ref=cv2.resize(src,(w,h),interpolation=interp); ip='/tmp/constructed.png'; op='/tmp/constructed_out.png'; cv2.imwrite(ip,src)
  subprocess.run(['/workspace/opencv_mx/build/mx_resize_example',ip,op,str(interp),'-1',str(w),str(h)],check=True,stdout=subprocess.DEVNULL)
  got=cv2.imread(op,cv2.IMREAD_UNCHANGED)
  if ref.ndim == 2 and got.ndim == 3:
   ref = np.repeat(ref[:, :, None], got.shape[2], axis=2)
  d=np.abs(ref.astype(np.int16)-got.astype(np.int16))
  print(f'CV_8UC{cn} {name} -> {w}x{h}: max={d.max()} mean={d.mean():.6f}')
