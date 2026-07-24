#!/usr/bin/env python3
"""GB1 — ensemble nhiều bộ render (mean/median) rồi chấm. Ảnh cùng tên giữa các dir."""
import argparse, glob, os
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["mean","median"], default="mean")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = set(os.path.basename(f) for f in glob.glob(a.dirs[0]+"/*.png"))
for n in names:
    stk = [cv2.imread(os.path.join(d,n)).astype(np.float32) for d in a.dirs if os.path.exists(os.path.join(d,n))]
    if len(stk) < 2: continue
    arr = np.stack(stk)
    out = arr.mean(0) if a.mode=="mean" else np.median(arr,0)
    cv2.imwrite(os.path.join(a.out,n), out.round().astype(np.uint8))
print(f"ensemble {a.mode} {len(a.dirs)} dirs → {a.out}")
