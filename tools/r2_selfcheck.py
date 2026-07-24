#!/usr/bin/env python3
"""ROUND 2 không có GT test (test/images trống toàn bộ) → không thể chấm local.
Lưới an toàn thay thế: render N pose TRAIN qua ĐÚNG đường ống test (CSV →
normalize_compat → render_test_poses) rồi PSNR với ảnh train thật.

Bắt được lớp bug đắt nhất (đã trả giá ở round 1): lệch transform T (thiếu T3 flip
→ render lật 180°, PSNR 14, không hề báo lỗi). Model đã nhìn thấy pose train khi
học → PSNR phải CAO (>=20); nếu thấp là sai transform/intrinsics, KHÔNG phải model kém.

Dùng:
  gen   : python r2_selfcheck.py gen --ws workspace_r2/bonsai --n 3 --out /tmp/sc_bonsai.csv
  score : python r2_selfcheck.py score --render_dir /tmp/sc_render --ws workspace_r2/bonsai \
              [--min_psnr 20]     ← exit 1 nếu dưới ngưỡng (bash bắt được)
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}


def read_camera(ws):
    with open(ws / "sparse/0/cameras.bin", "rb") as f:
        f.read(8)
        cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        p = struct.unpack(f"<{NPARAMS[model]}d", f.read(8 * NPARAMS[model]))
    if model == 0:      # SIMPLE_PINHOLE f cx cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == 2:    # SIMPLE_RADIAL f cx cy k1
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == 1:    # PINHOLE fx fy cx cy
        fx, fy, cx, cy = p
    else:
        sys.exit(f"❌ camera model {model} chưa hỗ trợ selfcheck")
    return fx, fy, cx, cy, w, h


def read_train_poses(ws):
    """images.bin của workspace (hệ COLMAP GỐC — prepare chỉ lọc, không transform)."""
    out = []
    with open(ws / "sparse/0/images.bin", "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            q = struct.unpack("<dddd", f.read(32))
            t = struct.unpack("<ddd", f.read(24))
            struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts, = struct.unpack("<Q", f.read(8))
            f.read(24 * npts)
            out.append((name.decode(), q, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--ws", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", help="gen: file CSV ra")
    ap.add_argument("--render_dir", help="score: thư mục render PNG")
    ap.add_argument("--min_psnr", type=float, default=20.0)
    a = ap.parse_args()
    ws = Path(a.ws)

    if a.mode == "gen":
        fx, fy, cx, cy, w, h = read_camera(ws)
        poses = read_train_poses(ws)
        # lấy đều dọc quỹ đạo (đầu/giữa/cuối) — tránh cụm 1 góc
        idx = np.linspace(0, len(poses) - 1, a.n).astype(int)
        with open(a.out, "w") as f:
            f.write("image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n")
            for i in idx:
                name, q, t = poses[i]
                f.write(f"{name},{q[0]},{q[1]},{q[2]},{q[3]},{t[0]},{t[1]},{t[2]},"
                        f"{fx},{fy},{cx},{cy},{w},{h}\n")
        print(f"✓ {a.out}: {a.n} pose train ({[poses[i][0] for i in idx]})")
        return

    # ---- score ----
    import cv2
    rd = Path(a.render_dir)
    renders = sorted(rd.glob("*.png")) + sorted(rd.glob("*.jpg")) + sorted(rd.glob("*.JPG"))
    if not renders:
        sys.exit(f"❌ selfcheck: không có render nào trong {rd}")
    psnrs = []
    for rp in renders:
        gt_p = None
        for ext in ("", ".JPG", ".jpg", ".png"):
            c = ws / "images" / (rp.stem + ext) if ext else ws / "images" / rp.name
            if c.exists():
                gt_p = c
                break
        if gt_p is None:
            sys.exit(f"❌ selfcheck: không tìm thấy ảnh train gốc cho {rp.name}")
        r = cv2.imread(str(rp)).astype(np.float64)
        g = cv2.imread(str(gt_p)).astype(np.float64)
        if r.shape != g.shape:
            sys.exit(f"❌ selfcheck: {rp.name} kích thước {r.shape} ≠ GT {g.shape}")
        mse = ((r - g) ** 2).mean()
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))
        psnrs.append(psnr)
        print(f"  {rp.name}: PSNR={psnr:.2f}")
    # MEDIAN chứ không mean: 1 view ngoại biên học kém (PSNR ~6 ở đầu quỹ đạo — đã gặp
    # HCM0421 smoke 16/07) không được phép đánh trượt cả scene; bug transform thật
    # thì lật TẤT CẢ view → median vẫn bắt được.
    m = float(np.median(psnrs))
    print(f"  mean={np.mean(psnrs):.2f}  median={m:.2f}")
    if m < a.min_psnr:
        print(f"❌ SELFCHECK FAIL: PSNR median {m:.2f} < {a.min_psnr} — "
              f"transform/intrinsics SAI (kiểu bug T3-flip round 1), model KHÔNG kém. "
              f"DỪNG scene này, đừng phí GPU cho seed tiếp theo.")
        sys.exit(1)
    print(f"✅ SELFCHECK OK: PSNR train-pose median {m:.2f} ≥ {a.min_psnr} — "
          f"đường ống render chuẩn cho scene này.")


if __name__ == "__main__":
    main()
