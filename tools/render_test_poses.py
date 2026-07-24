#!/usr/bin/env python3
"""Render ảnh tại các pose trong test_poses.csv từ checkpoint gsplat simple_trainer.

Quy ước pose ĐÃ XÁC MINH (docs/00 §3.1): qvec/tvec trong CSV là COLMAP world-to-camera.
viewmat = [[R, t], [0, 1]] — đúng input `viewmats` của gsplat.rasterization.

Render-time tricks (docs/06 L3):
  --supersample 2   : render 2× rồi downsample INTER_AREA (khử aliasing)
  --redistort_k1 K1 : áp lại méo SIMPLE_RADIAL lên ảnh render (GT test là ảnh gốc còn méo)

Dùng:
  python render_test_poses.py --ckpt results/HCM0204/ckpts/ckpt_29999_rank0.pt \
      --csv .../HCM0204/test/test_poses.csv --out renders/HCM0204 [--antialiased]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def load_splats(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    s = ckpt["splats"]
    means = s["means"].to(device)
    quats = torch.nn.functional.normalize(s["quats"].to(device), dim=-1)
    scales = torch.exp(s["scales"].to(device))
    opacities = torch.sigmoid(s["opacities"].to(device))
    colors = torch.cat([s["sh0"], s["shN"]], dim=1).to(device)  # [N, K, 3]
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)
    print(f"splats: {means.shape[0]:,} gaussians, sh_degree={sh_degree}")
    return means, quats, scales, opacities, colors, sh_degree


def redistort_map(K, k1, w, h):
    """Map để tạo ảnh MÉO từ ảnh pinhole: với mỗi pixel méo x_d, lấy mẫu tại vị trí
    undistort(x_d) trên ảnh pinhole (đảo chiều đúng của phép undistort)."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)[:, None, :]
    dist = np.array([k1, 0, 0, 0, 0.0])
    und = cv2.undistortPoints(pts, K, dist, P=K)  # pixel coords trên ảnh pinhole
    m = und.reshape(h, w, 2)
    return m[..., 0], m[..., 1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None,
                    help="workspace scene đã train (BẮT BUỘC nếu train với "
                         "normalize_world_space mặc định của gsplat — tức là hầu hết "
                         "mọi run). Dùng để tính transform chuẩn hoá y hệt Parser.")
    ap.add_argument("--no_normalize", action="store_true",
                    help="chỉ dùng khi model train với --normalize-world-space False")
    ap.add_argument("--with_ut", action="store_true",
                    help="[EXPERIMENTAL] render bằng Unscented Transform (3DGUT) cho model "
                         "train với --with_ut --with_eval3d; kết hợp --radial_k1 để render "
                         "TRỰC TIẾP với distortion (thay thế --redistort_k1 warp)")
    ap.add_argument("--radial_k2", type=float, default=0.0, help="B2: k2 của model OPENCV refined")
    ap.add_argument("--tangential", type=float, nargs=2, default=None, help="B2: p1 p2 OPENCV refined")
    ap.add_argument("--radial_k1", type=float, default=None,
                    help="k1 SIMPLE_RADIAL cho render 3DGUT (chỉ dùng cùng --with_ut)")
    ap.add_argument("--antialiased", action="store_true",
                    help="PHẢI khớp với chế độ lúc train")
    ap.add_argument("--supersample", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--redistort_k1", type=float, default=None)
    ap.add_argument("--ext", default=".png", help=".png (mặc định) hoặc giữ '.same' theo CSV")
    args = ap.parse_args()

    from gsplat import rasterization  # import muộn để --help chạy được không cần GPU

    device = "cuda"
    means, quats, scales, opacities, colors, sh_degree = load_splats(args.ckpt, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ss = args.supersample

    # Transform chuẩn hoá world space (model được train trong hệ này!)
    transform = None
    if args.data_dir:
        sys.path.insert(0, str(Path(__file__).parent))
        from colmap_io import read_images_bin, read_points3D_bin
        from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                                      compute_parser_transform)
        ws = Path(args.data_dir)
        imgs = read_images_bin(ws / "sparse/0/images.bin")
        _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
        transform = compute_parser_transform(imgs, xyz)
        print(f"normalize transform: scale={np.linalg.norm(transform[0, :3]):.4f}")
    elif args.no_normalize:
        print("no_normalize: dùng pose COLMAP thô (model phải được train với "
              "--normalize-world-space False)")
    else:
        raise SystemExit(
            "LỖI: thiếu --data_dir <workspace>. Model train bằng simple_trainer mặc định "
            "sống trong hệ toạ độ CHUẨN HOÁ — render pose thô sẽ ra ảnh đen/lệch "
            "(bug đã gặp 10/7). Thêm --data_dir, hoặc --no_normalize nếu chắc chắn "
            "model train với --normalize-world-space False.\n"
            "→ Trong notebook: chạy lại cell HELPERS (S0.4) để nạp render() bản mới!")

    rows = list(csv.DictReader(open(args.csv)))
    for i, r in enumerate(rows):
        q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
        t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        fx, fy, cx, cy = (float(r[k]) for k in ("fx", "fy", "cx", "cy"))
        w, h = int(r["width"]), int(r["height"])

        if transform is not None:
            viewmat = colmap_w2c_to_normalized_viewmat(qvec2rotmat(q), t, transform)
        else:
            viewmat = np.eye(4)
            viewmat[:3, :3] = qvec2rotmat(q)
            viewmat[:3, 3] = t
        # Supersample: scale intrinsics theo quy ước tâm pixel (x+0.5)*ss-0.5
        K = np.array([[fx * ss, 0, (cx + 0.5) * ss - 0.5],
                      [0, fy * ss, (cy + 0.5) * ss - 0.5],
                      [0, 0, 1.0]])
        ut_kwargs = {}
        if args.with_ut:
            assert args.redistort_k1 is None, "--with_ut render distortion nguyên bản — bỏ --redistort_k1"
            # gsplat 1.5.3: rasterization() mặc định packed=True nhưng UT cấm packed
            ut_kwargs = dict(with_ut=True, with_eval3d=True, packed=False)
            if args.radial_k1 is not None:
                ut_kwargs["radial_coeffs"] = torch.tensor(
                    [[args.radial_k1, args.radial_k2, 0, 0, 0, 0]], device=device).float()
                if args.tangential is not None:
                    ut_kwargs["tangential_coeffs"] = torch.tensor(
                        [list(args.tangential)], device=device).float()
        render, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            torch.from_numpy(viewmat).float()[None].to(device),
            torch.from_numpy(K).float()[None].to(device),
            w * ss, h * ss, sh_degree=sh_degree,
            near_plane=0.01, far_plane=1e10,
            rasterize_mode="antialiased" if args.antialiased else "classic",
            render_mode="RGB",
            **ut_kwargs,
        )
        img = render[0].clamp(0, 1).cpu().numpy()
        img = (img * 255).round().astype(np.uint8)[..., ::-1]  # RGB→BGR cho cv2
        if ss > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if args.redistort_k1 is not None:
            Kp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            mx, my = redistort_map(Kp, args.redistort_k1, w, h)
            img = cv2.remap(img, mx.astype(np.float32), my.astype(np.float32),
                            interpolation=cv2.INTER_LANCZOS4)
        name = r["image_name"]
        if args.ext != ".same":
            name = str(Path(name).with_suffix(args.ext))
        cv2.imwrite(str(out / name), img)
        if (i + 1) % 20 == 0 or i == len(rows) - 1:
            print(f"  render {i + 1}/{len(rows)}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
