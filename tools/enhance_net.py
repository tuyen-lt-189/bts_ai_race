#!/usr/bin/env python3
"""L16 — Mạng hậu xử lý per-scene (post-render enhancement).

Ý tưởng: render các TRAIN view từ checkpoint → có cặp (render, GT train).
Train U-Net nhỏ sửa residual (exposure, blur nhẹ, artifact) với loss mô phỏng
đúng thang BTC: 0.4·LPIPS-vgg + 0.3·(1−SSIM) + 0.3·L1. Áp mạng lên render test.
Không đụng GT test — chỉ học từ train views. Validate trên public trước khi dùng.

Dùng (2 bước):
  # 1) train net từ các render train-view đã có (tự render nếu thiếu)
  python enhance_net.py train --workspace workspace_raw/SC --ckpt ck.pt \
      --with_ut --radial_k1 K1 --out results/SC__l16/net.pt [--steps 3000]
  # 2) áp lên thư mục render test
  python enhance_net.py apply --net results/SC__l16/net.pt \
      --in_dir renders/SC__SUB2 --out_dir renders/SC__l16
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


class UNetSmall(nn.Module):
    """U-Net 3 tầng ~1.9M params, đầu ra residual (khởi tạo ~identity)."""

    def __init__(self, ch=(32, 64, 128), ch_mult=1):
        ch = tuple(c * ch_mult for c in ch)
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.e1 = block(3, ch[0]); self.e2 = block(ch[0], ch[1]); self.e3 = block(ch[1], ch[2])
        self.d2 = block(ch[2] + ch[1], ch[1]); self.d1 = block(ch[1] + ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear",
                                              align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear",
                                              align_corners=False), e1], 1))
        return (x + self.head(d1)).clamp(0, 1)


class UNetVGG(nn.Module):
    """B1 — U-Net với encoder VGG16 PRETRAINED (ImageNet): mang tri thức ngoài vào
    đúng chỗ L16 bão hoà (đã đo: net to hơn from-scratch chỉ +0.00007 — nút thắt là
    THÔNG TIN chứ không phải capacity). Decoder nhẹ + residual, head khởi tạo zero."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.e1 = feats[:4]     # 64ch  (relu1_2)  full-res
        self.e2 = feats[4:9]    # 128ch (relu2_2)  1/2
        self.e3 = feats[9:16]   # 256ch (relu3_3)  1/4
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.d2 = block(256 + 128, 128)
        self.d1 = block(128 + 64, 64)
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        z = (x - self.mean) / self.std
        e1 = self.e1(z); e2 = self.e2(e1); e3 = self.e3(e2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([up(e3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return (x + self.head(d1)).clamp(0, 1)


def build_net(arch, ch_mult=1):
    return UNetVGG() if arch == "vgg" else UNetSmall(ch_mult=ch_mult)


def render_train_views(ws, ckpt_path, out_dir, with_ut, radial_k1, antialiased=True, radial_k2=0.0):
    """Render mọi train view của workspace từ ckpt (skip nếu đã đủ)."""
    from colmap_io import read_cameras_bin, read_images_bin, read_points3D_bin
    from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                                  compute_parser_transform)
    from gsplat import rasterization
    ws = Path(ws); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    items = sorted(imgs.values(), key=lambda im: im.name)
    if sum((out_dir / (Path(im.name).stem + ".png")).exists() for im in items) == len(items):
        print(f"render train views: đã đủ {len(items)} — skip"); return
    _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
    transform = compute_parser_transform(imgs, xyz)
    dev = "cuda"
    ck = torch.load(ckpt_path, map_location=dev, weights_only=True)["splats"]
    means = ck["means"].to(dev)
    quats = F.normalize(ck["quats"].to(dev), dim=-1)
    scales = torch.exp(ck["scales"].to(dev))
    opac = torch.sigmoid(ck["opacities"].to(dev))
    colors = torch.cat([ck["sh0"], ck["shN"]], 1).to(dev)
    shd = int(np.sqrt(colors.shape[1]) - 1)
    kw = {}
    if with_ut:
        kw = dict(with_ut=True, with_eval3d=True, packed=False)
        if radial_k1 is not None:
            kw["radial_coeffs"] = torch.tensor([[radial_k1, radial_k2, 0, 0, 0, 0]],
                                               device=dev).float()
    with torch.no_grad():
        for i, im in enumerate(items):
            cam = cams[im.camera_id]
            if cam.model == "SIMPLE_RADIAL":
                f, cx, cy, _ = cam.params; fx = fy = f
            elif cam.model == "SIMPLE_PINHOLE":
                f, cx, cy = cam.params; fx = fy = f
            else:
                fx, fy, cx, cy = cam.params[:4]
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float()[None].to(dev)
            vm = colmap_w2c_to_normalized_viewmat(im.R, im.tvec, transform)
            vm = torch.from_numpy(vm).float()[None].to(dev)
            r, _, _ = rasterization(
                means, quats, scales, opac, colors, vm, K,
                cam.width, cam.height, sh_degree=shd, near_plane=0.01, far_plane=1e10,
                rasterize_mode="antialiased" if antialiased else "classic",
                render_mode="RGB", **kw)
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(out_dir / (Path(im.name).stem + ".png")), img[..., ::-1])
            if (i + 1) % 60 == 0:
                print(f"  render train {i + 1}/{len(items)}")


def cmd_train(a):
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    dev = "cuda"
    ws = Path(a.workspace)
    rt_dir = Path(a.out).parent / "renders_train"
    render_train_views(ws, a.ckpt, rt_dir, a.with_ut, a.radial_k1, radial_k2=a.radial_k2)
    stems = sorted(p.stem for p in rt_dir.glob("*.png"))
    # nạp cặp (render, GT) lên RAM dạng uint8
    gt_dir = ws / "images"
    gt_map = {p.stem: p for p in gt_dir.iterdir()}
    pairs = []
    for s in stems:
        r = cv2.imread(str(rt_dir / f"{s}.png"))
        g = cv2.imread(str(gt_map[s]))
        pairs.append((r, g))
    # giữ 10% cuối làm val (theo tên — ổn định)
    n_val = max(4, len(pairs) // 10)
    tr, va = pairs[:-n_val], pairs[-n_val:]
    print(f"L16: {len(tr)} train / {len(va)} val pairs")

    net = build_net(a.arch, a.ch_mult).to(dev)
    if a.arch == "vgg":  # encoder pretrained: lr thấp hơn 10× để không phá prior
        enc = [p for n, p in net.named_parameters() if n.startswith("e")]
        dec = [p for n, p in net.named_parameters() if not n.startswith("e")]
        opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": dec, "lr": 2e-4}],
                                weight_decay=1e-5)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    def to_t(bgr):
        return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255

    def batch(pairs, ps, bs):
        xs, ys = [], []
        for _ in range(bs):
            r, g = pairs[rng.integers(len(pairs))]
            h, w = r.shape[:2]
            y0 = rng.integers(0, h - ps + 1); x0 = rng.integers(0, w - ps + 1)
            xs.append(to_t(r[y0:y0 + ps, x0:x0 + ps]))
            ys.append(to_t(g[y0:y0 + ps, x0:x0 + ps]))
        return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

    def score_val(apply_net):  # chấm v50 TỪNG ẢNH val (né OOM full-res)
        P, S, L = [], [], []
        with torch.no_grad():
            for r, g in va:
                x = to_t(r)[None].to(dev)
                y = to_t(g)[None].to(dev)
                if apply_net:
                    h, w = x.shape[-2:]
                    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
                    x = net(F.pad(x, (0, pw, 0, ph), mode="replicate"))[..., :h, :w]
                mse = ((x - y) ** 2).mean().item()
                P.append(10 * np.log10(1 / max(mse, 1e-10)))
                S.append(tm_ssim(x, y, data_range=1.0).item())
                L.append(lp(x * 2 - 1, y * 2 - 1).mean().item())
                del x, y; torch.cuda.empty_cache()
        p, s, l = float(np.mean(P)), float(np.mean(S)), float(np.mean(L))
        return 0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50, 1), p, s, l

    rng = np.random.default_rng(0)
    base = score_val(apply_net=False)
    print(f"val BASE:    v50={base[0]:.5f} PSNR={base[1]:.3f} SSIM={base[2]:.4f} LPIPS={base[3]:.4f}")

    best = (-1, None)
    for step in range(1, a.steps + 1):
        x, y = batch(tr, a.patch, a.batch)
        out = net(x)
        l1 = (out - y).abs().mean()
        ss = tm_ssim(out, y, data_range=1.0)
        lv = lp(out * 2 - 1, y * 2 - 1).mean()
        loss = 0.4 * lv + 0.3 * (1 - ss) + 0.3 * l1
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps:
            sc = score_val(apply_net=True)
            marker = ""
            if sc[0] > best[0]:
                best = (sc[0], {k: v.detach().cpu() for k, v in net.state_dict().items()})
                marker = " ← best"
            print(f"step {step}: val v50={sc[0]:.5f} PSNR={sc[1]:.3f} "
                  f"SSIM={sc[2]:.4f} LPIPS={sc[3]:.4f}{marker}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best[1], "base_v50": base[0], "best_v50": best[0],
                "ch_mult": a.ch_mult, "arch": a.arch}, a.out)
    print(f"VAL-GAIN {best[0] - base[0]:+.5f} → {a.out}")


def cmd_apply(a):
    dev = "cuda"
    ck = torch.load(a.net, map_location=dev, weights_only=True)
    net = build_net(ck.get("arch", "unet"), ck.get("ch_mult", 1)).to(dev).eval()
    net.load_state_dict(ck["state"])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.in_dir).glob("*.png"))
    with torch.no_grad():
        for f in files:
            bgr = cv2.imread(str(f))
            x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)[None].to(dev) / 255
            # pad chia hết 4 cho U-Net 2 lần pool
            h, w = x.shape[-2:]
            ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
            y = net(F.pad(x, (0, pw, 0, ph), mode="replicate"))[..., :h, :w]
            img = (y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(out / f.name), img[..., ::-1])
    print(f"applied {len(files)} ảnh → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--workspace", required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--with_ut", action="store_true")
    t.add_argument("--radial_k1", type=float, default=None)
    t.add_argument("--radial_k2", type=float, default=0.0)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--patch", type=int, default=256)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--ch_mult", type=int, default=1, help="nhân độ rộng U-Net (G3: 2)")
    t.add_argument("--arch", choices=["unet", "vgg"], default="unet",
                   help="vgg = B1 encoder VGG16 pretrained (tri thức ngoài)")
    p = sub.add_parser("apply")
    p.add_argument("--net", required=True)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    (cmd_train if a.cmd == "train" else cmd_apply)(a)
