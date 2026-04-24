"""Visualize the parse-derived routing classes at latent res, overlayed on GT."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"
OUT = f"{BASE}/vtonautoresearch/diag_out/parse_v6"
os.makedirs(OUT, exist_ok=True)

for sid in ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]:
    pg = torch.load(f"{LATENTS}/{sid}_parse_garment_latent.pt", weights_only=True).float()
    ps = torch.load(f"{LATENTS}/{sid}_parse_skin_latent.pt", weights_only=True).float()
    pb = torch.load(f"{LATENTS}/{sid}_parse_bg_latent.pt", weights_only=True).float()
    # dilate warped as edit mask
    wm = torch.load(f"{LATENTS}/{sid}_warped_mask_128.pt", weights_only=True).float()
    if wm.dim() == 3: wm = wm.unsqueeze(0)
    if pg.dim() == 3: pg = pg.unsqueeze(0)
    if ps.dim() == 3: ps = ps.unsqueeze(0)
    if pb.dim() == 3: pb = pb.unsqueeze(0)
    M_edit = F.max_pool2d((wm > 0.5).float(), 15, 1, 7).clamp(0, 1)    # dilate 7 latent px ≈ 56 image px
    M_core = -F.max_pool2d(-(wm > 0.5).float(), 5, 1, 2)                # erode 2
    M_core = (M_core > 0.5).float()

    # 4-way inside M_edit
    M_g = (M_edit * (pg > 0.5).float()).clamp(0, 1)
    M_s = (M_edit * (ps > 0.5).float() * (1 - M_g)).clamp(0, 1)
    M_b = (M_edit - M_g - M_s).clamp(0, 1)
    M_k = (1.0 - M_edit)

    # Color-code and upsample for viz
    def up(m): return F.interpolate(m, size=(1024, 768), mode="nearest")[0, 0]
    gt_path = f"{BASE}/vtonautoresearch/runs/vton_20260423_133455/final/panel/{sid}_gt.png"
    gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32)

    col = gt.copy()
    col[up(M_g).numpy() > 0.5] = 0.5 * col[up(M_g).numpy() > 0.5] + 0.5 * np.array([0, 255, 0])
    col[up(M_s).numpy() > 0.5] = 0.5 * col[up(M_s).numpy() > 0.5] + 0.5 * np.array([255, 200, 0])
    col[up(M_b).numpy() > 0.5] = 0.5 * col[up(M_b).numpy() > 0.5] + 0.5 * np.array([255, 0, 0])
    col = col.clip(0, 255).astype(np.uint8)

    n = 128 * 96
    print(f"{sid}: M_g={int(M_g.sum()):>4}  M_s={int(M_s.sum()):>4}  "
          f"M_b={int(M_b.sum()):>4}  M_k={int(M_k.sum()):>4}  "
          f"(sum={int(M_g.sum())+int(M_s.sum())+int(M_b.sum())+int(M_k.sum())}/{n})")

    Image.fromarray(np.concatenate([gt.astype(np.uint8), col], axis=1)).save(f"{OUT}/{sid}_classes.png")
print(f"saved to {OUT}")
