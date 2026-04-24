"""
Sanity-check the 3-region masks that train.py builds at latent resolution (128×96),
using the exact same operations as train_step: body_latent via interpolate(area) +
threshold, body_repair = repair_band * body_latent, bg_repair = complement.

Outputs per-sample panel:
  row 1: GT | agnostic_mask_128 | warped_mask_128 | body_latent(dp)
  row 2: core | body_repair | bg_repair | (uncertain+keep overlay)

And prints pixel counts.
"""
import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"
OUT = f"{BASE}/vtonautoresearch/diag_out/region_masks_latent"
os.makedirs(OUT, exist_ok=True)

SIDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]

def norm255(x):
    return (x.clamp(0, 1).float().numpy() * 255).astype(np.uint8)

for sid in SIDS:
    # ─── Build regions at latent res exactly like train.py ───
    M_ag_lat = torch.load(f"{LATENTS}/{sid}_agnostic_mask_latent.pt", weights_only=True).float()
    if M_ag_lat.dim() == 3: M_ag_lat = M_ag_lat.unsqueeze(0)       # (1, 1, 128, 96)
    M_full = (M_ag_lat > 0.5).float()                               # binary

    tm = torch.load(f"{LATENTS}/{sid}_target_mask.pt", weights_only=True).float()
    if tm.dim() == 3: tm = tm.unsqueeze(0)
    garment_prior = (tm > 0.5).float()                              # core

    wm_lat = torch.load(f"{LATENTS}/{sid}_warped_mask_128.pt", weights_only=True).float()
    if wm_lat.dim() == 3: wm_lat = wm_lat.unsqueeze(0)
    # warped_mask is used in silhouette slot; target_mask is for core loss

    repair_band = (M_full - garment_prior).clamp(0, 1)

    # uncertain_band (dilate-erode ring around garment_prior)
    tm_dil = F.max_pool2d(garment_prior, 5, 1, 2)
    tm_ero = -F.max_pool2d(-garment_prior, 5, 1, 2)
    uncertain_band = (tm_dil - tm_ero).clamp(0, 1)

    keep_mask = 1.0 - M_full

    # densepose → body_latent
    dp = torch.load(f"{LATENTS}/{sid}_densepose.pt", weights_only=True).float()
    # (3, 1024, 768) → body mask → interpolate(area) to (128, 96)
    dp = dp.unsqueeze(0) if dp.dim() == 3 else dp                   # (1, 3, 1024, 768)
    body_img = (dp.sum(dim=1, keepdim=True) > 0.02).float()
    body_latent_soft = F.interpolate(body_img, size=(128, 96), mode="area")
    body_latent = (body_latent_soft > 0.5).float()

    body_repair = (repair_band * body_latent).clamp(0, 1)
    bg_repair   = (repair_band * (1.0 - body_latent)).clamp(0, 1)

    # ─── Counts ───
    def count(m):
        return int(m.sum().item())
    print(f"\n=== {sid}  (latent 128×96, total={128*96}) ===")
    print(f"  M_full (agnostic)  : {count(M_full):>6}")
    print(f"  core (garment)     : {count(garment_prior):>6}")
    print(f"  repair_band        : {count(repair_band):>6}")
    print(f"  body_repair        : {count(body_repair):>6}  ({100*count(body_repair)/max(count(repair_band),1):.1f}% of repair)")
    print(f"  bg_repair          : {count(bg_repair):>6}  ({100*count(bg_repair)/max(count(repair_band),1):.1f}% of repair)")
    print(f"  uncertain_band     : {count(uncertain_band):>6}")
    print(f"  keep               : {count(keep_mask):>6}")
    # Check core + body_rep + bg_rep = M_full (should hold)
    reconstructed = garment_prior + body_repair + bg_repair
    overlap = (reconstructed > 1).float()
    missing = (M_full - reconstructed.clamp(0, 1)).clamp(0).float()
    print(f"  overlap (should=0): {count(overlap):>6}")
    print(f"  missing (should=0): {count(missing):>6}")

    # ─── Visualize at image resolution (upsample for clarity) ───
    def up(m):
        return F.interpolate(m, size=(1024, 768), mode="nearest")[0, 0]

    # Load GT person image (from panel/ if exists, else decode)
    gt_path = f"{BASE}/vtonautoresearch/runs/vton_20260423_133455/final/panel/{sid}_gt.png"
    if os.path.exists(gt_path):
        gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.uint8)
    else:
        gt = np.zeros((1024, 768, 3), dtype=np.uint8)

    # Row 1: GT | agnostic mask (from _agnostic_mask.pt full-res for fidelity) | warped fullres | body at image res
    M_ag_full = torch.load(f"{LATENTS}/{sid}_agnostic_mask.pt", weights_only=True).float()
    if M_ag_full.dim() == 3: M_ag_full = M_ag_full[0]
    ag_img = (M_ag_full.numpy() * 255).astype(np.uint8)
    ag_img_rgb = np.stack([ag_img]*3, axis=-1)

    wm_full = torch.load(f"{LATENTS}/{sid}_warped_fullres_mask.pt", weights_only=True).float()
    if wm_full.dim() == 3: wm_full = wm_full[0]
    wm_img = (wm_full.numpy() * 255).astype(np.uint8)
    wm_img_rgb = np.stack([wm_img]*3, axis=-1)

    body_up = up(body_latent.unsqueeze(0) if body_latent.dim()==3 else body_latent)
    body_img_rgb = (body_up.numpy() * 255).astype(np.uint8)
    body_img_rgb = np.stack([body_img_rgb]*3, axis=-1)

    row1 = np.concatenate([gt, ag_img_rgb, wm_img_rgb, body_img_rgb], axis=1)

    # Row 2: GT overlay with regions colored: green=core, orange=body_rep, red=bg_rep
    over = gt.astype(np.float32)
    core_up = up(garment_prior).numpy() > 0.5
    body_rep_up = up(body_repair).numpy() > 0.5
    bg_rep_up   = up(bg_repair).numpy() > 0.5
    over[core_up]     = 0.5 * over[core_up]     + 0.5 * np.array([0, 255, 0])
    over[body_rep_up] = 0.5 * over[body_rep_up] + 0.5 * np.array([255, 200, 0])
    over[bg_rep_up]   = 0.5 * over[bg_rep_up]   + 0.5 * np.array([255, 0, 0])
    over = over.clip(0, 255).astype(np.uint8)

    # Also show each region separately as white-on-black
    def mask_rgb(mask_up_bool):
        im = (mask_up_bool * 255).astype(np.uint8)
        return np.stack([im]*3, axis=-1)

    row2 = np.concatenate([over, mask_rgb(core_up), mask_rgb(body_rep_up), mask_rgb(bg_rep_up)], axis=1)

    panel = np.concatenate([row1, row2], axis=0)
    # downscale for output
    pil = Image.fromarray(panel).resize((panel.shape[1]//2, panel.shape[0]//2))
    pil.save(f"{OUT}/{sid}_regions.png")
    print(f"  saved {OUT}/{sid}_regions.png")
