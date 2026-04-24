"""
Precompute per-sample routing class masks at latent resolution (128×96) for v6.

Classes (LIP/ATR label schema):
  garment (g): {5 upper-clothes, 6 dress, 7 coat, 10 jumpsuit}
  skin (s)   : {13 face, 14 left-arm, 15 right-arm}
  bg (b)     : {0 background}
  other      : {2 hair, 9 pants, 8 socks, ...}     -- treated as keep

Outputs per sample:
  {sid}_parse_garment_latent.pt  (1, 128, 96) float in [0, 1]
  {sid}_parse_skin_latent.pt     (1, 128, 96) float
  {sid}_parse_bg_latent.pt       (1, 128, 96) float

Built from VITON-HD's image-parse-v3/{sid}.png (8-bit label PNG at 1024×768).
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/home/link/Desktop/Code/fashion gen testing"
VITON = f"{BASE}/VITON-HD-dataset"
LATENTS = f"{BASE}/my_vton_cache/latents"

SIDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]

# VITON-HD image-parse-v3 schema (not standard LIP-20):
#   0 bg   2 hair   5 upper-clothes   9 pants   10 NECK   13 face
#   14 L-arm   15 R-arm   (6 dress, 7 coat/long-top appear in some samples)
GARMENT_LABELS = {5, 6, 7}       # upper-body garment (5=tee/top, 6=dress, 7=coat/long-top)
SKIN_LABELS    = {10, 13, 14, 15}  # NECK (10) + face (13) + arms (14, 15)
BG_LABELS      = {0}

def find_parse(sid):
    # check test/ first, then train/
    for split in ("test", "train"):
        p = f"{VITON}/{split}/image-parse-v3/{sid}.png"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(sid)

for sid in SIDS:
    parse_path = find_parse(sid)
    arr = np.array(Image.open(parse_path), dtype=np.int32)           # (1024, 768)
    assert arr.shape == (1024, 768)

    def mask_for(labels):
        m = np.zeros_like(arr, dtype=np.float32)
        for l in labels:
            m = m + (arr == l).astype(np.float32)
        m = np.clip(m, 0, 1)
        return m

    m_g = mask_for(GARMENT_LABELS)
    m_s = mask_for(SKIN_LABELS)
    m_b = mask_for(BG_LABELS)

    def down(m_np):
        t = torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0)         # (1, 1, 1024, 768)
        # 8× average pool → (1, 1, 128, 96)
        t = F.avg_pool2d(t, 8, 8).clamp(0, 1)
        return t[0].float()                                          # (1, 128, 96)

    out_g = down(m_g)
    out_s = down(m_s)
    out_b = down(m_b)

    torch.save(out_g, f"{LATENTS}/{sid}_parse_garment_latent.pt")
    torch.save(out_s, f"{LATENTS}/{sid}_parse_skin_latent.pt")
    torch.save(out_b, f"{LATENTS}/{sid}_parse_bg_latent.pt")

    n = 128 * 96
    print(f"{sid}: g={int((out_g>0.5).sum()):>5} s={int((out_s>0.5).sum()):>5} "
          f"b={int((out_b>0.5).sum()):>5} other={n-int((out_g>0.5).sum())-int((out_s>0.5).sum())-int((out_b>0.5).sum()):>5}")

print("Done.")
