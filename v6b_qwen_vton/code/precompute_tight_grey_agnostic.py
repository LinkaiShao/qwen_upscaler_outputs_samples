"""
Tight mask + GREY fill (not body-propagated).

  - tight_mask_pixel = dilate(warped_fullres_mask, 10px)
  - source image     = decode(person_latent)
  - new_agnostic     = outside tight_mask: source
                       inside tight_mask : 0.5 flat grey
  - VAE-encode → `_tight_grey_agnostic_latent.pt`

Tests whether flat grey fill was the speckle-preventer in exp524, but with
tight mask → small repair band → halo shouldn't show up.
"""
import os, sys
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"
SIDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
DEVICE = "cuda:1"; DTYPE = torch.bfloat16
MARGIN_PX = int(os.environ.get("TIGHT_MARGIN_PX", "10"))

sys.path.insert(0, f"{BASE}/vtonautoresearch")
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

vae = AutoencoderKLQwenImage.from_pretrained(
    f"{BASE}/Qwen-Image-Edit-2511", subfolder="vae", torch_dtype=DTYPE
).to(DEVICE).eval()
for p in vae.parameters(): p.requires_grad_(False)
m_v = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1).to(DEVICE, DTYPE)
s_v = torch.tensor(vae.config.latents_std).view(1,16,1,1,1).to(DEVICE, DTYPE)

def decode_person(plat):
    lat = plat.unsqueeze(0).unsqueeze(2).to(DEVICE, DTYPE) * s_v + m_v
    with torch.no_grad():
        img = vae.decode(lat, return_dict=False)[0][:, :, 0]
    return ((img[0].clamp(-1, 1) + 1) / 2)

for sid in SIDS:
    plat = torch.load(f"{LATENTS}/{sid}_person_latent.pt", weights_only=True).to(DEVICE, DTYPE)
    source = decode_person(plat).float().unsqueeze(0)

    wm = torch.load(f"{LATENTS}/{sid}_warped_fullres_mask.pt", weights_only=True).float().to(DEVICE)
    if wm.dim() == 3: wm = wm.unsqueeze(0)
    wm_bin = (wm > 0.5).float()
    tight = F.max_pool2d(wm_bin, 2 * MARGIN_PX + 1, 1, MARGIN_PX).clamp(0, 1)

    result = source * (1 - tight) + 0.5 * tight   # grey inside tight, source outside

    x = (result * 2 - 1).unsqueeze(2).to(DTYPE)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample()[:, :, 0]
    lat_norm = (lat - m_v[0, :, 0]) / s_v[0, :, 0]
    out_lat = f"{LATENTS}/{sid}_tight_grey_agnostic_latent.pt"
    torch.save(lat_norm[0].contiguous().cpu(), out_lat)

    # Tight mask at latent res (already computed in precompute_tight_body_agnostic — reuse)
    # But ensure it exists:
    tight_lat_path = f"{LATENTS}/{sid}_tight_agnostic_mask_latent.pt"
    if not os.path.exists(tight_lat_path):
        tight_lat = F.avg_pool2d(tight, 8, 8).clamp(0, 1)
        torch.save(tight_lat[0].cpu().float(), tight_lat_path)

    img_viz = (result[0].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    Image.fromarray(img_viz).save(f"{BASE}/vtonautoresearch/diag_out/contour_source/{sid}_14_tight_grey.png")
    print(f"  {sid} → {out_lat}  mean={lat_norm.mean():.3f} std={lat_norm.std():.3f}", flush=True)

print("Done.", flush=True)
