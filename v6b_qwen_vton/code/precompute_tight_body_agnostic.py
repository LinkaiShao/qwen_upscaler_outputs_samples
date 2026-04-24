"""
#1 (tight mask) + #3 (body-aware fill) preprocessing.

For each sample:
  - tight_mask_pixel = dilate(warped_fullres_mask, 10px)      # at pixel res
  - source image     = decode(person_latent)                   # paired-mode source
  - body_fill        = iterate_blur_and_paste(source, keep=~tight_mask)
                        → body colors propagate INTO tight_mask region
  - new_agnostic     = outside-tight: source                   (unchanged)
                       inside-tight : body_fill                (body-ish content)
  - VAE-encode → `_tight_agnostic_latent.pt`

Also save `_tight_agnostic_mask_latent.pt` = tight_mask at latent resolution (128×96).

Training uses these two files in place of `_agnostic_latent.pt` and
`_agnostic_mask_latent.pt`. The tighter mask means the repair band is almost
empty → halo target region collapses.
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
FILL_ITERS = int(os.environ.get("FILL_ITERS", "60"))
FILL_SIGMA = float(os.environ.get("FILL_SIGMA", "8.0"))
print(f"margin_px={MARGIN_PX}, fill_iters={FILL_ITERS}, fill_sigma={FILL_SIGMA}", flush=True)

sys.path.insert(0, f"{BASE}/vtonautoresearch")
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

vae = AutoencoderKLQwenImage.from_pretrained(
    f"{BASE}/Qwen-Image-Edit-2511", subfolder="vae", torch_dtype=DTYPE
).to(DEVICE).eval()
for p in vae.parameters(): p.requires_grad_(False)
m_v = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1).to(DEVICE, DTYPE)
s_v = torch.tensor(vae.config.latents_std).view(1,16,1,1,1).to(DEVICE, DTYPE)

def gk(k, s, dev, dt):
    ax = torch.arange(k, device=dev, dtype=torch.float32) - (k-1)/2
    g = torch.exp(-(ax**2)/(2*s*s)); g = g/g.sum()
    return (g[:, None] * g[None, :]).to(dt)
def gblur(x, s):
    k = int(2 * round(3 * s) + 1); kern = gk(k, s, x.device, x.dtype)
    return F.conv2d(x, kern.expand(x.shape[1], 1, k, k), padding=k // 2, groups=x.shape[1])

def decode_person(plat):
    lat = plat.unsqueeze(0).unsqueeze(2).to(DEVICE, DTYPE) * s_v + m_v
    with torch.no_grad():
        img = vae.decode(lat, return_dict=False)[0][:, :, 0]
    return ((img[0].clamp(-1, 1) + 1) / 2)   # (3, 1024, 768) in [0, 1]

for sid in SIDS:
    # Source person (paired-mode proxy) at 1024×768
    plat = torch.load(f"{LATENTS}/{sid}_person_latent.pt", weights_only=True).to(DEVICE, DTYPE)
    source = decode_person(plat).float().unsqueeze(0)               # (1, 3, 1024, 768), [0, 1]

    # warped garment mask at pixel res
    wm = torch.load(f"{LATENTS}/{sid}_warped_fullres_mask.pt", weights_only=True).float().to(DEVICE)
    if wm.dim() == 3: wm = wm.unsqueeze(0)                           # (1, 1, 1024, 768)
    wm_bin = (wm > 0.5).float()

    # Tight mask via dilation
    kernel = 2 * MARGIN_PX + 1
    tight = F.max_pool2d(wm_bin, kernel, 1, MARGIN_PX).clamp(0, 1)   # (1, 1, 1024, 768)

    # Body-fill inside tight mask via iterative propagation from OUTSIDE tight
    keep = (1 - tight)
    result = source.clone()
    for _ in range(FILL_ITERS):
        blurred = gblur(result, FILL_SIGMA)
        result = result * keep + blurred * tight

    # VAE-encode
    x = (result * 2 - 1).unsqueeze(2).to(DTYPE)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample()[:, :, 0]
    lat_norm = (lat - m_v[0, :, 0]) / s_v[0, :, 0]
    out_lat = f"{LATENTS}/{sid}_tight_agnostic_latent.pt"
    torch.save(lat_norm[0].contiguous().cpu(), out_lat)

    # Downsample tight mask to latent resolution (8× downsample → 128×96)
    tight_lat = F.avg_pool2d(tight, 8, 8).clamp(0, 1)                 # (1, 1, 128, 96)
    out_m = f"{LATENTS}/{sid}_tight_agnostic_mask_latent.pt"
    torch.save(tight_lat[0].cpu().float(), out_m)

    # visualize
    img_viz = (result[0].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    Image.fromarray(img_viz).save(f"{BASE}/vtonautoresearch/diag_out/contour_source/{sid}_13_tight_body.png")

    print(f"  {sid} → lat:{out_lat}  mask:{out_m}  "
          f"tight_area={int(tight.sum().item())}  wide_area_orig="
          f"{int((torch.load(f'{LATENTS}/{sid}_agnostic_mask.pt', weights_only=True).float() > 0.5).sum().item())}",
          flush=True)

print("Done.", flush=True)
