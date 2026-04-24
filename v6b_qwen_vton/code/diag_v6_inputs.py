"""
Visualize ALL v6-lite training inputs for the 5 benchmark samples.

Replicates train.py's preprocessing exactly for USE_V6=1:
  USE_AGNOSTIC_INPAINT=1 (iterated blur-fill of masked region at latent res)
  V6_ZERO_G_CORE=1 (zero agnostic inside erode(warped, V6_R_IN))
  AGNOSTIC_FILE=_tight_agnostic_latent.pt
  AGNOSTIC_MASK_FILE=_tight_agnostic_mask_latent.pt

Output (decoded to images) per sample:
  {sid}_agn_raw.png             — the tight agnostic as-cached, decoded
  {sid}_agn_after_inpaint.png   — after USE_AGNOSTIC_INPAINT blur-fill
  {sid}_agn_after_zero.png      — after V6_ZERO_G_CORE (this is the model's input)
  {sid}_garment.png             — garment latent decoded
  {sid}_silhouette.png          — silhouette slot (warped_mask broadcast) decoded
  {sid}_classes.png             — M_g/M_s/M_b/M_k colored overlay on GT
  {sid}_all_inputs_panel.png    — composite of the above

And one overall panel: v6_inputs_all.png
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur as gblur

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"
OUT = f"{BASE}/vtonautoresearch/diag_out/v6_inputs"
os.makedirs(OUT, exist_ok=True)

SIDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]

# v6-lite env constants
V6_R_OUT = 7    # latent px
V6_R_IN  = 2
INPAINT_K = 7
INPAINT_SIGMA = 2.0
INPAINT_ITERS = 20

DEVICE = "cuda:1"; DTYPE = torch.bfloat16

print("Loading VAE...")
vae = AutoencoderKLQwenImage.from_pretrained(
    f"{BASE}/Qwen-Image-Edit-2511", subfolder="vae", torch_dtype=DTYPE
).to(DEVICE).eval()
for p in vae.parameters(): p.requires_grad_(False)
m_v = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1).to(DEVICE, DTYPE)
s_v = torch.tensor(vae.config.latents_std).view(1,16,1,1,1).to(DEVICE, DTYPE)

def decode(lat_tensor):
    """(16, 128, 96) → u8 (1024, 768, 3)"""
    lat = lat_tensor.unsqueeze(0).to(DEVICE, DTYPE)
    if lat.dim() == 4: lat = lat.unsqueeze(2)
    with torch.no_grad():
        img = vae.decode(lat * s_v + m_v, return_dict=False)[0][:, :, 0]
    img = (img.clamp(-1, 1)[0] + 1) / 2.0
    return (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

def up_u8(mask_lat):
    """(1, 128, 96) → u8 (1024, 768)"""
    m = mask_lat.float()
    if m.dim() == 3: m = m.unsqueeze(0)
    m_up = F.interpolate(m, size=(1024, 768), mode="nearest")[0, 0]
    return (m_up.numpy() * 255).astype(np.uint8)

def label_img(arr_hw3, text):
    """stamp text in top-left corner"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(arr_hw3)
    d = ImageDraw.Draw(im)
    d.rectangle([(0, 0), (len(text)*10+8, 22)], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255))
    return np.array(im)

all_rows = []

for sid in SIDS:
    # Load inputs
    agn_lat = torch.load(f"{LATENTS}/{sid}_tight_agnostic_latent.pt", weights_only=True).float()       # (16, 128, 96)
    M_ag_lat = torch.load(f"{LATENTS}/{sid}_tight_agnostic_mask_latent.pt", weights_only=True).float()  # (1, 128, 96)
    if M_ag_lat.dim() == 2: M_ag_lat = M_ag_lat.unsqueeze(0)
    M_ag = (M_ag_lat > 0.5).float().unsqueeze(0)                                                  # (1, 1, 128, 96)

    gar_lat = torch.load(f"{LATENTS}/{sid}_garment_latent.pt", weights_only=True).float()         # (16, 128, 96)
    wm_128  = torch.load(f"{LATENTS}/{sid}_warped_mask_128.pt", weights_only=True).float()        # (1, 128, 96)
    if wm_128.dim() == 2: wm_128 = wm_128.unsqueeze(0)
    wm_lat  = wm_128.unsqueeze(0)                                                                  # (1, 1, 128, 96)
    wm_bin  = (wm_lat > 0.5).float()

    pg_lat  = torch.load(f"{LATENTS}/{sid}_parse_garment_latent.pt", weights_only=True).float()   # (1, 128, 96)
    ps_lat  = torch.load(f"{LATENTS}/{sid}_parse_skin_latent.pt",    weights_only=True).float()
    pb_lat  = torch.load(f"{LATENTS}/{sid}_parse_bg_latent.pt",      weights_only=True).float()
    for x in (pg_lat, ps_lat, pb_lat):
        if x.dim() == 2: x = x.unsqueeze(0)
    pg_b = (pg_lat.unsqueeze(0) > 0.5).float()
    ps_b = (ps_lat.unsqueeze(0) > 0.5).float()

    # ── Step 1: raw agnostic (as cached) ──
    agn_raw = agn_lat.clone()
    agn_raw_img = decode(agn_raw)

    # ── Step 2: apply USE_AGNOSTIC_INPAINT (latent-space blur-fill) ──
    agn_inp = agn_lat.unsqueeze(0).clone()               # (1, 16, 128, 96)
    keep = (1 - M_ag)
    for _ in range(INPAINT_ITERS):
        blurred = gblur(agn_inp.float(), kernel_size=[INPAINT_K, INPAINT_K], sigma=INPAINT_SIGMA).to(agn_inp.dtype)
        agn_inp = agn_inp * keep + blurred * M_ag
    agn_after_inpaint_img = decode(agn_inp[0])

    # ── Step 3: V6_ZERO_G_CORE — zero inside erode(warped, r_in) ──
    r_in = V6_R_IN
    core_t = -F.max_pool2d(-wm_bin, 2*r_in+1, 1, r_in)   # erode
    M_core = (core_t > 0.5).float()                       # (1, 1, 128, 96)
    agn_v6 = agn_inp * (1.0 - M_core)                     # (1, 16, 128, 96)
    agn_final_img = decode(agn_v6[0])

    # ── M_edit (dilated warped for v6 edit support) ──
    r_out = V6_R_OUT
    M_edit = F.max_pool2d(wm_bin, 2*r_out+1, 1, r_out).clamp(0, 1)

    # ── Routing classes ──
    M_g = (M_edit * pg_b).clamp(0, 1)
    M_s = (M_edit * ps_b * (1.0 - M_g)).clamp(0, 1)
    M_b = (M_edit - M_g - M_s).clamp(0, 1)
    M_k = (1.0 - M_edit)

    # ── Silhouette slot content (warped mask broadcast — what model sees in silhouette slot) ──
    # Just visualize the single-channel warped mask
    wm_u8 = up_u8(wm_128)
    sil_viz = np.stack([wm_u8, wm_u8, wm_u8], axis=-1)

    # ── Garment ──
    gar_img = decode(gar_lat)

    # ── GT ──
    gt_path = f"{BASE}/vtonautoresearch/runs/vton_20260423_133455/final/panel/{sid}_gt.png"
    gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.uint8)

    # ── Routing class overlay on GT ──
    col = gt.astype(np.float32)
    M_g_up = (F.interpolate(M_g, size=(1024, 768), mode="nearest")[0, 0].numpy() > 0.5)
    M_s_up = (F.interpolate(M_s, size=(1024, 768), mode="nearest")[0, 0].numpy() > 0.5)
    M_b_up = (F.interpolate(M_b, size=(1024, 768), mode="nearest")[0, 0].numpy() > 0.5)
    M_core_up = (F.interpolate(M_core, size=(1024, 768), mode="nearest")[0, 0].numpy() > 0.5)
    col[M_g_up] = 0.5 * col[M_g_up] + 0.5 * np.array([0, 255, 0])
    col[M_s_up] = 0.5 * col[M_s_up] + 0.5 * np.array([255, 200, 0])
    col[M_b_up] = 0.5 * col[M_b_up] + 0.5 * np.array([255, 0, 0])
    # outline M_core in black
    # (skip — visible from overlay already)
    class_viz = col.clip(0, 255).astype(np.uint8)

    # ── M_core viz ──
    core_u8 = up_u8(M_core[0])
    core_viz = np.stack([core_u8, core_u8, core_u8], axis=-1)

    # Save individual files
    Image.fromarray(label_img(gt, "GT")).save(f"{OUT}/{sid}_00_gt.png")
    Image.fromarray(label_img(agn_raw_img, "agnostic raw")).save(f"{OUT}/{sid}_01_agn_raw.png")
    Image.fromarray(label_img(agn_after_inpaint_img, "agn after inpaint")).save(f"{OUT}/{sid}_02_agn_after_inpaint.png")
    Image.fromarray(label_img(agn_final_img, "agn after V6 zero-core  ← MODEL INPUT")).save(f"{OUT}/{sid}_03_agn_final.png")
    Image.fromarray(label_img(gar_img, "garment latent")).save(f"{OUT}/{sid}_04_garment.png")
    Image.fromarray(label_img(sil_viz, "silhouette slot (warped mask)")).save(f"{OUT}/{sid}_05_silhouette.png")
    Image.fromarray(label_img(core_viz, "M_core (eroded warped)")).save(f"{OUT}/{sid}_06_M_core.png")
    Image.fromarray(label_img(class_viz, "classes G=g S=o B=r K=untinted")).save(f"{OUT}/{sid}_07_classes.png")

    # Per-sample composite row (horizontal strip)
    row = np.concatenate([
        label_img(gt, "GT"),
        label_img(agn_raw_img, "agn raw"),
        label_img(agn_after_inpaint_img, "after inpaint"),
        label_img(agn_final_img, "after V6 zero-core"),
        label_img(gar_img, "garment"),
        label_img(sil_viz, "silhouette"),
        label_img(class_viz, "classes"),
    ], axis=1)
    Image.fromarray(row).save(f"{OUT}/{sid}_all_inputs_panel.png")
    all_rows.append(row)

    print(f"  {sid} done")

# Overall panel
stacked = np.concatenate(all_rows, axis=0)
H, W = stacked.shape[:2]
# Downscale for file size
pil = Image.fromarray(stacked).resize((W//2, H//2))
pil.save(f"{OUT}/v6_inputs_all.png")
print(f"\nSaved to {OUT}/")
print(f"Per-sample panels: {{sid}}_all_inputs_panel.png")
print(f"Overall: v6_inputs_all.png")
