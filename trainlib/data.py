"""Latent utils + dataset (split from train.py)."""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

BASE = "/home/link/Desktop/Code/fashion gen testing"
LOCAL_CACHE = f"{BASE}/vtonautoresearch/local_cache"

# Set by train.py at startup (after it computes the id list / prompt from env + filesystem).
TRAIN_IDS = []
FIXED_PROMPT = ""


def pack_latents(lat, B, C, H, W):
    return lat.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)


def unpack_latents(lat, B, C, H, W):
    return lat.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def vae_decode_to_pil(vae, latent_norm, vae_device, dtype):
    """(16, H, W) normalized latent → PIL RGB image."""
    m = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    s = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    x = latent_norm.unsqueeze(0).unsqueeze(2).to(vae_device, dtype=dtype)  # (1,16,1,H,W)
    x = x * s + m
    with torch.no_grad():
        img = vae.decode(x, return_dict=False)[0][:, :, 0]                  # (1,3,H_img,W_img)
    img = img.clamp(-1, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(img)


def precompute_rough_pils(latent_cache_dir, vae, vae_device, dtype):
    cache = {}
    for sid in TRAIN_IDS:
        rough_file = os.environ.get("ROUGH_FILE", "_degraded_rough_latent.pt")
        lat = torch.load(os.path.join(latent_cache_dir, f"{sid}{rough_file}"), weights_only=True)
        pil = vae_decode_to_pil(vae, lat, vae_device, dtype)
        cache[sid] = pil
    return cache


def precompute_prompt_embeds(pipe, latent_cache_dir, local_cache, rough_pils, device, dtype):
    """Per-sample Qwen2.5-VL encode of [agnostic, pose, rough, garment] + fixed prompt.
    exp396: 4 source images fed to the text encoder so it has semantic access to all
    four signals, including the garment (which is no longer in the transformer's
    spatial image stream).
    """
    cache = {}
    for sid in TRAIN_IDS:
        agn_px = torch.load(os.path.join(latent_cache_dir, f"{sid}_agnostic_pixel.pt"), weights_only=True)
        agn_pil = Image.fromarray((agn_px.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
        pose_pil = Image.open(os.path.join(local_cache, f"pose_pixel_{sid}.png")).convert("RGB")
        rough_pil = rough_pils[sid]
        garment_px = torch.load(os.path.join(latent_cache_dir, f"{sid}_garment_pixel.pt"), weights_only=True)
        garment_pil = Image.fromarray((garment_px.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

        # exp486+: drop rough_pil from VL (was double-injecting artifacts via spatial
        # and semantic paths; rough is now only in the spatial slot). Keep pose_pil
        # and garment_pil and agn_pil. Pose goes ONLY via VL (the spatial pose slot
        # is off-manifold densepose and is dropped via SLOT_ORDER), so removing pose
        # from VL would eliminate pose info entirely.
        with torch.no_grad():
            if int(os.environ.get("ROUGH_AS_CONTROL", "0")):
                vl_images = [agn_pil, pose_pil, rough_pil, garment_pil]
            else:
                vl_images = [agn_pil, pose_pil, garment_pil]
            pe, pm = pipe.encode_prompt(
                image=vl_images,
                prompt=FIXED_PROMPT,
                device=device,
                num_images_per_prompt=1,
            )
        cache[sid] = (pe.detach().clone().to(device, dtype=dtype),
                      pm.detach().clone().to(device, dtype=torch.long))
    return cache


def load_pose_latents(local_cache, device, dtype):
    cache = {}
    for sid in TRAIN_IDS:
        lat = torch.load(os.path.join(local_cache, f"pose_latent_{sid}.pt"), weights_only=True)
        cache[sid] = lat.to(device, dtype=dtype)                              # (16, 128, 96)
    return cache


class VTONDataset(Dataset):
    def __init__(self, args, split="train"):
        self.latent_dir = args.latent_cache_dir
        # Filter by TRAIN_IDS + latent cache availability. Ignore split — caches
        # for all IDs live in the same cache dir regardless of which split the
        # original VITON-HD image came from.
        required = ["_person_latent.pt", "_garment_latent.pt", "_degraded_rough_latent.pt",
                    "_agnostic_latent.pt", "_agnostic_mask_latent.pt", "_target_mask.pt",
                    "_warped_mask_128.pt"]
        self.image_ids = [i for i in TRAIN_IDS if all(
            os.path.exists(os.path.join(self.latent_dir, f"{i}{s}")) for s in required)]
        print(f"VTONDataset [{split}]: {len(self.image_ids)} samples")

    def __len__(self): return len(self.image_ids)

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        L = lambda s: torch.load(os.path.join(self.latent_dir, f"{i}{s}"), weights_only=True)
        item = {
            "image_id":      i,
            "person_latent": L("_person_latent.pt"),
            "rough_latent":  L(os.environ.get("ROUGH_FILE", "_degraded_rough_latent.pt")),
            "garment_latent":L("_garment_latent.pt"),
            "agnostic_latent":       L(os.environ.get("AGNOSTIC_FILE", "_agnostic_latent.pt")),
            "warped_mask":           L(f"_warped_mask_128{os.environ.get('WARP_SUFFIX', '')}.pt"
                                        if os.path.exists(os.path.join(self.latent_dir, f"{i}_warped_mask_128{os.environ.get('WARP_SUFFIX', '')}.pt"))
                                        else "_warped_mask_128.pt"),
            "agnostic_mask_latent":  L(os.environ.get("AGNOSTIC_MASK_FILE", "_agnostic_mask_latent.pt")),
            "target_mask":           L("_target_mask.pt"),
        }
        if int(os.environ.get("USE_DP_SPLIT", "0")) or int(os.environ.get("USE_BG_HINT", "0")):
            item["densepose"] = L("_densepose.pt")                               # (3, 1024, 768)
        if int(os.environ.get("USE_V6", "0")):
            # parse files only exist for the 5 VTON IDs; gracefully skip when missing
            # so USE_V6=1 works for full_train mode (v6 routing fires per-sample only
            # when parse is available — when missing, v6_heads stay loaded but unused).
            for k, sfx in [("parse_garment", "_parse_garment_latent.pt"),
                           ("parse_skin",    "_parse_skin_latent.pt"),
                           ("parse_bg",      "_parse_bg_latent.pt")]:
                p = os.path.join(self.latent_dir, f"{i}{sfx}")
                if os.path.exists(p):
                    item[k] = torch.load(p, weights_only=True)
        if int(os.environ.get("USE_GARMENT_NET", "0")):
            item["garment_pixel"] = L("_garment_pixel.pt")                       # (3, 1024, 768)
        return item


def collate_fn(batch):
    keys = ["person_latent", "rough_latent", "garment_latent",
            "agnostic_latent", "agnostic_mask_latent", "target_mask",
            "warped_mask"]
    out = {
        **{k: torch.stack([b[k] for b in batch]) for k in keys},
        "image_id": [b["image_id"] for b in batch],
    }
    if "densepose" in batch[0]:
        out["densepose"] = torch.stack([b["densepose"] for b in batch])
    if "parse_garment" in batch[0]:
        out["parse_garment"] = torch.stack([b["parse_garment"] for b in batch])
        out["parse_skin"]    = torch.stack([b["parse_skin"]    for b in batch])
        out["parse_bg"]      = torch.stack([b["parse_bg"]      for b in batch])
    if "garment_pixel" in batch[0]:
        out["garment_pixel"] = torch.stack([b["garment_pixel"] for b in batch])
    return out
