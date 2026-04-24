#!/usr/bin/env python3
"""
Per-step pred-vs-GT diffmap across the full 20-step denoising trajectory for 3
samples. At every step, composites the running x0 estimate onto the agnostic,
decodes to image space, and renders a red-heat diffmap vs the GT person image,
with agn-mask boundary (green) and warped-mask boundary (cyan) overlays.

Outputs:
  diag_out/traj_diffmap/{sid}_step{NN}.png   — single panel: pred | gt | diffmap
  diag_out/traj_diffmap/{sid}_grid.png       — row of 20 diffmaps for one sample
  diag_out/traj_diffmap/all_grid.png         — 3×20 overall grid of diffmaps

Run:
  python diag_trajectory_diffmap.py --run-dir runs/vton_20260423_162948/final
"""
import argparse, os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

from diffusers import FlowMatchEulerDiscreteScheduler

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"
IDS = ["00006_00", "00017_00", "00034_00"]
SEEDS = {"00006_00": 101, "00008_00": 102, "00013_00": 103, "00017_00": 104, "00034_00": 105}
NUM_STEPS = 20


def _pack(lat, B, C, H, W):
    return lat.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

def _unpack(lat, B, C, H, W):
    return lat.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def ring_px(mask_pix, ring=1):
    m = torch.from_numpy(mask_pix.astype(np.float32))
    while m.dim() < 4: m = m.unsqueeze(0)
    mb = (m > 0.5).float()
    d = F.max_pool2d(mb, 2*ring+1, 1, ring)
    e = -F.max_pool2d(-mb, 2*ring+1, 1, ring)
    return ((d - e).clamp(0, 1))[0, 0].numpy().astype(bool)


def load_batch(sid):
    L = lambda s: torch.load(f"{LATENTS}/{sid}{s}", weights_only=True).unsqueeze(0)
    return {
        "image_id": [sid],
        "agnostic_latent": L(os.environ.get("AGNOSTIC_FILE", "_agnostic_latent.pt")),
        "garment_latent":  L("_garment_latent.pt"),
        "rough_latent":    L("_degraded_rough_latent.pt"),
        "agnostic_mask_latent": L(os.environ.get("AGNOSTIC_MASK_FILE", "_agnostic_mask_latent.pt")),
        "target_mask":          L("_target_mask.pt"),
        "person_latent":   L("_person_latent.pt"),
    }


def to_u8(img01):
    return (img01.clamp(0, 1).squeeze().permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)


def build_diffmap(pred_u8, gt_u8, ag_ring_mask, wm_ring_mask):
    """red=diff intensity, green=agn ring, cyan=warp ring. matches diag_where_differ.py."""
    pred_f = pred_u8.astype(np.float32)
    gt_f   = gt_u8.astype(np.float32)
    diff   = np.abs(pred_f - gt_f).sum(axis=-1)
    diff_vis = np.clip(diff / 150.0 * 255, 0, 255).astype(np.uint8)
    H, W = pred_f.shape[:2]
    heat = np.zeros((H, W, 3), dtype=np.uint8)
    heat[..., 0] = diff_vis
    heat[ag_ring_mask] = [0, 200, 0]
    heat[wm_ring_mask] = [100, 200, 255]
    return heat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or f"{BASE}/vtonautoresearch/diag_out/traj_diffmap"
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, args.run_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", os.path.join(args.run_dir, "inference.py"))
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

    _cfg_path = os.path.join(os.path.dirname(args.run_dir), "config.json")
    _run_rank, _run_alpha = 32, 32
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _cf:
            _rc = json.load(_cf)
        _run_rank = _rc.get("rank", 32)
        _run_alpha = _rc.get("alpha", 32)
    print(f"LoRA rank={_run_rank} alpha={_run_alpha}")

    class Config:
        pretrained_model = f"{BASE}/Qwen-Image-Edit-2511"
        rank = _run_rank
        alpha = _run_alpha
        init_lora_weights = "gaussian"
        lora_targets = ["to_k", "to_q", "to_v", "to_out.0"]

    device = "cuda:0"
    td = torch.device(device); wd = torch.bfloat16

    print("Loading model...")
    model = inference.load_model(args.run_dir, device, Config)
    t = model["transformer"]
    prompt_cache = model["prompt_cache"]
    pose_cache = model["pose_cache"]

    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        Config.pretrained_model, subfolder="vae", torch_dtype=wd).to("cuda:1").eval()
    m_stats = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to("cuda:1", wd)
    s_stats = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to("cuda:1", wd)
    def decode(lat):
        lat = lat.to("cuda:1", dtype=wd)
        if lat.dim() == 4: lat = lat.unsqueeze(2)
        with torch.no_grad():
            img = vae.decode(lat * s_stats + m_stats, return_dict=False)[0][:, :, 0]
        return (img.clamp(-1, 1) + 1) / 2.0

    default_order = getattr(inference, "DEFAULT_SLOT_ORDER", [0, 1, 2, 3])
    print(f"DEFAULT_SLOT_ORDER = {default_order}")

    per_sample_diffs = {sid: [] for sid in IDS}  # list of (step, diffmap_u8)
    per_sample_preds = {sid: [] for sid in IDS}
    per_sample_gt    = {}
    ring_cache = {}

    for sid in IDS:
        print(f"\n=== {sid}")
        batch = load_batch(sid)
        al = batch["agnostic_latent"].to(td, dtype=wd)
        gl = batch["garment_latent"].to(td, dtype=wd)
        rl = batch["rough_latent"].to(td, dtype=wd)
        am = batch["agnostic_mask_latent"].to(td, dtype=torch.float32)
        person = batch["person_latent"].to(td, dtype=wd)
        B, C, H, W = al.shape
        M = (am > 0.5).to(dtype=wd)

        pose = pose_cache[sid].unsqueeze(0)

        al_for_slot = al if getattr(inference, "USE_PURE_NOISE", 0) else al
        if getattr(inference, "USE_ROUGH_BLUR_FIXED", 0):
            from torchvision.transforms.functional import gaussian_blur
            _sig = float(getattr(inference, "ROUGH_BLUR_FIXED_SIG", 4.0))
            _k = int(2 * round(2 * _sig) + 1)
            rl = gaussian_blur(rl.float(), [_k, _k], _sig).to(rl.dtype)
        if getattr(inference, "USE_ROUGH_MASKED", 0):
            _wm_rm = torch.load(f"{LATENTS}/{sid}_warped_mask_128.pt", weights_only=True).unsqueeze(0).to(td, dtype=wd)
            if _wm_rm.dim() == 3: _wm_rm = _wm_rm.unsqueeze(1)
            rl = rl * (_wm_rm > 0.5).to(wd)
        if getattr(inference, "USE_AGNOSTIC_INPAINT", 0):
            from torchvision.transforms.functional import gaussian_blur as _gb_ai
            _keep_t = (1 - M)
            _res = al_for_slot.clone()
            for _ in range(20):
                _bl = _gb_ai(_res.float(), kernel_size=[7, 7], sigma=2.0).to(_res.dtype)
                _res = _res * _keep_t + _bl * M
            al_for_slot = _res

        agn_p   = _pack(al_for_slot, B, C, H, W)
        pose_p  = _pack(pose,        B, C, H, W)
        rough_p = _pack(rl,          B, C, H, W)
        gar_p   = _pack(gl,          B, C, H, W)

        if 4 in default_order:
            _wm_t = torch.load(f"{LATENTS}/{sid}_warped_mask_128.pt", weights_only=True).unsqueeze(0).to(td, dtype=wd)
            if _wm_t.dim() == 3: _wm_t = _wm_t.unsqueeze(1)
            _wm_bin = (_wm_t > 0.5).to(wd)
            _sil = _wm_bin.expand(B, C, H, W)
            sil_p = _pack(_sil, B, C, H, W)
        else:
            sil_p = None

        slot_tensors = [agn_p, pose_p, rough_p, gar_p, sil_p]
        cond_seq = [slot_tensors[i] for i in default_order]
        img_shapes_base = [(1, H//2, W//2)] * (1 + len(default_order))

        pe, pm = prompt_cache[sid]
        pe = pe.unsqueeze(0) if pe.dim() == 2 else pe
        pm = pm.unsqueeze(0) if pm.dim() == 1 else pm
        pe = pe.to(td, dtype=wd)
        pm = pm.to(td, dtype=torch.long)
        txt_seq_lens = pm.sum(dim=1).tolist()

        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(Config.pretrained_model, subfolder="scheduler")
        isl = (H//2) * (W//2)
        sl  = (sch.config.max_shift - sch.config.base_shift) / (sch.config.max_image_seq_len - sch.config.base_image_seq_len)
        mu  = isl * sl + (sch.config.base_shift - sl * sch.config.base_image_seq_len)
        sch.set_timesteps(NUM_STEPS, mu=mu)

        g = torch.Generator(device=td).manual_seed(SEEDS[sid])
        noise = torch.randn(al.shape, device=td, dtype=wd, generator=g)
        use_pure_noise = getattr(inference, "USE_PURE_NOISE", 0)
        C_lat = noise if use_pure_noise else (1 - M) * al + M * noise

        # GT decoding + image-resolution ring caches (match panel file size)
        gt_path = os.path.join(args.run_dir, "panel", f"{sid}_gt.png")
        if os.path.exists(gt_path):
            gt_img = np.array(Image.open(gt_path).convert("RGB")).astype(np.uint8)
        else:
            gt_img = to_u8(decode(person.to("cuda:1")))
        per_sample_gt[sid] = gt_img
        Hi, Wi = gt_img.shape[:2]

        ag_full = torch.load(f"{LATENTS}/{sid}_agnostic_mask.pt", weights_only=True).float()
        wm_full = torch.load(f"{LATENTS}/{sid}_warped_fullres_mask.pt", weights_only=True).float()
        if ag_full.dim() == 3: ag_full = ag_full[0]
        if wm_full.dim() == 3: wm_full = wm_full[0]
        ag_np = ag_full.numpy() > 0.5
        wm_np = wm_full.numpy() > 0.5
        ag_ring = ring_px(ag_np, 1)
        wm_ring = ring_px(wm_np, 1)
        ring_cache[sid] = (ag_ring, wm_ring)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=wd):
            for step_idx, ts in enumerate(sch.timesteps):
                C_p_tok = _pack(C_lat, B, C, H, W)
                hidden = torch.cat([C_p_tok] + cond_seq, dim=1)
                sig = (ts / 1000).to(device=td, dtype=wd).expand(B)
                out = t(
                    hidden_states              = hidden,
                    timestep                   = sig,
                    encoder_hidden_states      = pe,
                    encoder_hidden_states_mask = pm,
                    img_shapes                 = [img_shapes_base] * B,
                    txt_seq_lens               = txt_seq_lens,
                    return_dict                = False,
                )[0]
                v_pred_tok = out[:, :C_p_tok.size(1), :]
                v_pred = _unpack(v_pred_tok, B, C, H, W)
                sigma_norm = (ts / 1000).to(td, dtype=wd)
                x0_pred = C_lat - sigma_norm * v_pred

                # Soft outward composite (matches final inference's _M_soft = _M_blur)
                from torchvision.transforms.functional import gaussian_blur as _gb
                M_blur = _gb(M.float(), kernel_size=[7, 7], sigma=2.0).to(M.dtype)
                x0_composed = (1 - M_blur) * al + M_blur * x0_pred
                x0_img = to_u8(decode(x0_composed.to("cuda:1")))
                per_sample_preds[sid].append((step_idx, x0_img))

                diff = build_diffmap(x0_img, gt_img, ag_ring, wm_ring)
                per_sample_diffs[sid].append((step_idx, diff))

                # Save per-step 3-panel (pred | gt | diffmap)
                panel = np.concatenate([x0_img, gt_img, diff], axis=1)
                Image.fromarray(panel).save(f"{out_dir}/{sid}_step{step_idx:02d}.png")

                C_p_stepped = sch.step(v_pred_tok, ts, C_p_tok, return_dict=False)[0]
                C_lat = _unpack(C_p_stepped, B, C, H, W)

        # Per-sample grid: 20 diffmaps in a row (scaled down for readability)
        scale = 4
        h_s, w_s = Hi // scale, Wi // scale
        row_imgs = []
        for step_idx, diff in per_sample_diffs[sid]:
            small = np.array(Image.fromarray(diff).resize((w_s, h_s)))
            row_imgs.append(small)
        row = np.concatenate(row_imgs, axis=1)
        Image.fromarray(row).save(f"{out_dir}/{sid}_grid.png")
        print(f"  saved {sid}_grid.png (20 diffmaps, scale={scale}x smaller)")

    # All-sample grid: 3 rows × 20 cols (scaled)
    scale = 4
    h_s = list(per_sample_diffs.values())[0][0][1].shape[0] // scale
    w_s = list(per_sample_diffs.values())[0][0][1].shape[1] // scale
    all_rows = []
    for sid in IDS:
        row = np.concatenate([np.array(Image.fromarray(d).resize((w_s, h_s)))
                              for _, d in per_sample_diffs[sid]], axis=1)
        all_rows.append(row)
    all_grid = np.concatenate(all_rows, axis=0)
    Image.fromarray(all_grid).save(f"{out_dir}/all_grid.png")
    print(f"\nSaved → {out_dir}/all_grid.png  (rows=samples, cols=steps 0..19)")


if __name__ == "__main__":
    main()
