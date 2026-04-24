#!/usr/bin/env python3
"""
Generate 5-image benchmark panel for a trained run.

Usage:
    python generate_panel.py --run-dir vtonautoresearch/runs/<run_name>/final
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib import cm

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

BASE = "/home/link/Desktop/Code/fashion gen testing"
BENCHMARK_IDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
BENCHMARK_SEEDS = {"00006_00": 101, "00008_00": 102, "00013_00": 103, "00017_00": 104, "00034_00": 105}


def load_batch(sample_id, latent_dir, text_dir, device):
    L = lambda s: torch.load(os.path.join(latent_dir, f"{sample_id}{s}"), weights_only=True).unsqueeze(0)
    T = lambda s: torch.load(os.path.join(text_dir, f"{sample_id}{s}"), weights_only=True).unsqueeze(0)
    return {
        "image_id": [sample_id],
        "person_latent": L("_person_latent.pt"),
        "garment_latent": L("_garment_latent.pt"),
        "agnostic_latent": L(os.environ.get("AGNOSTIC_FILE", "_agnostic_latent.pt")),
        "rough_latent": L(os.environ.get("ROUGH_FILE", "_degraded_rough_latent.pt")),
        "densepose": L("_densepose.pt"),
        "agnostic_mask_latent": L(os.environ.get("AGNOSTIC_MASK_FILE", "_agnostic_mask_latent.pt")),
        "target_mask": L("_target_mask.pt"),
        "warped_mask": L("_warped_mask_128.pt"),
        "warped_silhouette_latent": L("_warped_silhouette_latent.pt")
            if os.path.exists(os.path.join(latent_dir, f"{sample_id}_warped_silhouette_latent.pt")) else None,
        "prompt_embeds": T("_prompt_embeds.pt"),
        "prompt_mask": T("_prompt_mask.pt"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.output or os.path.join(run_dir, "panel")
    os.makedirs(out_dir, exist_ok=True)

    # Read region weights from the run's config.json (written by train.py)
    # Fall back to exp395/exp401 defaults if absent.
    cfg_path = os.path.join(os.path.dirname(run_dir), "config.json")
    run_weights = {"w_out": 0.05, "w_core": 1.0, "w_rep": 0.25, "w_bdy": 1.0}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            run_cfg = json.load(f)
        # train.py doesn't currently serialize region_weights directly; check env-var
        # hints saved via log, else reuse defaults. (Future: persist w_core/w_rep to config.json)
        for k in ("w_out", "w_core", "w_rep", "w_bdy"):
            if k in run_cfg:
                run_weights[k] = float(run_cfg[k])
    print(f"Using region weights: {run_weights}")

    # Load inference adapter
    sys.path.insert(0, run_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", os.path.join(run_dir, "inference.py"))
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

    # Config — read rank/alpha from run's config.json if available
    _run_rank = 32
    _run_alpha = 32
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            _rc = json.load(f)
        _run_rank = _rc.get("rank", 32)
        _run_alpha = _rc.get("alpha", 32)

    class Config:
        pretrained_model = f"{BASE}/Qwen-Image-Edit-2511"
        rank = _run_rank
        alpha = _run_alpha
        init_lora_weights = "gaussian"
        lora_targets = ["to_k", "to_q", "to_v", "to_out.0"]

    device = "cuda:0"
    print("Loading model...")
    model = inference.load_model(run_dir, device, Config)

    # Load VAE for decoding
    print("Loading VAE...")
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(Config.pretrained_model, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda:1").eval()

    def decode(latents):
        lat = latents.to("cuda:1", dtype=torch.bfloat16)
        if lat.dim() == 4: lat = lat.unsqueeze(2)
        m = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1).to(lat.device, lat.dtype)
        s = torch.tensor(vae.config.latents_std).view(1,16,1,1,1).to(lat.device, lat.dtype)
        with torch.no_grad():
            img = vae.decode(lat * s + m, return_dict=False)[0][:, :, 0]
        img = img[0].clamp(-1, 1).permute(1, 2, 0).cpu().float().numpy()
        return ((img + 1) / 2 * 255).astype(np.uint8)

    latent_dir = f"{BASE}/my_vton_cache/latents"
    text_dir = f"{BASE}/my_vton_cache/text"

    all_rows = []  # list of (gt, pred, composite, garment, rough) per sample
    heat_rows = []  # list of (gt, pred, err_heat, err_masked_heat) per sample
    per_sample_l1 = {}
    for sid in BENCHMARK_IDS:
        print(f"\n=== Generating {sid} ===")
        batch = load_batch(sid, latent_dir, text_dir, device)
        seed = BENCHMARK_SEEDS[sid]

        settings = {"num_inference_steps": 20, "pose_mode": "normal"}
        output = inference.predict_sample(model, batch, device, seed, settings)
        pred_latents = output["pred_latents"]

        # Decode everything
        pred_img = decode(pred_latents.to("cuda:1"))
        gt_img = decode(batch["person_latent"].to("cuda:1"))
        garment_img = decode(batch["garment_latent"].to("cuda:1"))
        rough_img = decode(batch["rough_latent"].to("cuda:1"))
        agnostic_img = decode(batch["agnostic_latent"].to("cuda:1"))

        # Save individual
        Image.fromarray(pred_img).save(os.path.join(out_dir, f"{sid}_pred.png"))
        Image.fromarray(gt_img).save(os.path.join(out_dir, f"{sid}_gt.png"))

        all_rows.append((gt_img, agnostic_img, garment_img, rough_img, pred_img))

        # ── Training-loss heatmap: uses this RUN's actual region weights ──
        # weight_map = w_out + (w_core - w_out)*M_core + w_rep*M_repair + w_bdy*core_boundary
        M_ag = batch["agnostic_mask_latent"].to(torch.float32)          # (1, 1, 128, 96)
        M_tg = batch["target_mask"].to(torch.float32)                   # (1, 1, 128, 96)
        M_full = (M_ag > 0.5).to(torch.float32)
        M_core = (M_tg > 0.5).to(torch.float32)
        M_repair = (M_full - M_core).clamp(0, 1)
        core_dilated  = F.max_pool2d(M_core, kernel_size=7, stride=1, padding=3)
        core_boundary = (core_dilated - M_core).clamp(0, 1)
        w_out  = run_weights["w_out"]
        w_core = run_weights["w_core"]
        w_rep  = run_weights["w_rep"]
        w_bdy  = run_weights["w_bdy"]
        wm_lat = w_out + (w_core - w_out) * M_core + w_rep * M_repair + w_bdy * core_boundary

        Hi, Wi = gt_img.shape[:2]
        wm_img = F.interpolate(wm_lat, size=(Hi, Wi), mode="bilinear", align_corners=False)
        wm_np  = wm_img[0, 0].cpu().numpy()                              # (Hi, Wi), ≥ 0.05

        err_rgb = np.abs(pred_img.astype(np.float32) - gt_img.astype(np.float32)) / 255.0  # (H, W, 3)
        err_gray = err_rgb.mean(axis=-1)                                 # (H, W)
        err_weighted = err_gray * wm_np                                  # training image L1 integrand

        per_sample_l1[sid] = {
            "unweighted_mean":  float(err_gray.mean()),
            "weighted_mean":    float(err_weighted.mean()),
            "core_mean":        float((err_gray * F.interpolate(M_core, (Hi, Wi), mode="bilinear", align_corners=False)[0,0].cpu().numpy()).sum()
                                       / max(F.interpolate(M_core, (Hi, Wi), mode="bilinear", align_corners=False)[0,0].cpu().numpy().sum(), 1.0)),
        }

        # Weight map visualization (col 3): normalize by max so highest weight = brightest
        wm_vis_norm = wm_np / max(wm_np.max(), 1e-6)
        wm_heat = (cm.viridis(wm_vis_norm)[..., :3] * 255.0).astype(np.uint8)

        # Error heatmap (col 4): raw per-pixel L1
        err_norm = err_gray / max(err_gray.max(), 1e-6)
        err_heat = (cm.inferno(err_norm)[..., :3] * 255.0).astype(np.uint8)

        # Training-loss heatmap (col 5): weighted L1, normalized per-sample
        loss_norm = err_weighted / max(err_weighted.max(), 1e-6)
        loss_heat = (cm.inferno(loss_norm)[..., :3] * 255.0).astype(np.uint8)

        heat_rows.append((gt_img, pred_img, wm_heat, err_heat, loss_heat))

    # Build grid panel: 5 rows × 5 cols (gt, agnostic, garment, rough, pred)
    h, w = all_rows[0][0].shape[:2]
    panel = np.zeros((5 * h, 5 * w, 3), dtype=np.uint8)
    labels = ["GT", "Agnostic", "Garment", "Rough", "Pred"]
    for r, row in enumerate(all_rows):
        for c, img in enumerate(row):
            panel[r*h:(r+1)*h, c*w:(c+1)*w] = img

    Image.fromarray(panel).save(os.path.join(out_dir, "panel.png"))
    print(f"\nSaved panel to {out_dir}/panel.png")
    print(f"Columns: {' | '.join(labels)}")

    # Heatmap panel: 5 rows × 5 cols (gt, pred, weight_map, raw L1 err, training-weighted L1 loss)
    heat_panel = np.zeros((5 * h, 5 * w, 3), dtype=np.uint8)
    heat_labels = ["GT", "Pred", "Weight map", "L1 err", "Training loss"]
    for r, row in enumerate(heat_rows):
        for c, img in enumerate(row):
            heat_panel[r*h:(r+1)*h, c*w:(c+1)*w] = img
    Image.fromarray(heat_panel).save(os.path.join(out_dir, "heatmap.png"))
    print(f"Saved heatmap to {out_dir}/heatmap.png")
    print(f"Heatmap columns: {' | '.join(heat_labels)}")

    # Per-sample loss summary
    print("\nPer-sample L1 (lower = better):")
    print(f"{'sample':<12}{'unweighted':>12}{'weighted':>12}{'core':>10}")
    for sid, v in per_sample_l1.items():
        print(f"{sid:<12}{v['unweighted_mean']:>12.4f}{v['weighted_mean']:>12.4f}{v['core_mean']:>10.4f}")
    mean_unw = np.mean([v["unweighted_mean"] for v in per_sample_l1.values()])
    mean_wgt = np.mean([v["weighted_mean"] for v in per_sample_l1.values()])
    mean_core = np.mean([v["core_mean"] for v in per_sample_l1.values()])
    print(f"{'mean':<12}{mean_unw:>12.4f}{mean_wgt:>12.4f}{mean_core:>10.4f}")

    with open(os.path.join(out_dir, "loss_summary.json"), "w") as f:
        json.dump({"per_sample": per_sample_l1,
                   "mean_unweighted": float(mean_unw),
                   "mean_weighted": float(mean_wgt),
                   "mean_core": float(mean_core)}, f, indent=2)


if __name__ == "__main__":
    main()
