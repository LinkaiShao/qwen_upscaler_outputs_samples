"""
Per-region quantitative metrics for a run's final panel preds.
Splits repair band via densepose: body_repair vs bg_repair.

Outputs per-sample and overall:
  mean |pred - gt| L1 (scale 0-255 per-channel sum) by region:
    core, body_repair, bg_repair, keep (outside)
  95th percentile diff in each region
  # pixels > 50 diff in each region

Usage:
  python diag_region_metrics.py --run-dir <run>/final
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F, numpy as np
from PIL import Image

BASE = "/home/link/Desktop/Code/fashion gen testing"
LATENTS = f"{BASE}/my_vton_cache/latents"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = args.run_dir
    panel_dir = os.path.join(run, "panel")

    if not os.path.isdir(panel_dir):
        print(f"no panel dir at {panel_dir}")
        return

    sids = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
    totals = {k: [0.0, 0] for k in ("core", "body_rep", "bg_rep", "keep", "halo_bg")}
    per_sample = {}

    for sid in sids:
        pp = os.path.join(panel_dir, f"{sid}_pred.png")
        gp = os.path.join(panel_dir, f"{sid}_gt.png")
        if not os.path.exists(pp) or not os.path.exists(gp):
            continue
        pred = np.array(Image.open(pp).convert("RGB")).astype(np.float32)
        gt   = np.array(Image.open(gp).convert("RGB")).astype(np.float32)
        H, W = pred.shape[:2]
        diff = np.abs(pred - gt).sum(axis=-1)          # (H, W), 0-765

        ag = torch.load(f"{LATENTS}/{sid}_agnostic_mask.pt", weights_only=True).float()
        wm = torch.load(f"{LATENTS}/{sid}_warped_fullres_mask.pt", weights_only=True).float()
        dp = torch.load(f"{LATENTS}/{sid}_densepose.pt", weights_only=True).float()
        if ag.dim() == 3: ag = ag[0]
        if wm.dim() == 3: wm = wm[0]
        if dp.shape[-2:] != (H, W):
            dp = F.interpolate(dp.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
        ag_np = ag.numpy() > 0.5
        wm_np = wm.numpy() > 0.5
        body_np = (dp.sum(0).numpy() > 0.02)

        core    = ag_np & wm_np
        repair  = ag_np & ~wm_np
        body_rep = repair & body_np
        bg_rep   = repair & ~body_np
        keep     = ~ag_np

        # halo ring: pixels in agnostic, within 3px of outer agnostic boundary
        ag_t = torch.from_numpy(ag_np.astype(np.float32))[None, None]
        ag_eroded = (-F.max_pool2d(-ag_t, 7, 1, 3))[0, 0].numpy() > 0.5
        halo_ring = ag_np & ~ag_eroded                  # 3-px inner band of agnostic
        halo_bg   = halo_ring & ~body_np & ~wm_np       # halo ring overlapping bg_rep

        row = {}
        for name, mask in [("core", core), ("body_rep", body_rep),
                           ("bg_rep", bg_rep), ("keep", keep),
                           ("halo_bg", halo_bg)]:
            if mask.sum() == 0:
                row[name] = (0.0, 0.0, 0, 0)
                continue
            d = diff[mask]
            mean = float(d.mean())
            p95  = float(np.percentile(d, 95))
            n    = int(mask.sum())
            n50  = int((d > 50).sum())
            row[name] = (mean, p95, n, n50)
            totals[name][0] += mean * n
            totals[name][1] += n
        per_sample[sid] = row

    # Report
    print(f"\n=== run: {run}")
    header = f"{'sid':<12} {'core_mean':>10} {'body_mean':>10} {'bg_mean':>10} {'keep_mean':>10} "
    print(header)
    for sid, row in per_sample.items():
        print(f"{sid:<12} {row['core'][0]:>10.2f} {row['body_rep'][0]:>10.2f} "
              f"{row['bg_rep'][0]:>10.2f} {row['keep'][0]:>10.2f}")
    print()
    print("=== OVERALL (pixel-weighted mean diff) ===")
    for k, (s, n) in totals.items():
        if n > 0:
            print(f"  {k:<10} mean = {s/n:.3f}  (n_px = {n})")

if __name__ == "__main__":
    main()
