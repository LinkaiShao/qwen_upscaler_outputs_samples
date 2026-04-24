#!/bin/bash
# Post-experiment: generate panel + heatmap + per-step diffmaps for a run.
# Usage: post_exp.sh <run_dir>  (run_dir = .../runs/vton_YYYYMMDD_HHMMSS)
set -e
RUN_DIR="$1"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR/final" ]; then
  echo "usage: post_exp.sh <run_dir with /final>"; exit 1
fi
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch"

echo "=== generate_panel.py (pred + heatmap) ==="
/home/link/venvs/ootd/bin/python generate_panel.py --run-dir "$RUN_DIR/final" 2>&1 | tail -30

echo ""
echo "=== diag_trajectory_diffmap.py (per-step diffmaps for 3 samples) ==="
OUT_DIR="$RUN_DIR/final/traj_diffmap"
/home/link/venvs/ootd/bin/python diag_trajectory_diffmap.py --run-dir "$RUN_DIR/final" --out-dir "$OUT_DIR" 2>&1 | tail -20

echo ""
echo "=== where_differ on final preds ==="
OUT_DIR2="$RUN_DIR/final/where_differ"
mkdir -p "$OUT_DIR2"
/home/link/venvs/ootd/bin/python -c "
import os, sys
sys.path.insert(0, '.')
sys.argv = ['', '$RUN_DIR']
# Inline mini version of diag_where_differ.py using this RUN's panel/
import torch, torch.nn.functional as F, numpy as np
from PIL import Image

RUN = '$RUN_DIR/final'
LATENTS = '/home/link/Desktop/Code/fashion gen testing/my_vton_cache/latents'
OUT = '$OUT_DIR2'
def ring_px(mask_pix, ring=1):
    m = torch.from_numpy(mask_pix.astype(np.float32))
    while m.dim() < 4: m = m.unsqueeze(0)
    mb = (m > 0.5).float()
    d = F.max_pool2d(mb, 2*ring+1, 1, ring)
    e = -F.max_pool2d(-mb, 2*ring+1, 1, ring)
    return ((d - e).clamp(0, 1))[0, 0].numpy().astype(bool)

for sid in ['00006_00', '00017_00', '00034_00']:
    pred = np.array(Image.open(f'{RUN}/panel/{sid}_pred.png').convert('RGB')).astype(np.float32)
    gt   = np.array(Image.open(f'{RUN}/panel/{sid}_gt.png').convert('RGB')).astype(np.float32)
    H, W = pred.shape[:2]
    diff = np.abs(pred - gt).sum(axis=-1)
    vis = np.clip(diff/150.0*255, 0, 255).astype(np.uint8)
    heat = np.zeros((H, W, 3), dtype=np.uint8); heat[..., 0] = vis

    ag = torch.load(f'{LATENTS}/{sid}_agnostic_mask.pt', weights_only=True).float()
    wm = torch.load(f'{LATENTS}/{sid}_warped_fullres_mask.pt', weights_only=True).float()
    dp = torch.load(f'{LATENTS}/{sid}_densepose.pt', weights_only=True).float()
    if ag.dim() == 3: ag = ag[0]
    if wm.dim() == 3: wm = wm[0]
    ag_np = ag.numpy() > 0.5; wm_np = wm.numpy() > 0.5
    body_np = (dp.sum(0).numpy() > 0.02)
    heat[ring_px(ag_np, 1)] = [0, 200, 0]
    heat[ring_px(wm_np, 1)] = [100, 200, 255]
    heat[ring_px(body_np, 1) & ~ring_px(ag_np, 2) & ~ring_px(wm_np, 2)] = [255, 255, 0]

    panel = np.concatenate([pred.astype(np.uint8), gt.astype(np.uint8), heat], axis=1)
    Image.fromarray(panel).save(f'{OUT}/{sid}_pred_gt_diff.png')

    repair = ag_np & ~wm_np
    body_rep = repair & body_np
    bg_rep   = repair & ~body_np
    print(f'{sid}: body_rep_mean_diff={diff[body_rep].mean():.2f}, '
          f'bg_rep_mean_diff={diff[bg_rep].mean():.2f}, '
          f'body_rep_npx={int(body_rep.sum())}, bg_rep_npx={int(bg_rep.sum())}')
" 2>&1 | tail -20

echo ""
echo "=== DONE — outputs in $RUN_DIR/final/{panel, traj_diffmap, where_differ} ==="
