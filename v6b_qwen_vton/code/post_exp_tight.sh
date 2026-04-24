#!/bin/bash
# Post-exp but with tight agnostic env vars set for panel generation.
set -e
RUN_DIR="$1"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR/final" ]; then
  echo "usage: post_exp_tight.sh <run_dir with /final>"; exit 1
fi
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch"

export AGNOSTIC_FILE=_tight_agnostic_latent.pt
export AGNOSTIC_MASK_FILE=_tight_agnostic_mask_latent.pt

echo "=== generate_panel.py (tight agnostic) ==="
/home/link/venvs/ootd/bin/python generate_panel.py --run-dir "$RUN_DIR/final" 2>&1 | tail -30

echo ""
echo "=== diag_trajectory_diffmap.py ==="
OUT_DIR="$RUN_DIR/final/traj_diffmap"
ROUGH_FILE=_degraded_rough_latent.pt /home/link/venvs/ootd/bin/python diag_trajectory_diffmap.py --run-dir "$RUN_DIR/final" --out-dir "$OUT_DIR" 2>&1 | tail -20

echo ""
echo "=== DONE — outputs in $RUN_DIR/final/{panel, traj_diffmap} ==="
