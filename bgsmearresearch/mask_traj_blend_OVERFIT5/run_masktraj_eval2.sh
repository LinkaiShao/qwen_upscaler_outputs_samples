#!/bin/bash
# run_masktraj_eval2.sh — TWO-TRACK deploy eval on the EXISTING mask-traj checkpoint (no retraining).
# Renders 4 conditions with the two-track state quarantine (x_base pure run37 + x_edit garment track,
# hard composite x_edit=x_base*(1-M)+x_edit*M every step). Writes numbers_twotrack.txt +
# IDENTITY_panel_twotrack.png (does NOT overwrite the single-track files). ALL under the run folder.
# LAUNCH: nohup bash run_masktraj_eval2.sh > bgsmearresearch/mask_traj_blend_OVERFIT5/eval2.out 2>&1 &
set +e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
V=/home/link/venvs/ootd/bin/python
IDS="00006_00,00008_00,00013_00,00017_00,00034_00"
RES="bgsmearresearch/mask_traj_blend_OVERFIT5"
CDIR="${MASK_TRAJ_CDIR:-$(for d in $(ls -dt runs/MASKTRAJ_*/ 2>/dev/null); do [ -f "${d}final/garment_vel_denoiser.pt" ] && { echo "${d%/}"; break; }; done)}"
CK="$CDIR/final/garment_vel_denoiser.pt"
mkdir -p "$RES"
echo "[e2] START $(date)  ckpt=$CK"
[ -f "$CK" ] || { echo "NO ckpt $CK" | tee "$RES/numbers_twotrack.txt"; exit 1; }

wait_gpu(){ for k in $(seq 1 200); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${m:-99999}" -lt 6000 ] && return 0; sleep 15; done; }
kill_rn(){ for p in $(pgrep -f "train.py" 2>/dev/null); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -q "RUN_NAME=$1" && kill -9 "$p" 2>/dev/null; done; sleep 5; }

# render_cond IMGDIR MODE VLIM
render_cond(){
  local IMGDIR=$1 MODE=$2 VLIM=$3
  local BLEND=1 TWO=1 WRONG=0 ZERO=0
  if [ "$MODE" = "baseline" ]; then BLEND=0; TWO=0; fi
  [ "$MODE" = "wrong" ] && WRONG=1
  [ "$MODE" = "zero" ] && ZERO=1
  rm -rf "$IMGDIR"; mkdir -p "$IMGDIR"; wait_gpu
  USE_MASK_TRAJ_BLEND=$BLEND MASK_TRAJ_TWOTRACK=$TWO MASK_TRAJ_BLOCKS=4 MASK_TRAJ_INIT_PATH="$CK" \
    MASK_TRAJ_DEPLOY_WRONG=$WRONG MASK_TRAJ_ZERO=$ZERO POSE_USE_WARPED_RGB=1 BRIDGE_WRONG_SIDS="$IDS" \
    SAVE_PANEL_PREDICT=0 SAVE_DEPLOY_IMG=1 DEPLOY_IMG_DIR="$IMGDIR" \
    USE_GARMENT_OOTD=0 USE_POSE_RESIDUAL=0 USE_GARMENT_KV=0 USE_MULTI_BLOCK_INJ=0 USE_GARMENT_BRIDGE=0 USE_SOAR=0 USE_V6=1 \
    VAL_USE_RESERVED=1 VAL_LIMIT=$VLIM TIME_BUDGET=1400 RUN_DIR="$CDIR" WHICH=none \
    bash render_panels.sh >/dev/null 2>&1
  sleep 90
  for j in $(seq 1 60); do [ "$(ls "$IMGDIR"/*_rawpred.png 2>/dev/null | wc -l)" -ge "$VLIM" ] && break; pgrep -f "train.py" >/dev/null || break; sleep 12; done
  kill_rn "RENDER_$(basename "$CDIR")"
}

# ── SMOKE (2 ids): baseline vs correct -> bg/skin OUTSIDE garment (CLOUD/farbg/skinL1) must be
#    IDENTICAL (the two-track quarantine proof). garment (garL1) may differ. ──
echo "[e2] SMOKE (2 ids) baseline vs correct — quarantine proof (bg/skin identical) $(date)"
render_cond "$RES/sk_baseline" baseline 2
render_cond "$RES/sk_correct"  correct  2
{
  echo "=== TWO-TRACK SMOKE (2 ids): bg/skin OUTSIDE garment (CLOUD/farbg/skinL1) must be IDENTICAL ==="
  echo "cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1"
  echo -n "BASELINE : "; $V metric_rawpred.py "$RES/sk_baseline" n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "CORRECT  : "; $V metric_rawpred.py "$RES/sk_correct"  n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo "(CLOUD/farbg/skinL1 identical -> quarantine works; garL1 may differ = garment changed)"
} | tee "$RES/smoke_twotrack.txt"

# ── FULL: 4 conditions on 5 ids ──
echo "[e2] FULL (5 ids) 4 conditions $(date)"
render_cond "$RES/tt_baseline" baseline 5
render_cond "$RES/tt_correct"  correct  5
render_cond "$RES/tt_wrong"    wrong    5
render_cond "$RES/tt_zero"     zero     5

OUT="$RES/numbers_twotrack.txt"; : > "$OUT"
{
  echo "=== TWO-TRACK MASKED TRAJECTORY BLEND — deploy on 5 reserved. $(date) ==="
  echo "cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1"
  echo -n "run37 BASELINE      : "; $V metric_rawpred.py "$RES/tt_baseline" n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend CORRECT       : "; $V metric_rawpred.py "$RES/tt_correct"  n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend WRONG garment : "; $V metric_rawpred.py "$RES/tt_wrong"    n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend ZERO (empty ga): "; $V metric_rawpred.py "$RES/tt_zero"     n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo "QUARANTINE PROOF: bg/skin OUTSIDE garment (CLOUD/farbg/skinL1) must be IDENTICAL across ALL 4 rows"
  echo "  (x_edit hard-composited to x_base outside the mask every step). NOT zero==baseline."
  echo "GARMENT: WIN = garL1(correct) < baseline [improved], < wrong [identity], < zero [garment-driven, not generic]."
} >> "$OUT" 2>&1

RES="$RES" $V - >> "$OUT" 2>&1 <<'PY'
import os, numpy as np
from PIL import Image, ImageDraw
B="/home/link/Desktop/Code/fashion gen testing"; RES=os.environ["RES"]
IDS=["00006_00","00008_00","00013_00","00017_00","00034_00"]
def lab(a,t):
    im=Image.fromarray(a).copy(); d=ImageDraw.Draw(im); d.rectangle([0,0,300,30],fill=(0,0,0)); d.text((5,8),t,fill=(255,255,255)); return np.asarray(im)
rows=[]
for s in IDS:
    try:
        gt=np.asarray(Image.open(f"{B}/VITON-HD-dataset/test/image/{s}.jpg").convert("RGB")); H,W=gt.shape[:2]
        def g(sub): return np.asarray(Image.open(f"{B}/vtonautoresearch/{RES}/{sub}/{s}_rawpred.png").convert("RGB").resize((W,H)))
        p=np.concatenate([lab(g("tt_baseline"),"run37"),lab(g("tt_correct"),"correct"),lab(g("tt_wrong"),"WRONG"),lab(g("tt_zero"),"ZERO(empty gar)"),lab(gt,"GT")],1)
        rows.append(np.asarray(Image.fromarray(p).resize((int(W*5*0.22),int(H*0.22)))))
    except Exception as e: print("skip",s,e)
if rows: Image.fromarray(np.concatenate(rows,0)).save(f"{B}/vtonautoresearch/{RES}/IDENTITY_panel_twotrack.png"); print("panel ok")
PY
echo "two-track eval done $(date)" > "$RES/DONE_twotrack.txt"
echo "[e2] ALL DONE $(date)"
