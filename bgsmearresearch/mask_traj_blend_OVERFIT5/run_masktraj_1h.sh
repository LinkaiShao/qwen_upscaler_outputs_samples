#!/bin/bash
# run_masktraj_1h.sh — SINGLE run: masked-trajectory-blend 1h (5-ID overfit) -> deploy eval
# (run37 baseline / blend-CORRECT / blend-WRONG / blend-ZERO) -> 5-panel + garL1/bg/skin.
# EVERYTHING under bgsmearresearch/mask_traj_blend_OVERFIT5/. NOTHING in /tmp.
# LAUNCH: nohup bash run_masktraj_1h.sh > bgsmearresearch/mask_traj_blend_OVERFIT5/wrapper.out 2>&1 &
set +e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
V=/home/link/venvs/ootd/bin/python
IDS="00006_00,00008_00,00013_00,00017_00,00034_00"
RES="bgsmearresearch/mask_traj_blend_OVERFIT5"
mkdir -p "$RES"
export RUN_NAME="MASKTRAJ_$(date +%m%d_%H%M)"
CDIR="runs/$RUN_NAME"
LOG="$RES/${RUN_NAME}.log"
echo "[mt] START $RUN_NAME $(date)  log=$LOG"

wait_gpu(){ for k in $(seq 1 200); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${m:-99999}" -lt 6000 ] && return 0; sleep 15; done; }
kill_rn(){ for p in $(pgrep -f "train.py" 2>/dev/null); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -q "RUN_NAME=$1" && kill -9 "$p" 2>/dev/null; done; sleep 5; }

# ── 1) train 1h (5-ID overfit) ──
TIME_BUDGET="${TIME_BUDGET:-3600}" MASK_TRAJ_LOG="$LOG" bash run_masktraj.sh
echo "[mt] train done rc=$? $(date)"
if [ ! -f "$CDIR/final/garment_vel_denoiser.pt" ]; then
  echo "NO denoiser ckpt ($CDIR/final) -> crashed, see $LOG" > "$RES/numbers.txt"
  echo "crashed $(date)" > "$RES/DONE.txt"; exit 0
fi

# ── 2) deploy render: baseline / correct / wrong / zero (blend applied at every Euler step in _fwd) ──
render(){  # $1=imgdir  $2=mode(baseline|correct|wrong|zero)
  local BLEND=1 WRONG=0 ZERO=0
  [ "$2" = "baseline" ] && BLEND=0
  [ "$2" = "wrong" ] && WRONG=1
  [ "$2" = "zero" ] && ZERO=1
  rm -f "$CDIR"/deploy_imgs/*_rawpred.png 2>/dev/null; wait_gpu
  USE_MASK_TRAJ_BLEND=$BLEND MASK_TRAJ_BLOCKS=4 MASK_TRAJ_INIT_PATH="$CDIR/final/garment_vel_denoiser.pt" \
    MASK_TRAJ_DEPLOY_WRONG=$WRONG MASK_TRAJ_ZERO=$ZERO POSE_USE_WARPED_RGB=1 BRIDGE_WRONG_SIDS="$IDS" \
    USE_GARMENT_OOTD=0 USE_POSE_RESIDUAL=0 USE_GARMENT_KV=0 USE_MULTI_BLOCK_INJ=0 USE_GARMENT_BRIDGE=0 USE_SOAR=0 USE_V6=1 \
    VAL_USE_RESERVED=1 VAL_LIMIT=5 TIME_BUDGET=1400 RUN_DIR="$CDIR" WHICH=none \
    bash render_panels.sh >/dev/null 2>&1
  sleep 120
  for j in $(seq 1 60); do [ "$(ls "$CDIR"/deploy_imgs/*_rawpred.png 2>/dev/null | wc -l)" -ge 5 ] && break; pgrep -f "train.py" >/dev/null || break; sleep 12; done
  kill_rn "RENDER_$(basename "$CDIR")"
  mkdir -p "$1"; cp "$CDIR"/deploy_imgs/*_rawpred.png "$1"/ 2>/dev/null
}
render "$RES/imgs_baseline" baseline
render "$RES/imgs_correct"  correct
render "$RES/imgs_wrong"    wrong
render "$RES/imgs_zero"     zero

OUT="$RES/numbers.txt"; : > "$OUT"
{
  echo "=== MASKED TRAJECTORY BLEND — deploy on 5 reserved (5-ID overfit). run=$RUN_NAME $(date) ==="
  echo "cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1"
  echo -n "run37 BASELINE      : "; $V metric_rawpred.py "$RES/imgs_baseline" n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend CORRECT       : "; $V metric_rawpred.py "$RES/imgs_correct"  n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend WRONG garment : "; $V metric_rawpred.py "$RES/imgs_wrong"    n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "blend ZERO (den off): "; $V metric_rawpred.py "$RES/imgs_zero"     n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo "WIN = garL1(correct) < garL1(baseline) [garment improved] AND < garL1(wrong) [garment-specific]."
  echo "By construction bg/skin (CLOUD/farbg/skinL1) MUST be ~identical across all rows; ZERO row MUST == BASELINE."
} >> "$OUT" 2>&1

# ── 5-panel [run37 | correct | WRONG | ZERO | GT] ──
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
        p=np.concatenate([lab(g("imgs_baseline"),"run37"),lab(g("imgs_correct"),"correct"),lab(g("imgs_wrong"),"WRONG"),lab(g("imgs_zero"),"ZERO=run37?"),lab(gt,"GT")],1)
        rows.append(np.asarray(Image.fromarray(p).resize((int(W*5*0.22),int(H*0.22)))))
    except Exception as e: print("skip",s,e)
if rows: Image.fromarray(np.concatenate(rows,0)).save(f"{B}/vtonautoresearch/{RES}/IDENTITY_panel.png"); print("panel ok")
PY
echo "mask-traj 1h + deploy eval done $(date)" > "$RES/DONE.txt"
echo "[mt] ALL DONE $(date)"
