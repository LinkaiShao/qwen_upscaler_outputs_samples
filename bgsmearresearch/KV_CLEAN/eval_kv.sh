#!/bin/bash
# eval_kv.sh — KV-aware eval for a co-trained garment-K/V run (does NOT require garment_adapter.pt).
# Renders the 5 reserved ids in 3 conditions via the DEPLOY halo_eval rollout, writes a panel.
#   Usage: bash eval_kv.sh RUN_DIR OUT_DIR   (RUN_DIR has final/garment_kv_branch.pt + final/garment_kv_procs.pt)
# Conditions:
#   baseline : run37 only (USE_GARMENT_KV=0)                    — the bar
#   correct  : trained K/V ON  (branch+procs loaded, KV appended) — the try-on
#   bypass   : K/V loaded but XATTN_BYPASS=1 (branch off)       — isolates the K/V contribution vs run37
# NOTE: a WRONG-garment eval for the K/V needs a wrong-garment feed at inference (the branch reads
# batch.garment_latent) — not wired here; the TRAINING kvc_wrong metric (correct vs wrong) is the
# reliable garment-specificity signal. This panel shows: does the trained K/V CHANGE the garment
# (correct vs baseline/bypass) and does it keep bg/skin clean.
set +e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
V=/home/link/venvs/ootd/bin/python
IDS="00006_00,00008_00,00013_00,00017_00,00034_00"
R37="runs/BGBAND37_012549"
RUN_DIR="${1:?RUN_DIR}"; OUT_DIR="${2:?OUT_DIR}"; RUN_DIR="${RUN_DIR%/}"; OUT_DIR="${OUT_DIR%/}"
FINAL="$RUN_DIR/final"; KVB="$FINAL/garment_kv_branch.pt"; KVP="$FINAL/garment_kv_procs.pt"
KV_BLOCKS="${GARMENT_KV_BLOCKS:-44,50,55,59}"
mkdir -p "$OUT_DIR"
echo "[eval_kv] START $(date) RUN_DIR=$RUN_DIR blocks=$KV_BLOCKS"
[ -f "$KVB" ] || { echo "NO garment_kv_branch.pt at $KVB" | tee "$OUT_DIR/numbers.txt"; exit 1; }
wait_gpu(){ for k in $(seq 1 240); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${m:-99999}" -lt 6000 ] && return 0; sleep 15; done; }
kill_rn(){ for p in $(pgrep -f "train.py" 2>/dev/null); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -q "RUN_NAME=$1" && kill -9 "$p" 2>/dev/null; done; sleep 5; }

# render_cond IMGDIR COND
render_cond(){
  local IMGDIR=$1 COND=$2 RD="$RUN_DIR" KV=1 BYP=0 WRONG=0
  if [ "$COND" = "baseline" ]; then RD="$R37"; KV=0; fi
  [ "$COND" = "bypass" ] && BYP=1
  [ "$COND" = "wrong" ]  && WRONG=1
  rm -rf "$IMGDIR"; mkdir -p "$IMGDIR"; wait_gpu
  USE_GARMENT_KV=$KV GARMENT_KV_BLOCKS="$KV_BLOCKS" GARMENT_KV_GATE_INIT_LOGIT=0.0 GARMENT_KV_FP32=1 \
    GARMENT_KV_BRANCH_INIT_PATH="$KVB" GARMENT_KV_PROCS_INIT_PATH="$KVP" XATTN_BYPASS=$BYP \
    GARMENT_KV_DEPLOY_WRONG=$WRONG BRIDGE_WRONG_SIDS="$IDS" \
    GARMENT_SCHEDULE=0 GARMENT_SLOT_CORRUPT="${EVAL_SLOT_CORRUPT:-}" POSE_USE_WARPED_RGB=1 \
    SAVE_PANEL_PREDICT=0 SAVE_DEPLOY_IMG=1 DEPLOY_IMG_DIR="$IMGDIR" \
    USE_GARMENT_ADAPTER=0 USE_GARMENT_OOTD=0 USE_POSE_RESIDUAL=0 USE_MULTI_BLOCK_INJ=0 USE_GARMENT_BRIDGE=0 \
    USE_MASK_TRAJ_BLEND=0 USE_SOAR=0 USE_V6=1 \
    VAL_USE_RESERVED=1 VAL_LIMIT=5 TIME_BUDGET=1400 RUN_DIR="$RD" WHICH=none \
    bash render_panels.sh >/dev/null 2>&1
  sleep 90
  for j in $(seq 1 70); do [ "$(ls "$IMGDIR"/*_rawpred.png 2>/dev/null | wc -l)" -ge 5 ] && break; pgrep -f "train.py" >/dev/null || break; sleep 12; done
  kill_rn "RENDER_$(basename "$RD")"
}
echo "[eval_kv] rendering baseline / correct / wrong / bypass $(date)"
render_cond "$OUT_DIR/ev_baseline" baseline
render_cond "$OUT_DIR/ev_correct"  correct
render_cond "$OUT_DIR/ev_wrong"    wrong
render_cond "$OUT_DIR/ev_bypass"   bypass

OUT="$OUT_DIR/numbers.txt"; : > "$OUT"
mline(){ $V metric_rawpred.py "$1" n 2>&1 | grep '^MEAN' | sed 's/MEAN//'; }
{ echo "=== KV EVAL — $RUN_DIR — cols: CLOUD editBGdL editNBdL editAlldL wholeBGdL farBGdL ePSNR eSSIM garL1 skinL1 ==="
  echo "run37 BASELINE : $(mline "$OUT_DIR/ev_baseline")"
  echo "KV CORRECT     : $(mline "$OUT_DIR/ev_correct")"
  echo "KV WRONG       : $(mline "$OUT_DIR/ev_wrong")"
  echo "KV BYPASS      : $(mline "$OUT_DIR/ev_bypass")"
  echo "SUCCESS = garL1(CORRECT) < garL1(WRONG) [reads content] AND < garL1(BYPASS) [KV helps], bg/skin clean."; } >> "$OUT" 2>&1
cat "$OUT"
# 4-panel [run37 | KV-correct | KV-bypass | GT]
OUT_DIR="$OUT_DIR" $V - >> "$OUT" 2>&1 <<'PY'
import os, numpy as np
from PIL import Image, ImageDraw
B="/home/link/Desktop/Code/fashion gen testing"; OD=os.environ["OUT_DIR"]
IDS=["00006_00","00008_00","00013_00","00017_00","00034_00"]
def lab(a,t):
    im=Image.fromarray(a).copy(); d=ImageDraw.Draw(im); d.rectangle([0,0,300,30],fill=(0,0,0)); d.text((5,8),t,fill=(255,255,255)); return np.asarray(im)
rows=[]
for s in IDS:
    try:
        gt=np.asarray(Image.open(f"{B}/VITON-HD-dataset/test/image/{s}.jpg").convert("RGB")); H,W=gt.shape[:2]
        def g(sub): return np.asarray(Image.open(f"{OD}/{sub}/{s}_rawpred.png").convert("RGB").resize((W,H)))
        p=np.concatenate([lab(g("ev_baseline"),"run37"),lab(g("ev_correct"),"KV-correct"),lab(g("ev_wrong"),"KV-WRONG"),lab(g("ev_bypass"),"KV-bypass"),lab(gt,"GT")],1)
        rows.append(np.asarray(Image.fromarray(p).resize((int(W*5*0.22),int(H*0.22)))))
    except Exception as e: print("panel skip",s,e)
if rows: Image.fromarray(np.concatenate(rows,0)).save(f"{OD}/IDENTITY_panel.png"); print("panel ->",f"{OD}/IDENTITY_panel.png")
PY
echo "kv eval done $(date)" > "$OUT_DIR/DONE_eval.txt"
echo "[eval_kv] ALL DONE $(date) -> $OUT_DIR/numbers.txt  $OUT_DIR/IDENTITY_panel.png"
