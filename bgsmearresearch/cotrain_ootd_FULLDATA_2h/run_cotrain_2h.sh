#!/bin/bash
# run_cotrain_2h.sh — SINGLE run: co-train run37 LoRA + OOTD garment net 2h, then deploy identity
# eval (run37 baseline vs co-train correct vs co-train WRONG) + [run37|correct|WRONG|GT] panel.
# LAUNCH: nohup bash run_cotrain_2h.sh > /tmp/cotrain_2h.out 2>&1 &
set +e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
V=/home/link/venvs/ootd/bin/python
IDS="00006_00,00008_00,00013_00,00017_00,00034_00"
R37="runs/BGBAND37_012549"
export RUN_NAME="COTRAIN_$(date +%m%d_%H%M)"
CDIR="runs/$RUN_NAME"
FOLD="bgsmearresearch/cotrain_ootd_FULLDATA_2h"; mkdir -p "$FOLD"
echo "[c2h] START $RUN_NAME $(date)"

# ── 1) 2h co-train (FULL data, clean recipe) ──
TIME_BUDGET="${TIME_BUDGET:-7200}" BATCH_SIZE=1 bash run_cotrain.sh
echo "[c2h] train done rc=$? $(date)"

kill_rn(){ for p in $(pgrep -f "train.py" 2>/dev/null); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -q "RUN_NAME=$1" && kill -9 "$p" 2>/dev/null; done; sleep 5; }
wait_gpu(){ for k in $(seq 1 200); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${m:-99999}" -lt 6000 ] && return 0; sleep 15; done; }

if [ ! -f "$CDIR/final/garment_branch.pt" ]; then
  echo "NO co-train ckpt ($CDIR/final) -> train crashed, see logs/$RUN_NAME.log" | tee "$FOLD/numbers.txt"
  echo "crashed $(date)" > "$FOLD/COTRAIN_DONE.txt"; exit 0
fi

# ── 2) deploy identity render: baseline(run37) / correct / WRONG ──
render(){  # $1=imgdir $2=mode(baseline|correct|wrong)
  local WRONG=0 OOTD=1 RD="$CDIR"
  [ "$2" = "baseline" ] && { OOTD=0; RD="$R37"; }
  [ "$2" = "wrong" ] && WRONG=1
  rm -f "$RD"/deploy_imgs/*_rawpred.png 2>/dev/null; wait_gpu
  OOTD_DEPLOY_WRONG=$WRONG BRIDGE_WRONG_SIDS="$IDS" \
    USE_GARMENT_OOTD=$OOTD GARMENT_OOTD_LAYERS=4 GARMENT_OOTD_COPY_BLOCKS="36,44,50,55" GARMENT_OOTD_INJECT_BLOCKS="36,44,50,55" \
    GARMENT_OOTD_DEPTH_SPECIFIC=1 GARMENT_OOTD_GATE_SOURCE=warped_mask GARMENT_OOTD_VG_ZERO_INIT=1 GARMENT_OOTD_GATE_INIT_LOGIT=-2.0 \
    GARMENT_BRANCH_INIT_PATH="$CDIR/final/garment_branch.pt" OOTD_INJECTORS_INIT_PATH="$CDIR/final/ootd_injectors.pt" \
    USE_MULTI_BLOCK_INJ=0 USE_POSE_RESIDUAL=0 USE_GARMENT_BRIDGE=0 USE_SOAR=0 USE_V6=1 \
    VAL_USE_RESERVED=1 VAL_LIMIT=5 TIME_BUDGET=1400 RUN_DIR="$RD" WHICH=none \
    bash render_panels.sh >/dev/null 2>&1
  sleep 120
  for j in $(seq 1 60); do [ "$(ls "$RD"/deploy_imgs/*_rawpred.png 2>/dev/null | wc -l)" -ge 5 ] && break; pgrep -f "train.py" >/dev/null || break; sleep 12; done
  kill_rn "RENDER_$(basename "$RD")"
  mkdir -p "$1"; cp "$RD"/deploy_imgs/*_rawpred.png "$1"/ 2>/dev/null
}
rm -rf /tmp/ct_base /tmp/ct_correct /tmp/ct_wrong
render /tmp/ct_base     baseline
render /tmp/ct_correct  correct
render /tmp/ct_wrong    wrong

OUT="$FOLD/numbers.txt"; : > "$OUT"
{
  echo "=== CO-TRAIN run37 LoRA + OOTD gnet (FULL data 2h) — deploy identity on 5 held-out reserved ==="
  echo "run=$RUN_NAME  $(date)"
  echo "cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1"
  echo -n "run37 BASELINE  : "; $V metric_rawpred.py /tmp/ct_base    n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "co-train CORRECT: "; $V metric_rawpred.py /tmp/ct_correct n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo -n "co-train WRONG  : "; $V metric_rawpred.py /tmp/ct_wrong   n 2>&1 | grep '^MEAN' | sed 's/MEAN//'
  echo "WIN iff garL1(correct) < garL1(baseline) [detail added] AND garL1(correct) < garL1(wrong) [specific garment]."
} >> "$OUT" 2>&1

# ── 3) [run37 | co-train correct | co-train WRONG | GT] panel ──
$V - >> "$OUT" 2>&1 <<'PY'
import numpy as np
from PIL import Image, ImageDraw
B="/home/link/Desktop/Code/fashion gen testing"; IDS=["00006_00","00008_00","00013_00","00017_00","00034_00"]
FOLD=f"{B}/vtonautoresearch/bgsmearresearch/cotrain_ootd_FULLDATA_2h"
def lab(a,t):
    im=Image.fromarray(a).copy(); d=ImageDraw.Draw(im); d.rectangle([0,0,300,30],fill=(0,0,0)); d.text((5,8),t,fill=(255,255,255)); return np.asarray(im)
rows=[]
for s in IDS:
    try:
        gt=np.asarray(Image.open(f"{B}/VITON-HD-dataset/test/image/{s}.jpg").convert("RGB")); H,W=gt.shape[:2]
        b=np.asarray(Image.open(f"/tmp/ct_base/{s}_rawpred.png").convert("RGB").resize((W,H)))
        c=np.asarray(Image.open(f"/tmp/ct_correct/{s}_rawpred.png").convert("RGB").resize((W,H)))
        w=np.asarray(Image.open(f"/tmp/ct_wrong/{s}_rawpred.png").convert("RGB").resize((W,H)))
        p=np.concatenate([lab(b,"run37"),lab(c,"cotrain correct"),lab(w,"cotrain WRONG"),lab(gt,"GT")],1)
        rows.append(np.asarray(Image.fromarray(p).resize((int(W*4*0.26),int(H*0.26)))))
    except Exception as e: print("skip",s,e)
if rows: Image.fromarray(np.concatenate(rows,0)).save(f"{FOLD}/IDENTITY_panel.png"); print("panel ok")
PY
echo "cotrain 2h + deploy eval done $(date)" > "$FOLD/COTRAIN_DONE.txt"
echo "[c2h] ALL DONE $(date)"
