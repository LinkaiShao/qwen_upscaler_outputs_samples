#!/bin/bash
# ============================================================================
# run_kv_clean.sh — CLEAN co-train test of "does an UNFROZEN run37 learn to READ an
# injected garment K/V?". Every value is HARD-SET (no ${VAR:-} layering) so what is
# written here is exactly what runs. Rebuilt from scratch after run_branch_credit.sh
# became an unreliable tangle of layered overrides.
#
# The experiment:
#   - pure garment K/V append (USE_GARMENT_KV) — the reference-net garment branch
#   - global LoRA CO-TRAIN (FREEZE_LORA=0, all 60 blocks) — the untested lever
#   - garment-slot STARVATION (zero the main garment+rough slot) so the K/V is the source
#   - WRONG-garment CONTRASTIVE (USE_KV_WRONG_CONTRAST): correct garment must reconstruct
#     the garment region BETTER than a rolled WRONG garment → forces reading CONTENT, not
#     just "use the K/V". Watch the step log: bc_on(correct) vs kvc_wrong(wrong) vs bc_off(no-KV).
#   - image losses ON (NOT pure-latent) so bg/skin can't blow up during the co-train.
#   - MULTI-GARMENT (full 11647) so it CANNOT memorize per-image.
#   - BATCH_SIZE=2 REQUIRED (the wrong-garment roll needs >1 sample).
# ============================================================================
set -e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RUN_NAME="${RUN_NAME:-KV_CLEAN}"
export SEED=42

# ── warm-start from run37 (frozen base weights + LoRA + v6, then LoRA is unfrozen below) ──
_R37="runs/BGBAND37_012549/final"
export LORA_INIT_PATH="$_R37/tryon_lora.safetensors" V6_HEADS_INIT_PATH="$_R37/v6_heads.pt"

# ── run37 architecture essentials (proven; do not change) ──
export USE_V6=1 V6_R_IN=2 V6_R_OUT=7 DILATE_M_FULL=3
export USE_PURE_NOISE=1 SLOT_ORDER="agnostic,garment,silhouette"
export LORA_RANK=64 LORA_R=64 LORA_ALPHA=128
export USE_AGNOSTIC_RANDOM_FILL=1 AGNOSTIC_CORRUPT_MODE=binary AGNOSTIC_FILL_MODE=1 USE_AGNOSTIC_INPAINT=0
export POSE_USE_WARPED_RGB=1

# ── data: MULTI-GARMENT (full dataset, 5 reserved as val). NO overfit. ──
export USE_FULL_TRAIN=1 TRAIN_LIMIT=0 VAL_USE_RESERVED=1 VAL_LIMIT=5
# (do NOT set OVERFIT_SIDS)
export BATCH_SIZE=2 GRAD_ACCUM=1          # >=2 REQUIRED for the wrong-garment roll

# ── CO-TRAIN: unfreeze the FULL LoRA (all blocks, single LR — no per-block split) ──
export FREEZE_LORA=0 FREEZE_V6=1 USE_PCGRAD=0
export EARLY_BLOCK_CUTOFF=-1 LR_EARLY_MULT=1.0    # -1 disables the per-block LR split -> global LoRA @ LR
export LR=3e-5 LR_SCHEDULE=cosine LR_WARMUP=50 LR_COSINE_TOTAL=40000 LR_MIN_FRAC=0.1

# ── garment K/V append branch (reference-net) ──
export USE_GARMENT_KV=1
export GARMENT_KV_BLOCKS="44,50,55,59" GARMENT_KV_ENC_BLOCKS="0,1" GARMENT_KV_FP32=1
export GARMENT_KV_GATE_INIT_LOGIT=0.0             # sigmoid(0)=0.5 meaningful contribution from step 0
export GARMENT_KV_LR=3e-5 GARMENT_KV_LR_KV=1e-4 GARMENT_KV_LR_GATE=1e-3 FREEZE_KV_ENCODER=0

# ── STARVATION: zero the main garment + rough slot EVERY step (P=1.0, pure) so the K/V is the ONLY source ──
export GARMENT_SLOT_CORRUPT=zero GARMENT_SLOT_CORRUPT_ROUGH=1 GARMENT_SLOT_CORRUPT_P=1.0
export GARMENT_SCHEDULE=0                          # no adapter schedule (pure K/V, no state_enhancer)
export USE_GARMENT_ADAPTER=0

# ── CONTRASTIVE: two-pass credit. bc_on=gar(correct), bc_off=gar(no-KV), kvc_wrong=gar(wrong garment) ──
export USE_BRANCH_CREDIT=1                         # zero contrastive: correct must beat NO-KV (bypass)
export W_BRANCH_CREDIT=3.0 BC_MARGIN=0.03
export USE_KV_WRONG_CONTRAST=1                     # THE reading test: correct must beat WRONG (rolled) garment
export W_KV_WRONG_CONTRAST=5.0
export CONTRAST_SIGMA_MIN=0.6                      # only credit the contrastive at σ>=0.6 (un-leaked regime; fixes low-σ C_t leak w/o high-σ-only training)
export W_KEEP_BGSKIN=0.0                           # keep-loss is adapter-delta specific; off for K/V (image losses guard bg/skin)

# ── LOSSES: flow + region + IMAGE (NOT pure-latent) so bg/skin is protected during co-train ──
export W_FLOW=1.0 LAMBDA_RECON=1.0
export PURE_LATENT=0                               # image decode ON (protects bg/skin)
export IMG_LOSS_WEIGHT=0.5 USE_V6_IMG_STAGE=1
export W_IMG_V6_G=4.0 W_IMG_V6_G_ID=4.0 W_IMG_V6_G_STRUCT=0.3
export W_IMG_V6_S=3.0 W_IMG_V6_B=8.0 W_IMG_V6_OTHER=0.7 W_IMG_V6_K=0.3
export W_IMG_V6_UB=1.2 W_IMG_V6_UB_LO=0.3 W_IMG_V6_UB_HI=1.2
export LAMBDA_V6_REPAIR=0.8 LAMBDA_V6_ROUTE=0.5 LAMBDA_ALLOC=0.0
export USE_FLOW_BG_SPLIT=1 PCT_FLOW_CORE=0.50 PCT_FLOW_BODY=0.20 PCT_FLOW_BG=0.30 PCT_FLOW_UB=0.04 PCT_FLOW_KEEP=0.01
export USE_REGION_PCT_LOSS=1 RPCT_GARMENT=0.50 RPCT_SKIN=0.20 RPCT_BG=0.30 RPCT_BOUNDARY=0.08 RPCT_KEEP=0.02 RPCT_IMG_W=0.5 RPCT_DEBUG=1
# all other exotic losses OFF (explicit)
export USE_SOAR=0 SOAR_PROB=0 USE_GAN=0 LAMBDA_ADV=0 USE_GARMENT_BRIDGE=0 BRIDGE_ONLY=0 BRIDGE_SENS_EVERY=0
export USE_GARMENT_NET=0 USE_MULTI_BLOCK_INJ=0 USE_LATE_XATTN=0 USE_GARMENT_XATTN=0 USE_GARMENT_OOTD=0 USE_POSE_RESIDUAL=0 USE_MASK_TRAJ_BLEND=0 USE_QWEN_REFINER=0
export USE_SIGMA_SCHED=0 USE_SIGMA_LOSS_SCHED=0 USE_IMG_CROSSOVER=0
export LAMBDA_HF_GARMENT=0 LAMBDA_PERCEPTUAL=0 LAMBDA_TV=0 LAMBDA_ANTISLUDGE=0 LAMBDA_CHROMA=0 W_GARMENT_EDGE_MATCH=0 LAMBDA_LATE_SHELL=0 LAMBDA_BG_SHELL_AB=0

# ── cadence ──
export NUM_EPOCHS=100000 MAX_STEPS=500000 TIME_BUDGET="${TIME_BUDGET:-7200}"
export SAVE_PER_EPOCH=0 VAL_PER_EPOCH=0 SAVE_EVERY_SECONDS=2700 AUTO_STOP_EPS=-1e9
export LOGGING_STEPS=10 NAN_ABORT=1 SAVE_ON_NAN=0
export DEPLOY_HALO_EVAL=0

mkdir -p logs runs
RESDIR="${RESDIR:-bgsmearresearch/KV_CLEAN}"; mkdir -p "$RESDIR"; LOG="${ADAPTER_LOG:-$RESDIR/${RUN_NAME}.log}"
echo "RUN_NAME=${RUN_NAME}  LOG=${LOG}"
echo "READ THE LOG: bc_on=gar(correct)  kvc_wrong=gar(wrong)  bc_off=gar(no-KV).  WANT bc_on < kvc_wrong (reads content) AND bc_on < bc_off (KV helps)."
/home/link/venvs/ootd/bin/python -u train.py > "${LOG}" 2>&1
echo "train.py exited rc=$? RUN_NAME=${RUN_NAME}"
