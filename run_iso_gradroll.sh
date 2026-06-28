#!/bin/bash
# ============================================================================
# FINE-TUNE of soar_noise/final to fix the revealed-background COLOR defect:
#   - bg paints ~2.5L darker + warm (beige) cast vs real neutral studio white
#   - bg is blotchy (L-std 3-13 vs real 0.2), with a strong edge-ring spike
# Warm-start the good soar_noise model, ADD the bg color/flatness losses, low LR.
# ============================================================================
set -e
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RUN_NAME="vton_$(date +%Y%m%d_%H%M%S)"

# ── WARM-START: resume from the 1.5h fine-tune checkpoint (edge ring already -33..-74%
#    deploy-verified). Update this to the newest latest/ to chain further. ──
_BASE="runs/vton_20260623_110824/latest"
export LORA_INIT_PATH="${_BASE}/tryon_lora.safetensors"
export V6_HEADS_INIT_PATH="${_BASE}/v6_heads.pt"
export MULTI_GAR_INIT_PATH="${_BASE}/multi_block_injection.pt"
unset GARMENT_NET_INIT_PATH GARMENT_XATTN_INIT_PATH GARMENT_ENCODER_INIT_PATH

# ── architecture (identical to soar_noise) ──
export USE_FULL_TRAIN=1 TRAIN_LIMIT=0
export VAL_USE_RESERVED=1 VAL_LIMIT=5
export SEED=42
export DILATE_M_FULL=3 USE_V6=1 V6_R_IN=2 V6_R_OUT=7
export USE_PURE_NOISE=1
export SLOT_ORDER="agnostic,garment,silhouette"
export EARLY_BLOCK_CUTOFF=28 LR_EARLY_MULT=5.0
export LORA_RANK=64 LORA_R=64 LORA_ALPHA=128
# fine-tune LR: half of the from-scratch run, short cosine
export LR=1e-5 LR_SCHEDULE=cosine LR_WARMUP=300 LR_COSINE_TOTAL=24000 LR_MIN_FRAC=0.1
export USE_GARMENT_NET=0 USE_GARMENT_XATTN=0
export USE_MULTI_BLOCK_INJ=1 USE_MULTI_BLOCK_FULL=1
export MULTI_BLOCK_TARGETS="12,20,28,36,44,50,55,59" MULTI_BLOCK_LR=2e-5
export USE_SIGMA_SCHED=1 SIGMA_SCHED_LO=0.6 SIGMA_SCHED_HI=1.4

# ── v01 clean loss stack (identical) ──
export W_FLOW=1.0 LAMBDA_RECON=1.0 IMG_LOSS_WEIGHT=0.5
export USE_V6_IMG_STAGE=1
export W_IMG_V6_G=4.0 W_IMG_V6_G_ID=4.0 W_IMG_V6_G_STRUCT=0.3
export W_IMG_V6_UB=1.2 W_IMG_V6_UB_LO=0.3 W_IMG_V6_UB_HI=1.2
export W_IMG_V6_S=3.0 W_IMG_V6_B=5.0 W_IMG_V6_OTHER=0.7 W_IMG_V6_K=0.3
export LAMBDA_V6_REPAIR=0.8 LAMBDA_V6_ROUTE=0.5 LAMBDA_ALLOC=0.0 W_V6_UB=0.0
export USE_SIGMA_LOSS_SCHED=0 USE_IMG_CROSSOVER=0
export LAMBDA_TV_EDGE=0.4 DETAIL_LOSS_SIGMA_MAX=1.0
export LAMBDA_REPAIR_SKIN_IMG=0.0 LAMBDA_REPAIR_BG_IMG=0.0 LAMBDA_GARMENT_CROP_L1=0.0 LAMBDA_BOUNDARY_L1=0.0
export LAMBDA_HF_GARMENT=0.0 LAMBDA_PERCEPTUAL=0.0 LAMBDA_LATE_SHELL=0.0
export LAMBDA_TV=0.0 LAMBDA_INSIDE_AB=0.0 LAMBDA_INSIDE_HF=0.0 LAMBDA_ANTISLUDGE=0.0 LAMBDA_ANTI_GREY=0.0
export LAMBDA_CHROMA=0.0 LAMBDA_CHROMA_RATIO=0.0 LAMBDA_AB=0.0 LAMBDA_AB_DIRECTION=0.0
export LAMBDA_BG_SHELL_KEEP=0.0 LAMBDA_PERSON_HALO_KEEP=0.0
export LAMBDA_NO_BG_LEAK=0.0 USE_GAN=0 LAMBDA_ADV=0.0

# ── keep the soar_noise recipe (noise-fill agnostic + SOAR) ──
export USE_AGNOSTIC_RANDOM_FILL=1 AGNOSTIC_CORRUPT_MODE=binary AGNOSTIC_FILL_MODE=1
export USE_AGNOSTIC_INPAINT=0
# AGGRESSIVE deployed-style SOAR: start high-sigma w/ inference init (real outside hole, noise
# inside), drift K steps through the halo-forming band so x0_pred ACTUALLY exhibits the edge ring,
# then the gem/bg losses steer it to clean. (k=1/dsigma=0.05 was too small to reproduce the halo.)
export USE_SOAR=1 SOAR_PROB=1.0 SOAR_FORCE_START=1 SOAR_START_SIGMA=0.95
export SOAR_KSTEPS=4 SOAR_DSIGMA=0.16 SOAR_RENOISE=0.05 SOAR_SIGMA_MIN=0.05
# PROTECT GARMENT: anchor garment silhouette to GT through the rollout so only bg drifts -> the
# from-noise smoothing can't smear garment detail (the 00008-stripe regression). DIL=0 keeps the
# 0-15px edge ring (outside the garment) drifting/trainable.
export SOAR_PROTECT_GARMENT=1 SOAR_PROTECT_PERSON=1 SOAR_PROTECT_DIL=0

# ══════════════ THE BG-COLOR FIX (this run's additions) ══════════════
# (1) PUNISH THE COLOR DIFFERENCE: anchor pred bg mean -> real visible-bg mean (kills the warm/dark cast).
export LAMBDA_BG_CHROMA=0.0 BG_TARGET_BAND=80
# (2) FLATNESS / kill blotchiness: match each bg pixel to a smooth inpainted bg-field target.
export LAMBDA_BG_FIELD=0.0 BG_FIELD_KSIZE=61 BG_FIELD_SIGMA=24
# (3) Lab a,b color match in the bg shell around the body (reinforce neutral color).
export LAMBDA_BG_SHELL_AB=0.0 BG_SHELL_DIL_PX=30 BG_SHELL_SIGMA_GATE=0.6
# (4) EDGE-RING halo (THE priority): MEASURED as the BACKGROUND ring within 15px of the GARMENT
#     silhouette (|d|~32 at 0-15px), 0% overlap with the agnostic boundary. Force pred==GT(real bg)
#     there. STRONG weight for fast removal. (Old W_BOUNDARY_MATCH targeted the wrong ring AND is a
#     no-op in the v6 region-split path — disabled.)
export W_GARMENT_EDGE_MATCH=8.0 GARMENT_EDGE_RING_PX=8   # gem owns 0-8px bg ring
export BG_RING_ISOLATION=1
# NO-STOP-GRAD differentiable rollout: backprop through 2 steps (0.95->0.10) of the deployed trajectory
export GRAD_ROLLOUT_STEPS=2 GRAD_ROLLOUT_SIGMA=0.95 GRAD_ROLLOUT_SIGMA_MIN=0.10
# ── CLEAN region separation: the ONLY image-recon is the unified v6 region-split
#    L_img (img_g/img_s match GT = hard task; img_b matches the REAL SURROUNDING
#    WHITE = easy copy task). NO bg-loss soup (chroma/field/shell/gem/brs all OFF). ──
export BG_REGION_MATCH_WHITE=1 W_BG_ROUTE_SELF=0.0
# full-hole region partition (g+s+b+other == M_full, 100%) + global harmony L1
export V6_PARTITION_FULL_HOLE=1 W_HARMONY=2.0
# in-loop deployed-halo eval: full from-noise rollout + ring-contrast logged ~hourly ([HALO])
export DEPLOY_HALO_EVAL=1 DEPLOY_EVAL_HOURS=0.75 DEPLOY_EVAL_STEPS=20
export W_BOUNDARY_MATCH=0.0 LAMBDA_TV_AGN_RING=0.0
# ═════════════════════════════════════════════════════════════════════

# ── cadence: save every HALF epoch; fine-tune is shorter ──
# slow (~10.6s/it from the K=4 rollout) -> save+val by WALL-CLOCK (every 1.5h), not steps, so
# checkpoints arrive on a predictable schedule to DEPLOY-VERIFY the edge ring drop.
export NUM_EPOCHS=2 TIME_BUDGET=14400 MAX_STEPS=500000
export SAVE_PER_EPOCH=0 VAL_PER_EPOCH=0 SAVE_EVERY_STEPS=0 VAL_EVERY_STEPS=0 SAVE_EVERY_VAL=1
export SAVE_EVERY_SECONDS=2700   # 45min: one validation+snapshot+seeded HALO per 45min
export AUTO_STOP_EPS=-1e9
export LOGGING_STEPS=20

mkdir -p logs runs
LOG="logs/${RUN_NAME}.log"
echo "RUN_NAME=${RUN_NAME}  LOG=${LOG}"
echo "FINE-TUNE: warm-start soar_noise/final + bg color(chroma/field/shell_ab)+edge-ring losses, LR=1e-5"
nohup /home/link/venvs/ootd/bin/python -u train.py > "${LOG}" 2>&1 &
echo "PID=$!"
echo "${LOG}" > /tmp/bgcolor_log
