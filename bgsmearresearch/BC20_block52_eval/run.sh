#!/bin/bash
# ============================================================================
# run37 (PCGrad + MASK-GATED garment = best structural combo): run31 (SOAR + mask-gated GarmentChain, garment writes
# garment tokens only) PLUS PCGrad on shared LoRA+v6 across garment/skin/bg region losses. NO adapters. TEST: does
# preventing direct garment→bg write (mask-gate) AND shared-gradient conflict (PCGrad) together fix all 3 regions?
# noise-fill agnostic (kill color leakage) + SOAR (intrinsic
# boundary darkening). From scratch, v01 recipe otherwise. Save every half epoch.
# Runs the MAIN trainlib (so trainlib/forward.py AGNOSTIC_FILL_MODE edit applies).
# ============================================================================
set -e
_TB_PASSED="${TIME_BUDGET:-}"   # ENH v2: capture caller's TIME_BUDGET BEFORE base clobbers it to 7200 (:- ), so smoke TIME_BUDGET=150 survives and the default becomes 3600
cd "/home/link/Desktop/Code/fashion gen testing/vtonautoresearch"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RUN_NAME="${RUN_NAME:-RUN_ENHANCER}"   # state-aware enhancer: CrossAttn(Q=run37 hidden, K=V=garment) delta at late block, zero-init

# ── warm-start: fresh by default; RESUME=1 continues from the newest saved checkpoint (crash recovery) ──
if [ "${RESUME:-0}" = "1" ]; then
    _CK="runs/${RUN_NAME}/latest"                                    # finer within-epoch save, if present
    [ -f "$_CK/tryon_lora.safetensors" ] || _CK=$(ls -d runs/${RUN_NAME}/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$_CK" ] && [ -f "$_CK/tryon_lora.safetensors" ]; then
        echo "RESUMING from $_CK"
        export LORA_INIT_PATH="$_CK/tryon_lora.safetensors" V6_HEADS_INIT_PATH="$_CK/v6_heads.pt" MULTI_GAR_INIT_PATH="$_CK/multi_block_injection.pt"
        unset GARMENT_NET_INIT_PATH GARMENT_XATTN_INIT_PATH GARMENT_ENCODER_INIT_PATH
    else
        echo "RESUME=1 but no checkpoint found -> starting fresh"
        unset LORA_INIT_PATH V6_HEADS_INIT_PATH GARMENT_NET_INIT_PATH GARMENT_XATTN_INIT_PATH GARMENT_ENCODER_INIT_PATH MULTI_GAR_INIT_PATH
    fi
else
    # run02: FORCE warm-start from the soar baseline (bgsmear experiment), not fresh
    _R37="runs/BGBAND37_012549/final"   # run49: base = run37
    export LORA_INIT_PATH="$_R37/tryon_lora.safetensors" V6_HEADS_INIT_PATH="$_R37/v6_heads.pt"
    unset MULTI_GAR_INIT_PATH
    unset GARMENT_NET_INIT_PATH GARMENT_XATTN_INIT_PATH GARMENT_ENCODER_INIT_PATH
    echo "WARM-START from SOAR (long-form setup probe): $_SOAR"
fi

# ── v01 architecture (identical) ──
export USE_FULL_TRAIN=1 TRAIN_LIMIT=0
export VAL_USE_RESERVED=1 VAL_LIMIT=5
export SEED=42
export DILATE_M_FULL=3 USE_V6=1 V6_R_IN=2 V6_R_OUT=7
export USE_PURE_NOISE=1
export SLOT_ORDER="agnostic,garment,silhouette"
export EARLY_BLOCK_CUTOFF=28 LR_EARLY_MULT=5.0
export LORA_RANK=64 LORA_R=64 LORA_ALPHA=128
export LR=5e-6 LR_SCHEDULE=cosine LR_WARMUP=50 LR_COSINE_TOTAL=40000 LR_MIN_FRAC=0.1
export USE_GARMENT_NET=0
export USE_MULTI_BLOCK_INJ=0 USE_MULTI_BLOCK_FULL=0   # NO copied-Qwen hidden stream
export MULTI_BLOCK_TARGETS="12,20,28,36,44,50,55,59" MULTI_BLOCK_LR=7.5e-5
# ── run49: masked garment K/V CROSS-ATTENTION at proj_out (no downstream attn to leak through) ──
#   A_g = CrossAttn(Q=F_C, K=V=G(garment));  F_C += gamma * p_g_tok * A_g
#   out_proj ZERO-INIT => step 0 == run37 exactly.  fp32 scalar gate, small init.
#   p_g_tok from WARPED_MASK (deploy-available; NOT the GT target mask) => bg/skin get EXACTLY 0.
# Run1: LATE-BLOCK residual xattn (blocks 50,55,59) instead of proj_out.
# Frozen base still has blocks 51..59 of self-attention to INTERPRET the garment signal.
export GARMENT_XATTN_LAYERS=2 GARMENT_XATTN_COPY_BLOCKS="0,1"
# (gamma/gate set above for late xattn — do not override)
export GARMENT_XATTN_LR=1e-4 GARMENT_XATTN_LR_GATE=1e-3
export USE_PCGRAD=0
# ── run50: garment-bridge objective. Preserve bg/skin against the FROZEN BASE (branch OFF),
#    not against GT. Hinge/eps: numerical noise free, visible disturbance punished hard. ──
export USE_GARMENT_BRIDGE=1 BRIDGE_ONLY=1
export W_BRIDGE_GAR=1.0 W_BRIDGE_GAR_EDGE=1.0
export W_BRIDGE_BG=10.0 W_BRIDGE_SKIN=10.0 W_BRIDGE_FAR=25.0   # run58: reduced first so gate can open (raise later)
export W_BRIDGE_BND=10.0 W_BRIDGE_RESOUT=0.0 W_BRIDGE_SENS=8.0   # RESOUT n/a for KV-append; SENS strong
export BRIDGE_EPS_BG=0.01 BRIDGE_EPS_SKIN=0.01 BRIDGE_EPS_FAR=0.005 BRIDGE_EPS_BND=0.02
export BRIDGE_SENS_EVERY=1 BRIDGE_SENS_MARGIN=0.05 BRIDGE_FAR_DILATE=19
export BRIDGE_WRONG_SIDS="00006_00,00008_00,00013_00,00017_00,00034_00"
export GARMENT_XATTN_BOUNDARY_W=0.0
# Phase A: train xattn only (LoRA + v6 frozen). Phases B/C come after this validates.
export FREEZE_LORA=1 FREEZE_V6=1
# 5-image OVERFIT (train == val == the 5 reserved ids)
export OVERFIT_SIDS="00006_00,00008_00,00013_00,00017_00,00034_00"
export USE_SIGMA_SCHED=1 SIGMA_SCHED_LO=0.6 SIGMA_SCHED_HI=1.4

# ── v01 clean loss stack (identical) ──
export W_FLOW=1.0 LAMBDA_RECON=1.0 IMG_LOSS_WEIGHT=0.5
# ── TRUE bg BIFURCATION: bg_repair as its OWN normalized flow objective (was lumped in repair@0.1) ──
# de-starves the bg (fixes the deterministic under-shoot) AND clamps the ring (pulls revealed bg to GT everywhere)
# run02: bg share bumped in BOTH the flow split and the region-% (must move together). skin 0.25->0.20 to make room; garment stays 0.50 dominant.
export USE_FLOW_BG_SPLIT=1 PCT_FLOW_CORE=${PCT_FLOW_CORE:-0.50} PCT_FLOW_BODY=0.20 PCT_FLOW_BG=${PCT_FLOW_BG:-0.30} PCT_FLOW_UB=0.04 PCT_FLOW_KEEP=0.01
export W_GARMENT_EDGE_MATCH=8.0 GARMENT_EDGE_RING_PX=15
# ── UNIFIED region-percentage objective (one coherent bundle per region, no soup) ──
export RPCT_DEBUG=1 USE_REGION_PCT_LOSS=1 RPCT_GARMENT=${RPCT_GARMENT:-0.50} RPCT_SKIN=0.20 RPCT_BG=${RPCT_BG:-0.30} RPCT_BOUNDARY=0.08 RPCT_KEEP=0.02 RPCT_IMG_W=0.5
export USE_V6_IMG_STAGE=1
export W_IMG_V6_G=4.0 W_IMG_V6_G_ID=4.0 W_IMG_V6_G_STRUCT=0.3
export W_IMG_V6_UB=1.2 W_IMG_V6_UB_LO=0.3 W_IMG_V6_UB_HI=1.2
export W_IMG_V6_S=3.0 W_IMG_V6_B=${W_IMG_V6_B:-8.0} W_IMG_V6_OTHER=0.7 W_IMG_V6_K=0.3
export LAMBDA_V6_REPAIR=0.8 LAMBDA_V6_ROUTE=0.5 LAMBDA_ALLOC=0.0 W_V6_UB=0.0
export USE_SIGMA_LOSS_SCHED=0 USE_IMG_CROSSOVER=0
export LAMBDA_TV_EDGE=0.4 DETAIL_LOSS_SIGMA_MAX=1.0
export LAMBDA_REPAIR_SKIN_IMG=0.0 LAMBDA_REPAIR_BG_IMG=0.0 LAMBDA_GARMENT_CROP_L1=0.0 LAMBDA_BOUNDARY_L1=0.0
export LAMBDA_BG_CHROMA=0.0 LAMBDA_BG_FIELD=0.0 LAMBDA_HF_GARMENT=0.0 LAMBDA_PERCEPTUAL=0.0 LAMBDA_LATE_SHELL=0.0
export LAMBDA_TV=0.0 LAMBDA_INSIDE_AB=0.0 LAMBDA_INSIDE_HF=0.0 LAMBDA_ANTISLUDGE=0.0 LAMBDA_ANTI_GREY=0.0
export LAMBDA_CHROMA=0.0 LAMBDA_CHROMA_RATIO=0.0 LAMBDA_AB=0.0 LAMBDA_AB_DIRECTION=0.0
export LAMBDA_BG_SHELL_KEEP=0.0 LAMBDA_BG_SHELL_AB=1.0 BG_SHELL_DIL_PX=30 BG_SHELL_SIGMA_GATE=0.6 LAMBDA_PERSON_HALO_KEEP=0.0
export LAMBDA_TV_AGN_RING=0.0 LAMBDA_NO_BG_LEAK=0.0 USE_GAN=0 LAMBDA_ADV=0.0

# ══════════════ THE TWO CHANGES ══════════════
# 1) Decouple agnostic COLOR: fill the masked hole with high-freq Gaussian noise
#    (mode 1), full-region (binary). Unmasked agnostic untouched (real bg/skin context).
#    INPAINT OFF — else it overwrites the noise with propagated neighbors → leakage returns.
export USE_AGNOSTIC_RANDOM_FILL=1 AGNOSTIC_CORRUPT_MODE=binary AGNOSTIC_FILL_MODE=1
export USE_AGNOSTIC_INPAINT=0
# 2) SOAR — own-trajectory training for the intrinsic boundary darkening.
# ── PERIODIC multi-step deployed-style rollout (~once every 4 steps) — garment refinement w/o every-step sludge ──
# ═════════════════════════════════════════════

# ── cadence: save every HALF epoch (~5800 steps); no auto-stop ──
export NUM_EPOCHS=100000 TIME_BUDGET=${TIME_BUDGET:-7200} MAX_STEPS=500000
export SAVE_PER_EPOCH=0 KEEP_ONLY_LATEST_EPOCH=1 VAL_PER_EPOCH=0 SAVE_EVERY_SECONDS=2700 SAVE_EVERY_VAL=1
export AUTO_STOP_EPS=-1e9
export LOGGING_STEPS=20
# ── hourly in-loop DEPLOYED metrics (reuses the loaded model — no 2nd copy, so no OOM):
#    full from-noise rollout on the reserved ids, logs [DEPLOY] GARMENT/SKIN/BG dim+L1 every hour ──
export DEPLOY_HALO_EVAL=1 DEPLOY_EVAL_HOURS=0.7 DEPLOY_EVAL_STEPS=20


export USE_SOAR=0 SOAR_PROB=0

# ── DISABLE all other garment paths (only OOTD K/V on) ──
export USE_GARMENT_KV=0 USE_POSE_RESIDUAL=0 USE_LATE_XATTN=0 USE_GARMENT_XATTN=0
export USE_MULTI_BLOCK_INJ=0 USE_MULTI_BLOCK_FULL=0 USE_GARMENT_NET=0
export NAN_ABORT=1 SAVE_ON_NAN=0

# ══════════ COTRAIN: run37 LoRA + OOTD garment reference net (FULL DATA, clean recipe) ══════════
#  Reference-net paradigm (IDM-VTON/OOTD): a copied-run37 garment branch reads the CLEAN garment
#  latent and injects garment features via per-block K/V cross-attn at 36,44,50,55; run37's frozen
#  block QUERY attends to it (to_v_g ZERO-INIT -> step0 == run37 exactly). Co-train run37's LoRA
#  (FREEZE_LORA=0) + the gnet TOGETHER on run37's OWN standard reconstruction recipe, so the
#  low-sigma denoising steps force run37 to PULL real garment DETAIL from the reference.
#
#  LEARNED FROM run69 (garL1=33 blowup): DESIGN THE FAILURE OUT —
#    * NO CFG_DROPOUT / garment-slot corruption (run69 used 0.5 -> starved the garment).
#    * NO HF/bridge/edge soup (run69 bumped these -> destabilized).
#    * Loss = run37's OWN v01 recipe verbatim (inherited above) + PCGrad + maskgate (run37 stability).
#    * Gnet garment input CLEAN.
# ══════════ RUN1: Input Adapter, FROZEN run37 ══════════
#  First-layer aligned-garment injection: C_hidden += M * zero_init_proj([warped_rgb, garment, mask])
#  via a block-0 pre-hook. run37 FULLY FROZEN (LoRA + v6); train ONLY the input adapter.
#  Tests if a first-layer aligned garment signal is interpretable by frozen run37 at all.
export USE_GARMENT_OOTD=0
# ── STATE-AWARE ENHANCER: H_i' = H_i + M_garment * CrossAttn(Q=H_i after block i, K=V=garment_net(warped_rgb+garment_latent+warped_mask)) ──
#   ENH_BLOCK selects the inject block: 59 = RunA (final-block, safest, no downstream self-attn to smear); 55 = RunB (a few downstream blocks).
export USE_GARMENT_ADAPTER=1 GARMENT_ADAPTER_MODE=state_enhancer GARMENT_ADAPTER_LR="${GARMENT_ADAPTER_LR:-5e-5}"   # env-overridable (was clobbering the BC block)
export GARMENT_ADAPTER_BLOCKS="${ENH_BLOCK:-59}" GARMENT_ADAPTER_GNET_BLOCKS=2
export POSE_USE_WARPED_RGB=1                       # aligned garment (warped_rgb) into the adapter input
# ── COMMON PROTOCOL: 50/30/10/10 per-step schedule + asymmetric starvation of the STANDARD garment ──
export GARMENT_SCHEDULE=1 GARMENT_SLOT_CORRUPT=zero GARMENT_SLOT_CORRUPT_ROUGH=1
# ── FROZEN run37 (train only the adapter) ──
export FREEZE_LORA="${FREEZE_LORA:-0}" FREEZE_V6=1 USE_PCGRAD=0
export CFG_DROPOUT=0.0
export USE_GARMENT_BRIDGE=0 BRIDGE_ONLY=0 BRIDGE_SENS_EVERY=0 BRIDGE_SHUFFLE=0
# ── HF/edge soup REMOVED -> run37 defaults (no detail emphasis; denoising supplies detail) ──
export LAMBDA_HF_GARMENT=0.0 W_GARMENT_EDGE_MATCH=8.0 GARMENT_EDGE_RING_PX=15
export W_IMG_V6_G=4.0 W_IMG_V6_G_ID=4.0 W_IMG_V6_G_STRUCT=0.3
# ── FULL DATA: drop the 5-image overfit restriction (train on ALL ids) ──
unset OVERFIT_SIDS
export USE_FULL_TRAIN=1 TRAIN_LIMIT=0
# ── batch: GPU1 (VAE image-loss decode) is contended by another process; batch 1 = run37-proven
#    fits; larger OOMs the 31GB VAE device. PCGrad requires grad_accum=1. ──
export BATCH_SIZE="${BATCH_SIZE:-2}" GRAD_ACCUM=1
# ── PURE-LATENT: kill the per-step VAE decode (the ~27s/step bottleneck). Disable EVERY
#    image-space loss that decodes the prediction; keep the LATENT flow/region/route losses
#    that supervise garment detail in latent space. ~5-10x faster -> many more steps. ──
export PURE_LATENT=1
export IMG_LOSS_WEIGHT=0 USE_V6_IMG_STAGE=0 LAMBDA_V6_REPAIR=0 RPCT_IMG_W=0 LAMBDA_PERCEPTUAL=0
export W_IMG_V6_G=0 W_IMG_V6_G_ID=0 W_IMG_V6_G_STRUCT=0 W_IMG_V6_S=0 W_IMG_V6_B=0 W_IMG_V6_OTHER=0 W_IMG_V6_K=0 W_IMG_V6_UB=0
export W_GARMENT_EDGE_MATCH=0 LAMBDA_LATE_SHELL=0 LAMBDA_BG_SHELL_AB=0
# KEEP (latent, supervise garment detail): W_FLOW, LAMBDA_RECON, USE_REGION_PCT_LOSS (latent-only via RPCT_IMG_W=0),
#   USE_FLOW_BG_SPLIT, LAMBDA_V6_ROUTE (v6 route is latent). DEPLOY_HALO_EVAL off (that decodes too).
export DEPLOY_HALO_EVAL=0
# (batch set below; no decode -> GPU1 VAE no longer the bottleneck; GPU0-limited by 20B + PCGrad retain_graph)
# ── deploy identity test env (eval render passes OOTD_DEPLOY_WRONG=1) ──
export OOTD_DEPLOY_WRONG="${OOTD_DEPLOY_WRONG:-0}"
export TIME_BUDGET=${TIME_BUDGET:-7200}
export DEPLOY_EVAL_HOURS=0.6
# ════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ENH v2 OVERRIDES (state_enhancer_v2_block52_reader) — placed AFTER all base exports
# so they WIN. Thesis: can the main net learn a READER for the garment net (block 52
# enhancer + quarantine 53-59) instead of the LoRA solving alone at low sigma?
# ══════════════════════════════════════════════════════════════════════════════
export RUN_NAME="${RN:-${RUN_NAME:-BC_RUN}}"             # honor RN=... alias
# ══════════ BRANCH-CREDIT base (all 12 BC runs). Per-run knobs are env-overridable. ══════════
# adapter: mode + inject block(s). BC_MODE default state_enhancer; ENH_BLOCK default 52.
export USE_GARMENT_ADAPTER=1 GARMENT_ADAPTER_MODE="${BC_MODE:-state_enhancer}" GARMENT_ADAPTER_LR="${GARMENT_ADAPTER_LR:-5e-5}"
export GARMENT_ADAPTER_BLOCKS="${ENH_BLOCK:-52}" GARMENT_ADAPTER_GNET_BLOCKS=2
# reader: FREEZE run37 LoRA blocks [0..READER_CUTOFF] (mult 0), train (READER_CUTOFF+1)..59 only.
# NEVER mask existing LoRA; NEVER train full LoRA. READER_CUTOFF default 51 -> train 52-59.
export FREEZE_LORA="${FREEZE_LORA:-0}" FREEZE_V6=1 USE_PCGRAD=0
export EARLY_BLOCK_CUTOFF="${READER_CUTOFF:-51}" LR_EARLY_MULT=0.0
export LR="${BC_LR:-5e-5}"                                # reader LR (BC_LR wins over base LR=5e-6); only 52-59 train
unset LATE_BLOCK_CUTOFF LR_LATE_MULT
# ── BRANCH-CREDIT LOSS: two-pass ON(grad)/OFF(bypass,no-grad); credit adapter only when ON<OFF in garment ──
export USE_BRANCH_CREDIT=1
export LOGGING_STEPS=10
export W_BRANCH_CREDIT="${W_BRANCH_CREDIT:-1.0}" BC_MARGIN="${BC_MARGIN:-0.02}"
export W_KEEP_BGSKIN="${W_KEEP_BGSKIN:-1.0}" W_DELTA_REG="${W_DELTA_REG:-0.01}"
export BC_STARVE_FRAC="${BC_STARVE_FRAC:-0.60}"          # frac of steps that zero the standard garment slot
# schedule: branch-credit forces branch_on=True always; ENH_SCHED_V2 off (credit replaces base-solo).
export GARMENT_SCHEDULE=1 ENH_SCHED_V2=0 GARMENT_SLOT_CORRUPT=zero GARMENT_SLOT_CORRUPT_ROUGH=1
# high-sigma starvation: BC01 OFF (normal sigma), BC12 ON (STARVE_HIGH_SIGMA=1 STARVE_SIGMA_LO=0.65).
export STARVE_HIGH_SIGMA="${STARVE_HIGH_SIGMA:-0}" STARVE_SIGMA_LO="${STARVE_SIGMA_LO:-0.5}"
# attention quarantine: BC01 OFF; BC03/BC08 ON (USE_ENH_QUARANTINE=1).
export USE_ENH_QUARANTINE="${USE_ENH_QUARANTINE:-0}" ENH_QUARANTINE_BLOCKS="${ENH_QUARANTINE_BLOCKS:-53,54,55,56,57,58,59}"
# 1-HOUR 5-image OVERFIT (base unset OVERFIT_SIDS above; re-export here so it wins).
# NOTE: OVERFIT_SIDS is only honored inside the USE_FULL_TRAIN=1 code path (constants.py),
# and USE_FULL_TRAIN=0 routes prompts through precompute_prompt_embeds() which needs
# *_agnostic_pixel.pt / *_garment_pixel.pt caches that DO NOT EXIST in this repo. The proven
# 5-img overfit (base run50 precedent) uses USE_FULL_TRAIN=1 + OVERFIT_SIDS via the text cache
# (my_vton_cache/text has all 5 sids). So USE_FULL_TRAIN=1 here is the SAFE correct choice —
# it restricts TRAIN_IDS to the 5 sids AND uses available caches. train==val overfit.
export OVERFIT_SIDS="00006_00,00008_00,00013_00,00017_00,00034_00"
export USE_FULL_TRAIN=1 TRAIN_LIMIT=0
export VAL_USE_RESERVED=1 VAL_LIMIT=5
export TIME_BUDGET="${_TB_PASSED:-3600}"                  # caller value if passed, else 1h
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p logs runs
RESDIR="${RESDIR:-bgsmearresearch/state_enhancer_v2_block52_reader}"; mkdir -p "$RESDIR"; LOG="${ADAPTER_LOG:-$RESDIR/${RUN_NAME}.log}"
echo "RUN_NAME=${RUN_NAME}  LOG=${LOG}"
/home/link/venvs/ootd/bin/python -u train.py > "${LOG}" 2>&1
echo "train.py exited rc=$? RUN_NAME=${RUN_NAME}"
