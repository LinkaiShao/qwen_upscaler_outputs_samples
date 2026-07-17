# mask_traj_blend_OVERFIT5 — Masked Trajectory Blending (5-ID overfit)

**What it does.** A separate garment velocity denoiser (`model.py` = GarmentVelocityDenoiser: copied Qwen
blocks over [C_t 16 + garment_latent 16 + warped_mask 1 + warped_rgb 3] + σ embedding → 16-ch velocity, vel_head
zero-init) predicts garment-region velocity; frozen run37 owns everything else. At EVERY denoise step
`v_final = v_run37·(1−M) + v_garment·M` (M = warped garment mask). Only the denoiser trains, and only through M,
so bg/skin are locked by construction. Deploy uses a **two-track state quarantine** (`x_base` pure run37 +
`x_edit` garment track, hard-composited `x_edit = x_base·(1−M) + x_edit·M` after every Euler step) so garment
changes can never accumulate outside the mask.

**How the pieces connect.** `run.sh` (=run_masktraj.sh) sets the recipe (USE_MASK_TRAJ_BLEND=1, frozen run37,
PURE_LATENT single-step flow MSE, POSE_USE_WARPED_RGB=1) and launches `train.py`. The live wiring lives in
trainlib (guarded, importing root `mask_traj_blend.py`); the exact injected code is COPIED VERBATIM into
`WIRING.py` here (state singletons, data loader, the `_fwd` blend hook, the run.py build/save, the two-track
halo_eval rollout). `run_masktraj_1h.sh` = train 1h + deploy eval; `run_masktraj_eval2.sh` = two-track re-eval.

**Verdict (two-track deploy, `numbers_twotrack` / per-ID metric_rawpred).** The garment net WORKS:
garL1(correct)=**12.40** << baseline=21.54 (garment IMPROVED), << wrong=90.50 (identity-specific),
<< zero/empty-garment=68.45 (garment-DRIVEN, not a generic denoiser effect). Quarantine: farBG (far from the
garment) ≈ identical across all 4 conditions (12.44–12.64); bg/skin nearer the garment shift slightly with the
garment change purely from VAE-decode non-locality (latent quarantine is exact by construction).

**Contents:** model.py · run.sh (+ run_masktraj_1h.sh, run_masktraj_eval2.sh) · WIRING.py · MASKTRAJ_*.log ·
numbers.txt / numbers_twotrack.txt · IDENTITY_panel*.png · tt_{baseline,correct,wrong,zero}/ rawpreds.
