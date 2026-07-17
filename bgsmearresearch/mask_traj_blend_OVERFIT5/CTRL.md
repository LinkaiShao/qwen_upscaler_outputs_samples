# MASKED TRAJECTORY BLENDING — control file (5-ID overfit, 1h)

STATUS: **RUNNING**  (launched 2026-07-14 18:37)

## Mechanism
Every denoise step (training AND deploy rollout), inside forward.py `_fwd`:
`v_final = v_run37 * (1 - M) + v_garment * M`  (packed token space; run37 detached = frozen constant)
- `v_run37` = frozen run37 velocity (LoRA + v6 + base ALL frozen).
- `v_garment` = a separate garment velocity denoiser (Qwen blocks, trained).
- `M` = warped garment mask at token res (soft edge). bg/skin: v_final == v_run37 -> LOCKED, gradient can't reach.
- Trains ONLY the garment denoiser, ONLY through the M region. Single-step flow MSE on v_final (pure-latent, no decode).

## Process (ALL under this folder — nothing in /tmp)
- wrapper PID: **1712410**  (`run_masktraj_1h.sh`)
- RUN_NAME: **MASKTRAJ_0714_1837**
- train log: `bgsmearresearch/mask_traj_blend_OVERFIT5/MASKTRAJ_0714_1837.log`
- wrapper out: `bgsmearresearch/mask_traj_blend_OVERFIT5/wrapper.out`
- ckpt: `runs/MASKTRAJ_0714_1837/final/garment_vel_denoiser.pt`
- eval + panels: this folder (`numbers.txt`, `IDENTITY_panel.png`, `imgs_{baseline,correct,wrong,zero}/`)
- DONE marker: `bgsmearresearch/mask_traj_blend_OVERFIT5/DONE.txt`

## Smoke verified (before launch)
- Denoiser installed: GarmentVelocityDenoiser 4 blocks (fp32), 1.39B params, vel_head ZERO-INIT.
- **bg LOCKED (the whole point)**: `[mask_traj_dbg] bg-token max|v_final-v_run37| = 0.000e+00` (exactly 0). fg region differs (mean|Δ|=0.96).
- **Denoiser grad nonzero**: `[masktraj_dbg] denoiser_gradnorm=1.04, vel_head.grad=309`; vel_head.w.abssum climbs 0 -> 0.266 (off zero-init).
- v_garment=0 at step0 (v_final == run37). 0 errors, no NaN.

## Eval (auto after 1h) -> numbers.txt + IDENTITY_panel.png
Deploy render on 5 reserved, 4 conditions, blend applied every Euler step:
1. **run37 BASELINE** (blend off) — bar to beat.
2. **blend CORRECT** garment.
3. **blend WRONG** garment (MASK_TRAJ_DEPLOY_WRONG).
4. **blend ZERO** (denoiser off, MASK_TRAJ_ZERO) — MUST == baseline (blend-identity check).
5-panel: `[run37 | correct | WRONG | ZERO | GT]`.
- WIN = garL1(correct) < garL1(baseline) [garment improved] AND < garL1(wrong) [garment-specific].
- By construction bg/skin (CLOUD/farbg/skinL1) MUST be ~identical across all rows (if not, the blend is buggy).
- Diagnostic reading: if garment does NOT improve while bg/skin are LOCKED, the branch isn't learning detail (not contamination).

## Poll
- `tail bgsmearresearch/mask_traj_blend_OVERFIT5/numbers.txt`
- `tail bgsmearresearch/mask_traj_blend_OVERFIT5/wrapper.out`
- `cat bgsmearresearch/mask_traj_blend_OVERFIT5/DONE.txt`
