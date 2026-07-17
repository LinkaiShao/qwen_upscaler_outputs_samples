# BG-Smear Research Log

Tracking the fix for the background **smear / darkening** the model generates around the person silhouette.

## The metric (how every run is scored)

`CLOUD` = coherent (low-frequency) deviation of the **generated background** from the **real GT background**, measured in the **8–50 px band just outside the person silhouette** (parse-v3), on the model's **raw `predict_sample` output** (NOT the grey-hole-pasted deploy — the paste hides the defect).

Validated (`/tmp/bg_smear_metric.py`, `/tmp/smear_validate.py`):
- reads **0.00** on a clean GT-vs-GT; VAE round-trip floor ~**0.2**
- scales linearly with injected smear amplitude
- **blind** to garment differences and to incoherent noise
- ranks real generations the way the eye does

`darkening` = signed mean(luma(pred) − luma(GT)) in the band (negative = pred darker than real; this is the dominant defect).

## Baselines (raw `predict_sample` CLOUD, 5 IDs)

| model | mean CLOUD | note |
|---|---|---|
| **if5** (`BG_FIXED_I_f5_reference`) | **~3.4** | target — the model that already fixes the bg |
| snap (snap_2101) | ~6.9 | moderate |
| **soar** (`soar_noise_20260619_170822`) | **~8.2** | the baseline we train from |

Deployed (grey-hole pasted) CLOUD collapses to ~1.3–4.3 for all — do NOT score on that; it hides the problem.

## Runs

| # | name | what | mean CLOUD | Δ vs soar | verdict |
|---|---|---|---|---|---|
| 00 | soar_baseline | reference (no bg-band loss) | 8.19 | — | baseline |
| 01 | bgband | bg-band loss (cloud+dark+chroma) on single-step x0, from soar | 10.38 | +2.19 | ❌ REGRESSED (rollout effect; single-step loss backfired) |
| 02 | bifurcated_bgshare | bg share 0.25→0.35 (flow+region-%), from soar | 5.16 | -3.03 | ✅ IMPROVED bg; ⚠ garment regressed |
| 03 | bgshare030 | bg share 0.30, from soar | 4.84 | -3.36 | ✅ BEST valid |
| 04 | garshare060 | garment share 0.50→0.60 (bg 0.30) | 5.68 | -2.52 | ❌ REJECTED (bg worse, garment no better) |
| 05 | confirm030 | confirm run03 (bg 0.30 repeat) | 5.27 | -2.93 | ⚠ VARIANCE: run03 4.84 not reproduced; bg-0.30 true ~5.1 = same as 0.35 |
| 06 | frozenbase_v6heads | FREEZE LoRA, train only v6 heads | 8.19 | 0.00 | ❌ NO CHANGE — v6_heads had no deploy path |
| 07 | v6_deploy_path_smoke | wire+gate v6 heads into predict_sample | 58.8 | +50.6 | ✅ plumbing+gate proven non-destructive; ring garbage (undertrained heads) |
| 08 | v6_overfit5 | RETRACTED — was NOT train==deploy (scored raw x0_pred); rep-flat proves aux objective bad, not V6 impossible |  |  | ⚠ superseded by run09 |
| 09 | v6_traindeploy | v6 heads trained THROUGH gated deploy composite (train==deploy), 5-ID | 24.87 | +16.7 | ✅ LEARNS (val_img 2.68→1.97, CLOUD 59→25) but undertrained (305 steps); ring still confetti → train longer |
| 10 | bg_postcorrect | pixel post-correction | INVALID (GT leak) | — | ❌ near-0 was GT paste; deploy-legal agnostic interp = 7.35 (marginal). Only THIS method ruled out, not post-processing globally |

Each `runNN_*/` folder has its own `README.md` (setup + numeric results) and `images/` (pred-vs-gt panels with CLOUD burned in).
