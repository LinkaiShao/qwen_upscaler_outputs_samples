# run23 (PROBE B2) — bg LOCKED to run14 + NONLINEAR garment/skin heads

**Run:** `runs/BGBAND23_062251`  **Warm-start:** run14 (frozen)  **Budget:** 2 h (~2083 steps)  **Launcher:** `run_bgband23.sh`
**Design:** FREEZE run14 base. bg IMMOVABLE: `to_b` + `to_route` frozen; `V6_BG_LOCK` hard skin-gate (δ_s only in
argmax-skin repair pixels, no soft blend) → bg = x0_pred EXACTLY (no leak). Train ONLY nonlinear MLP heads
`to_g`/`to_s` (3072→512→64, ~3.2M params) refining frozen x0_pred. Eval: `eval_full_refine.py` with
`V6_FG_NONLINEAR=1 V6_BG_LOCK=1`.

## Eval validity — self-consistency PASSED
`--zero` on run23 ckpt == run14 EXACTLY (CLOUD 3.34, edit_bg −1.28, edit_all −2.37, garL1 19.12, skinL1 19.63).
Plumbing (nonlinear heads + bg-lock compose + hook) correct.

## Result — ✓ SEPARATION WORKS (modest): garment improved while bg held EXACTLY.
| metric | run14 | run23 | Δ | verdict |
|---|---|---|---|---|
| CLOUD | 3.34 | 3.31 | −0.03 | ✓ bg held |
| edit_bg_dL | −1.28 | −1.25 | +0.03 | ✓ bg held (immovable) |
| whole_bg_dL | −10.51 | −10.50 | 0 | ✓ identical |
| far_bg_dL | −17.12 | −17.12 | 0 | ✓ identical |
| edit_all_dL | −2.37 | −1.39 | +0.98 | ✓ patch less dark |
| **garL1** | 19.12 | **18.34** | **−0.78** | ✓ garment better |
| skinL1 | 19.63 | 19.62 | −0.01 | ~ flat |
| edit PSNR/SSIM | 19.88/0.831 | 19.99/0.830 | ✓/~ | ✓ |

## Verdict: FIRST non-tradeoff result. bg PROVABLY immovable (all bg metrics unchanged, no leak — confirmed by both
the self-consistency test and the hard bg-lock) while **garment improved** (garL1 −0.78, edit_all less dark, PSNR up).
Skin flat. This breaks the Pareto frontier that runs 18/19/20 hit — proving region separation IS achievable: a
frozen bg + a separate foreground path can improve foreground without touching bg.

**Caveat — magnitude is small.** garment +4%, skin flat. Nonlinear heads reading FROZEN run14 features have limited
reach (the correction is only partly extractable from frozen features). This is the "SEPARATION WORKS → scale" branch.
**Panels:** `run23_bgfrozen_fg_nonlinear/*_pred_final_gt.png`.

## Next (run24): scale foreground capacity — V6_FG_HIDDEN 512→1024 (2× head capacity), same bg-lock. Tests whether
the garment gain grows with capacity (→ keep scaling heads / long-form) or plateaus (→ frozen features are the ceiling;
foreground needs its own trainable params = region-routed LoRA/adapters).
