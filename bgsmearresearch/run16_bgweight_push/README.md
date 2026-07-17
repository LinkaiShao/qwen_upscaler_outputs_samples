# run16 — extend run14 weighting lever (W_IMG_V6_B 8→11)

**Run:** `runs/BGBAND16_074125`  **Warm-start:** run03  **Budget:** 2 h (1007 steps)  **Launcher:** `run_bgband16.sh`
**Single change vs run03:** bg-region img weight `W_IMG_V6_B` 5→11 (run14 used 8). Noise-fill ⇒ standard eval.

## Hypothesis
run14 (B=8) was the deploy-best. Trace the weighting curve further — does more bg-weight help more, or overshoot?

## Result — ✗ OVERSHOOT. Confirms the weighting optimum is at ~8 (run14).
| edit_all_dL (deploy gate) | run03 (B=5) | run14 (B=8) | run16 (B=11) |
|---|---|---|---|
| | −3.78 | **−2.37 (best)** | −5.23 (worse than run03) |

| metric | run03 | run14 | run16 |
|---|---|---|---|
| CLOUD | 3.28 | 3.34 | 4.06 |
| edit_bg_dL | −1.31 | −1.28 | −2.55 |
| edit_nonbg_dL | −5.02 | −2.91 | −6.61 |
| **edit_all_dL** | −3.78 | **−2.37** | −5.23 |
| whole_bg_dL | −19.71 | −10.51 | −11.34 |
| far_bg_dL | −36.65 | −17.12 | −17.72 |
| edit PSNR / SSIM | 19.52/0.829 | 19.88/0.831 | 20.19/0.832 |
| garL1 / skinL1 | 22.01/18.23 | 19.12/19.63 | 20.21/18.25 |

**Reading:** The bg-region weighting curve is non-monotonic with an optimum near B=8: 5→−3.78, **8→−2.37**, 11→−5.23.
Pushing past 8 re-darkens the deploy patch (edit_nonbg −2.91→−6.61). Brackets the optimum cleanly. **run14 (B=8)
remains deploy-best.** (Interesting: garL1 improves at both 8 and 11 vs run03, but the darkening metric overshoots.)
**Panels:** `/tmp/eval_run16/panel_all.png`. **Next:** run17 = run14 (B=8) + garment/skin weight bump — combine the
proven win with a direct lever on the garment/skin darkening driver.
