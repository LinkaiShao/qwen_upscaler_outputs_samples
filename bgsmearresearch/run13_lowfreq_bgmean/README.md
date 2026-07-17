# run13 — run03 + low-frequency bg correction only (LAMBDA_BG_CHROMA bg-mean anchor)

**Run:** `runs/BGBAND13_012531`  **Warm-start:** run03  **Budget:** 2 h (1000 steps)  **Launcher:** `run_bgband13.sh`
**Single lever:** `LAMBDA_BG_CHROMA=2.0` (anchor pred bg MEAN → visible-bg mean; lowest-freq, no HF copying). BG_FIELD off.

## Hypothesis (user experiment #2)
Supervise only low-freq luminance/chroma in bg → remove dark haze without forcing HF copying or hurting garment.
Contrast to run12's per-pixel field: this is a single mean per sample (pure low-freq). Isolates frequency.

## Result — verdict: ✗ NOT accepted over run03 (deploy gate); same pattern as run12.
| metric | run03 | run13 | Δ |
|---|---|---|---|
| CLOUD (ring) | 3.28 | 4.24 | +0.96 ✗ |
| edit_bg_dL (deploy hole bg) | −1.31 | −2.27 | −0.96 ✗ darker |
| **edit_all_dL (PRIMARY gate)** | **−3.78** | **−5.90** | −2.12 ✗ darker patch |
| whole_bg_dL (raw bg) | −19.71 | −10.03 | +9.7 ✓ |
| far_bg_dL (raw corners) | −36.65 | −14.87 | +21.8 ✓ |
| edit PSNR / SSIM | 19.52 / 0.829 | 19.49 / 0.833 | flat |
| garL1 / skinL1 | 22.01 / 18.23 | 23.54 / 18.07 | garL1 ✗ slight |

**Reading:** `bgc` lever learned (0.018→0.008). Result mirrors run12 (bg-field): the raw whole/far bg darkening is fixed (far −37→−15) but the deploy hole patch darkens (edit_all −3.78→−5.90) and CLOUD worsens. **Robust cross-lever finding: raw-bg brightness ↑ trades against deploy-hole quality ↓** under bg-luminance losses. run12 (field) slightly better on raw far bg (−11.7 vs −14.9), slightly worse garL1. Neither beats run03 for the deployed product.
**Panels:** `/tmp/eval_run13/panel_all.png`. **Implication:** the deploy-hole darkening isn't a bg-luminance-supervision problem — it's driven elsewhere (garment/skin interaction + SOAR rollout). Next levers (run14 weighting, run15 conditioning) probe different mechanisms.
