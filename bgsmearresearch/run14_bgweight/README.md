# run14 — run03 + bg/foreground bifurcated WEIGHTING (W_IMG_V6_B 5→8)

**Run:** `runs/BGBAND14_033138`  **Warm-start:** run03  **Budget:** 2 h (~1000 steps)  **Launcher:** `run_bgband14.sh`
**Single change:** bg-region image-loss weight `W_IMG_V6_B` 5.0 → 8.0. NO new loss, garment/skin weights untouched, BG_FIELD/CHROMA off. Noise-fill hole (same as run03) ⇒ standard eval (`/tmp/eval_full.py`) is matched.

## Hypothesis (user experiment #3)
Keep one deployed LoRA path; just re-weight loss by region (stronger bg, normal garment/skin). Tests whether
the darkening is a WEIGHTING problem rather than needing a new loss/architecture.

## Result — ✅ NEW DEPLOY-BEST over run03 (first genuine win in the queue)
| metric | run03 | run14 | Δ |
|---|---|---|---|
| CLOUD (ring) | 3.28 | 3.34 | +0.06 ~flat |
| edit_bg_dL (deploy hole bg) | −1.31 | −1.28 | +0.03 flat |
| edit_nonbg_dL (gar/skin in hole) | −5.02 | −2.91 | **+2.11 ✓** |
| **edit_all_dL (PRIMARY gate)** | **−3.78** | **−2.37** | **+1.41 ✓ less dark** |
| whole_bg_dL (raw bg) | −19.71 | −10.51 | +9.2 ✓ |
| far_bg_dL (raw corners) | −36.65 | −17.12 | +19.5 ✓ |
| edit PSNR / SSIM | 19.52 / 0.829 | 19.88 / 0.831 | ✓ / flat |
| garL1 | 22.01 | 19.12 | **−2.89 ✓ better** |
| skinL1 | 18.23 | 19.63 | +1.40 ✗ slight |

**Reading:** Up-weighting the bg-region reconstruction (pure weighting, most rule-compliant lever) improved the
deploy edit patch (edit_all −3.78→−2.37), the garment/skin-in-hole darkening (edit_nonbg −5.0→−2.9), garment L1
(22.0→19.1) and raw bg — while keeping CLOUD/edit_bg flat. Opposite of the bg-luminance LOSSES (run12/13), which
regressed the patch. Confirms experiment-3 hypothesis: the darkening responds to region WEIGHTING, not a new loss.
Only skinL1 slightly worse (+1.4). Per-ID: gains driven by 00013/00017/00034 (garment-detailed IDs).
**Panels:** `/tmp/eval_run14/panel_all.png`. **Next:** run15 (agnostic-bg conditioning) proceeds; but run14 suggests
pushing the WEIGHTING lever further (e.g. also re-weight garment/skin, or W_IMG_V6_B higher) may be the richer vein.
