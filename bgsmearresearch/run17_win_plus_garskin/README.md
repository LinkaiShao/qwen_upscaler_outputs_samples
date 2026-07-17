# run17 — run14 winner (bg-weight B=8) + garment/skin weight bump

**Run:** `runs/BGBAND17_094714`  **Warm-start:** run03  **Budget:** 2 h (~1000 steps)  **Launcher:** `run_bgband17.sh`
**Change vs run03:** W_IMG_V6_B 5→8 (run14's win) AND garment/skin img weight +~20% (W_IMG_V6_G 4→5, W_IMG_V6_S 3→3.6).
Tests whether adding a direct garment/skin-darkening lever improves on run14. Noise-fill ⇒ standard eval.

## Result — ✗ REGRESSED vs run14. Adding garment/skin weight UNDID run14's benefit.
| metric | run03 | run14 (best) | run17 | vs run14 |
|---|---|---|---|---|
| CLOUD | 3.28 | 3.34 | 4.54 | ✗ |
| edit_bg_dL | −1.31 | −1.28 | −3.01 | ✗ |
| edit_nonbg_dL | −5.02 | −2.91 | −5.75 | ✗ |
| **edit_all_dL (PRIMARY gate)** | −3.78 | **−2.37** | −4.80 | ✗ worse than run14 AND run03 |
| whole_bg_dL | −19.71 | −10.51 | −17.91 | ✗ |
| far_bg_dL | −36.65 | −17.12 | −31.44 | ✗ |
| garL1 | 22.01 | 19.12 | 22.02 | ✗ (lost run14's gain) |
| skinL1 | 18.23 | 19.63 | 18.84 | ✓ slight |
| edit PSNR / SSIM | 19.52/0.829 | 19.88/0.831 | 19.55/0.830 | ✗ |

**Reading:** Combining B=8 with a garment/skin weight bump broke run14's improvement — edit_all back to −4.80 (worse
than baseline), garL1 back to 22.0. ⇒ **run14's win is specifically the bg-region weight at 8 with garment/skin left
at defaults; it's a RELATIVE-balance effect, and raising garment/skin weight counteracts it.** The darkening driver
is garment/skin, but the *fix* is not raising their reconstruction weight — it's the bg-region up-weight that
rebalances the region-% allocation. **run14 remains the deploy-best.**
**Panels:** `/tmp/eval_run17/panel_all.png`.
