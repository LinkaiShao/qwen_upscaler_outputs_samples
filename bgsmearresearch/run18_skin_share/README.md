# run18 (A) — run14 + skin share (region-% only)

**Run:** `runs/BGBAND18_160239`  **Warm-start:** run03  **Budget:** 2 h (~1000 steps)  **Launcher:** `run_bgband18.sh`
**Single lever vs run14:** `RPCT_SKIN` 0.20→0.25, `RPCT_BG` 0.30→0.25. Everything else = run14 (W_IMG_V6_B=8). Noise-fill ⇒ standard eval.

## Result — ✗ REJECT per acceptance rule (bg NOT protected), but strongest garment/skin signal of the queue.
| metric | run14 | run18 | acceptance (keep if) | pass? |
|---|---|---|---|---|
| CLOUD | 3.34 | **4.24** | ≤ ~3.5 | ✗ |
| edit_bg_dL | −1.28 | −3.38 | (bg protect) | ✗ bg hole darker |
| edit_nonbg_dL | −2.91 | +1.12 | — | ✓ gar/skin no longer dark |
| edit_all_dL | −2.37 | −0.16 | ≥ −2.5 | ✓ (patch much less dark) |
| whole_bg_dL | −10.51 | −10.57 | — | flat |
| far_bg_dL | −17.12 | −16.02 | — | flat |
| garL1 | 19.12 | **16.83** | ≤ 19.12 | ✓ better |
| skinL1 | 19.63 | **18.14** | ≤ 19.63 | ✓ better |
| edit PSNR/SSIM | 19.88/0.831 | 20.27/0.825 | — | PSNR ✓ |

**Reading:** Shifting 5% region share bg→skin recovered skin (18.14) AND garment (16.83) AND de-darkened the whole
edit patch (edit_all −2.37→−0.16, gar/skin in hole flipped positive). BUT it took the share straight off bg, so the
**bg ring regressed** (CLOUD 3.34→4.24, edit_bg −1.28→−3.38). Panels: garments crisp, skin natural, but bg near
silhouette slightly darker. **Fails the "bg protected" gate → not accepted.**
**Panels:** `run18_skin_share/panel_all.png`.
**Follow-up (not auto-run):** the direction clearly works — retry a MILDER shift (e.g. RPCT_SKIN 0.22, RPCT_BG 0.28,
or take the 2% from KEEP/BOUNDARY instead of bg) to recover skin without the CLOUD hit. Best skin lever so far.
