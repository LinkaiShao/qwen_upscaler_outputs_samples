# run15 — run03 + agnostic-bg conditioning (hole fill noise→INPAINT)

**Run:** `runs/BGBAND15_053634`  **Warm-start:** run03  **Budget:** 2 h (1010 steps)  **Launcher:** `run_bgband15.sh`
**Single conceptual change:** hole fill noise → agnostic-INPAINT (`USE_AGNOSTIC_RANDOM_FILL=0 USE_AGNOSTIC_INPAINT=1
AGNOSTIC_INPAINT_SOFT_SIG=2`, iters20 k7 σ2). Propagates real surrounding bg/body into the hole so the model sees
authoritative bg color at the boundary. Agnostic-v3.2 hole is already grey (garment removed) ⇒ no garment leak.
**Eval:** `/tmp/eval_full_inpaint.py` (MATCHED inpaint fill — train==deploy faithful; NOT the noise-fill eval).
`USE_BG_HINT` (the cleaner explicit-bg-channel) was BLOCKED — 0 densepose files in cache.

## Hypothesis (user experiment #4)
Make agnostic bg authoritative during generation → does the bg darkening drop without a new loss.

## Result — ✗ NOT accepted (deploy gate); but BEST raw bg of any run.
| metric | run03 | run14 (best) | run15 | vs run03 |
|---|---|---|---|---|
| CLOUD (ring) | 3.28 | 3.34 | 5.70 | ✗ (00008 outlier 12.0) |
| edit_bg_dL | −1.31 | −1.28 | −6.09 | ✗✗ (00008 −18.2) |
| edit_nonbg_dL | −5.02 | −2.91 | −7.16 | ✗ |
| **edit_all_dL (PRIMARY gate)** | −3.78 | **−2.37** | −6.60 | ✗ worst |
| whole_bg_dL (raw bg) | −19.71 | −10.51 | **−6.26** | ✓✓ best |
| far_bg_dL (raw corners) | −36.65 | −17.12 | **−6.97** | ✓✓✓ best |
| edit PSNR / SSIM | 19.52/0.829 | 19.88/0.831 | 20.34/0.829 | PSNR best |
| garL1 / skinL1 | 22.01/18.23 | 19.12/19.63 | 19.32/18.56 | ~run14 |

**Reading:** Inpaint conditioning made the RAW bg nearly correct (far −37→−7 — the model does read agnostic bg when
given it authoritatively). But it REGRESSED the deploy hole patch (edit_all −6.60), with 00008 catastrophic
(edit_bg −18, CLOUD 12 — inpaint propagated dark boundary content into that hole). **Third confirmation of the
robust pattern: bg-authority/luminance interventions (runs 12,13,15) fix RAW bg but darken the DEPLOY hole patch.**
Only bg-region WEIGHTING (run14) improved the deploy patch. **run14 remains deploy-best.**
**Panels:** `/tmp/eval_run15/panel_all.png`. **Implication:** stop pursuing bg-authority for the deploy patch;
the weighting lever (run16 pushes it) is the productive vein.
