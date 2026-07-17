# run12 — run03 + edit-bg luminance anchoring (LAMBDA_BG_FIELD)

**Run:** `runs/BGBAND12_232005`  **Warm-start:** run03 (`BGBAND02_000508/final`)  **Budget:** 2 h  **Status:** _training_
**Launcher:** `run_bgband12.sh`  **Single lever:** `LAMBDA_BG_FIELD=2.0` (everything else identical to run03)

## Hypothesis (user experiment #1)
"Qwen generates the right structure but its edit-region luminance prior is too dark." Add a direct
per-pixel loss on generated background pixels *inside the edit hole* to match an **agnostic-derived
background field** (blur-inpaint of the VISIBLE bg into the hole). Deploy-legal — field source is
`parse_bg ∩ (outside hole)` real pixels, **no GT of the hidden region** (`trainlib/losses/background.py:404-424`, `_vis_bg`).

## Baseline (run03, full harness `/tmp/eval_full.py`, fixed 5 IDs, raw predict_sample seed 103)
| metric | run03 |
|---|---|
| CLOUD (ring) | 3.28 |
| **edit_bg_dL** (deploy hole bg) | **−1.31** |
| edit_nonbg_dL (gar/skin in hole) | −5.02 |
| **edit_all_dL** (whole edit patch = PRIMARY gate) | **−3.78** |
| whole_bg_dL (raw bg incl far; pasted over at deploy) | −19.71 |
| far_bg_dL (raw corners; pasted over) | −36.65 |
| edit PSNR / SSIM | 19.52 / 0.829 |
| garL1 / skinL1 | 22.01 / 18.23 |

**Note:** run03's deploy-relevant hole bg (edit_bg_dL) is already only −1.31. The big whole-bg/far darkening
is in the RAW generation and is masked by the deploy paste. This lever targets the hole bg specifically.

## Success criteria (vs run03)
- edit_bg_dL toward 0 **without overshooting positive** and **without garL1/skinL1 regression** (rule: lower CLOUD but darker/worse patch = NOT better).
- edit_all_dL not worse; edit PSNR/SSIM stable or better.
- Judge by panels (`/tmp/eval_run12`) + the whole-edit-patch darkening, not CLOUD alone.

## Result — 1010 steps (`bgf` lever active ~0.03 throughout). Verdict: ✗ NOT accepted over run03 (deploy gate), but ✓✓ fixes the RAW whole-bg darkening.

| metric | run03 | run12 | Δ |
|---|---|---|---|
| CLOUD (ring) | 3.28 | 4.38 | +1.10 ✗ worse |
| edit_bg_dL (deploy hole bg) | −1.31 | −2.73 | −1.42 ✗ darker |
| edit_nonbg_dL | −5.02 | −7.23 | −2.21 ✗ |
| **edit_all_dL (PRIMARY gate)** | **−3.78** | **−5.81** | −2.03 ✗ darker patch |
| whole_bg_dL (raw bg) | −19.71 | −8.53 | +11.2 ✓✓ |
| far_bg_dL (raw corners) | −36.65 | −11.72 | +24.9 ✓✓✓ |
| edit PSNR / SSIM | 19.52 / 0.829 | 19.49 / 0.833 | flat |
| garL1 / skinL1 | 22.01 / 18.23 | 22.42 / 19.17 | ~flat (slightly worse) |

**Reading:** `LAMBDA_BG_FIELD=2.0` trained a brighter GLOBAL bg-luminance prior — the RAW generated bg darkening (the user's visual complaint) collapsed (far −37→−12). But it darkened the deploy-relevant hole patch (edit_all −3.78→−5.81) and worsened CLOUD. By the rule (lower CLOUD or darker patch ⇒ not better), run12 is **not** a deploy win over run03. The per-pixel field target near the silhouette (blurred-in visible bg, slightly vignette-dark) pulls the hole slightly darker even as the global prior brightens.
**Panels:** `/tmp/eval_run12/panel_all.png` (pred|final|gt|dL-heatmap). **Next:** run13 tests low-freq MEAN anchor (LAMBDA_BG_CHROMA) — may lift the hole without the field's near-silhouette darkening.
