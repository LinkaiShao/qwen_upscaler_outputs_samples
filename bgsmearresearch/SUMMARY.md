# bgsmear queue (runs 12–17) — SUMMARY & final verdict

Goal: fix the background darkening in the VTON model, deploy-faithfully. All runs warm-start from **run03**
(`BGBAND02_000508`), 2 h each (~1000 steps), single interpretable lever, fixed 5 IDs, raw `predict_sample`
(seed 103, 20 steps), evaluated with `/tmp/eval_full.py` (run15 used the matched inpaint eval `/tmp/eval_full_inpaint.py`).

## Metric primer
- **edit_all_dL** = signed luminance (gen−GT) over the whole edit hole = **PRIMARY deploy gate** (lower magnitude/less negative = better). This is the pasted region the user sees.
- edit_bg_dL / edit_nonbg_dL = hole split into bg vs garment/skin.
- whole_bg_dL / far_bg_dL = RAW generated bg incl. far corners. **At deploy the far bg is pasted from the real image, so these are NOT in the product** — raw-image quality only.
- CLOUD = 8–50px ring metric. garL1 / skinL1 = region reconstruction. PSNR/SSIM on the edit region.

## Deploy-gate ranking (edit_all_dL, best→worst)
| rank | run | lever | edit_all_dL | CLOUD | garL1 | skinL1 | whole_bg_dL | verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **run14** | **bg-region weight W_IMG_V6_B 5→8** | **−2.37** | 3.34 | **19.12** | 19.63 | −10.51 | ✅ **DEPLOY-BEST** |
| 2 | run03 | (baseline) | −3.78 | 3.28 | 22.01 | 18.23 | −19.71 | baseline |
| 3 | run17 | B=8 + garment/skin weight +20% | −4.80 | 4.54 | 22.02 | 18.84 | −17.91 | ✗ regressed |
| 4 | run16 | bg-weight B=11 (overshoot) | −5.23 | 4.06 | 20.21 | 18.25 | −11.34 | ✗ overshoot |
| 5 | run12 | LAMBDA_BG_FIELD (per-pixel bg field) | −5.81 | 4.38 | 22.42 | 19.17 | −8.53 | ✗ deploy regress |
| 6 | run13 | LAMBDA_BG_CHROMA (bg-mean anchor) | −5.90 | 4.24 | 23.54 | 18.07 | −10.03 | ✗ deploy regress |
| 7 | run15 | agnostic-INPAINT conditioning | −6.60 | 5.70 | 19.32 | 18.56 | **−6.26** | ✗ deploy regress (BEST raw bg) |

## Two robust findings
1. **bg-authority / bg-luminance interventions (runs 12, 13, 15) fix the RAW background but REGRESS the deploy hole patch.** Three independent lever types (per-pixel field, mean anchor, inpaint conditioning) all show it. run15 (inpaint) gives the best raw bg of all (far −37→−7) — the model genuinely reads agnostic bg when given it authoritatively — but darkens the deployed patch. Since deploy pastes the far bg from the real image, these help an image the product never shows.
2. **The one thing that improves the deploy patch is bg-region loss WEIGHTING (run14, W_IMG_V6_B=8).** It's a fragile optimum: the curve is 5→−3.78, **8→−2.37**, 11→−5.23 (run16 overshoots), and adding garment/skin weight on top (run17) undoes it. The gain is a *relative region-% rebalance*, not more bg or more garment/skin loss.
3. The deploy-patch darkening is *dominated by* garment/skin in the hole (edit_nonbg), but the *fix* is not up-weighting garment/skin reconstruction (run17 proved that hurts) — it's the bg-region up-weight that rebalances allocation.

## Recommendation
- **Adopt run14 (`runs/BGBAND14_033138/final`, W_IMG_V6_B=8) as the new deploy checkpoint over run03**: better deploy patch (edit_all −3.78→−2.37), better garment (garL1 22.0→19.1), CLOUD flat. Only skinL1 slightly worse (+1.4).
- The user's original visual complaint ("the entire generated background is dark") is a RAW-image effect (far bg −37) that the deploy paste already removes; if a clean RAW generation is also wanted, run15's inpaint conditioning is the lever — but it needs the 00008-type instability (edit_bg −18 on one ID) fixed first.
- Next levers worth trying (not in this queue): (a) confirm run14 with a fresh seed / more IDs before locking; (b) bracket the bg-weight optimum tighter (W_IMG_V6_B 6, 7, 9); (c) attack the garment/skin darkening via the region-% allocation (RPCT_*) rather than the img weights.

## Artifacts
Per-run: `bgsmearresearch/run12–17/README.md` (+ run03/run14 baselines). Panels: `/tmp/eval_runNN/panel_all.png`
(pred | final-paste | gt | dL-heatmap). Harness: `/tmp/eval_full.py`, `/tmp/eval_full_inpaint.py`. Launchers: `run_bgband12–17.sh`.
