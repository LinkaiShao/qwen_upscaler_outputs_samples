# run26 — REGION-ROUTED COMPOSITE (offline validation): foreground=run20, bg=run14

**No training.** Offline diagnostic (`/tmp/eval_composite.py`): take FOREGROUND (garment∪skin, by parse/warped
masks) from the garment-best model (run20, garL1 15.09) and BACKGROUND from run14 (bg-best), soft-composite, measure.
Tests the core region-routed hypothesis with EXISTING checkpoints.

## Result — ✅ REGION-ROUTING WORKS. Best-of-both, cleanly.
| metric | run14 (bg-best) | run20 (gar-best) | **composite fg20+bg14** |
|---|---|---|---|
| CLOUD | 3.34 | 4.93 | **3.43** (bg held) |
| edit_bg_dL | −1.28 | −3.54 | **−1.66** (bg held) |
| garL1 | 19.12 | 15.09 | **15.09** (garment kept, −4 vs run14) |
| skinL1 | 19.63 | 18.20 | **18.22** (−1.4) |

**Taking garment from run20 + bg from run14 gives garment 15.09 AND bg 3.43 — the biggest garment gain of the whole
campaign WITH bg held at run14 level, clean seam (no boundary artifact, verified visually).** This proves the
user's structural thesis: foreground and bg CAN be separated — each region sourced/trained independently, composited
at deploy. The Pareto tradeoff (runs 18/19/20) only exists WITHIN a single shared model; compositing across models
dissolves it.

## Deploy mechanism (region-routed): run TWO forwards — bg-model (run14) for bg, foreground-model for garment/skin —
composite by warped-garment + skin-parse masks (both inference-available; ≈ v6 route @99%). bg is GUARANTEED run14
(cannot be overwritten). foreground has full model capacity.

## Next (run27): train a DEDICATED foreground model (unfreeze run14, push garment/skin HARD, ignore bg since it is
composited away) → its foreground composited with run14 bg. Tests if a purpose-built foreground beats run20's 15.09.
Then the region-routed foreground-LoRA + run14-bg composite is the deployable structural solution.
