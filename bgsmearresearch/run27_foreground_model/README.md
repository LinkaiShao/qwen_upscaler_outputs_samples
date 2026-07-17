# run27 — dedicated foreground model (aggressive garment) + 3-way region-routed composite

**Run:** `runs/BGBAND27_130351`  Warm-start run14, unfrozen, garment MAXed (G/G_ID=6, S=4, multiblock LR 1e-4). bg irrelevant (composited from run14).
**In-loop quality warned early:** [DEPLOY] GARMENT L1 35.04(s378)→39.83(s723) — degrading. Aggressive garment weighting over-cooked it (run17 pattern). Should have flagged at ~40min.

## run27 own eval (faithful): garL1 17.06, skinL1 17.58, CLOUD 3.98.
Garment 17.06 = better than run14 (19.12) but WORSE than run20 (15.09) → aggressive maximization did NOT beat run20. Skin 17.58 = best of any model.

## Region-routed composites (foreground from model(s) + bg from run14; offline, existing ckpts):
| composite | CLOUD | edit_bg | garL1 | skinL1 |
|---|---|---|---|---|
| run14 (baseline, no composite) | 3.34 | −1.28 | 19.12 | 19.63 |
| 2-way fg20+bg14 (run26) | 3.43 | −1.66 | **15.09** | 18.22 |
| 2-way fg27+bg14 | 3.48 | −1.84 | 17.04 | 17.62 |
| **3-way: gar=run20, skin=run27, bg=run14** | **3.43** | −1.63 | **15.10** | **17.60** |

## VERDICT: run27 rejected as GARMENT source (17.04 > run20 15.09), accepted as SKIN source (17.60). Best deployable
result = **3-way composite: garment=run20, skin=run27, bg=run14 → garL1 15.10 (−21% vs run14), skinL1 17.60 (−10%),
CLOUD 3.43 (bg held).** Each region from the model best at it. Fully validates region-routing / structural separation.

## Deploy: N-forward region-routed composite (garment-model + skin-model + bg-model, masked by warped-garment/skin
parse). Cost = N× inference. Productionizing options (USER DECISION): (a) ship the composite (2–3 forwards); (b) train
ONE region-routed foreground LoRA that yields good garment AND skin, composited with run14 bg (2 forwards); (c) distill.
