# run58 — Qwen garment K/V branch + QUERY MASK. First in-loop interface to move the deployed garment.

Base run37 frozen. Garment latent -> copied Qwen blocks (CORRECT rope) -> K/V appended inside the
frozen block's joint attention at blocks 50/55/59. QUERY MASK: only garment-region C-slot queries
attend to garment K/V (verified: 4.9% of queries allowed). Bridge loss (garment-improve inside mask;
bg/skin/far punished vs frozen base; correct-vs-wrong sensitivity). LoRA+v6 frozen, SOAR off.
Weights: BG/SKIN/FAR=10/10/25 (reduced so the gate could open), W_SENS=8, wrong garment every step.

## Deployed (5-id OVERFIT, canonical render, branch ON vs OFF)
| | CLOUD | edit_bg | whole_bg | far_bg | PSNR | SSIM | garL1 | skinL1 |
|---|---|---|---|---|---|---|---|---|
| branch OFF (==run37) | 3.41 | -3.73 | -4.05 | -4.40 | 21.61 | .836 | 15.09 | 10.97 |
| branch ON | 4.02 | -4.33 | -4.59 | -4.87 | 21.88 | .843 | **13.79** | 10.86 |

## Verdict: PARTIAL. Real but weak, and FAILS 2 of 3 acceptance criteria.
+ FIRST in-loop interface to improve the DEPLOYED garment: garL1 15.09 -> 13.79 (-9%). Gate opened
  0.018 -> 0.165, in-loop GAIN (+0.003..0.005 mid-training) converted to a deployed garment gain.
  PSNR/SSIM up. Proves K/V-append + query-mask + relaxed-preservation CAN move the garment through a
  frozen base -- unlike proj_out (run51: 15->19.88 WORSE) and late-block (run55/56: NaN/GAIN=0).
- FAILS "bg/skin held": CLOUD 3.41->4.02, far_bg -4.40->-4.87 (darker), whole_bg worse. The INDIRECT
  leak (garment tokens change -> downstream self-attn spreads to bg) shows in the deployed image.
  The query mask blocks DIRECT attendance (works, 4.9%) but cannot stop indirect propagation.
- FAILS "correct beats wrong": wrong ~= gar all the way through (0.0754 vs 0.0754), despite W_SENS=8.
  It learned "improve the garment REGION", not "use THIS garment" -- NOT garment-identity-specific.

## Context vs the output-space refiner (run47/48)
run58 garment gain -1.30 garL1, WITH bg regression, NOT identity-specific.
run47/48 refiner:  garment gain -12.2 garL1 (15.2->3.0), far_bg BIT-IDENTICAL, no bg cost.
=> the refiner dominates on both garment and bg. In-loop injection's ceiling looks low here.

## Decision per the plan
run58's gate met "GAIN>0" but FAILED "wrong>gar" and "bg bounded", so it does NOT meet the bar to
raise preservation back up. To continue this path: fix identity-specificity (why W_SENS=8 doesn't make
wrong>gar) and the bg tradeoff. Ceiling is uncertain and clearly below the refiner.
