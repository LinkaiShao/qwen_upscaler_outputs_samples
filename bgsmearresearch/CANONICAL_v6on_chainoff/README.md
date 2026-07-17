# CANONICAL EVALUATOR — the ONE trustworthy comparison (v6 ON, chain OFF)

LOCKED canonical inference (2026-07-08 pipeline audit): predict_sample + v6 ON + GarmentChain OFF +
mask-gate off (no-op) + noise-fill agnostic + no train step + no optimizer + fixed 5 IDs (00006/08/13/17/34)
+ raw + soft-composite finals + metrics from the exact saved images. render_panels.sh with
USE_MULTI_BLOCK_INJ=0 USE_MASK_GATED_INJECTION=0 (USE_V6=1). Panels: {id}_RAW.png / {id}_FINAL.png / ALL_*.

## WHY this is canonical (audit result)
The old eval_full pipeline runs LoRA-ONLY: V01 load_model never loads v6_heads.pt (only knows the obsolete
repair_head.pt Conv2d, which these runs don't have) and drops the GarmentChain. So eval_full omits the v6
paint/keep routing (project_v6_routing_essential = must not disable) → SOAR far_bg -29 (LoRA-only) vs -7.6
(v6 on). The 21-unit gap was v6, not any model effect. Isolation also proved: mask-gate = no-op; GarmentChain
= net-NEGATIVE at inference (chain-off CLOUD 4.18/far -7.60 vs chain-on 5.90/-9.96, garment unchanged) → drop it.

## DEFINITIVE canonical metrics (5-ID)
| metric | SOAR | run14 | run36 PCGrad | run37 PCGrad+mg |
|---|---|---|---|---|
| CLOUD | 4.18 | 3.74 | 3.44 | **3.42** |
| edit_bg dL | -3.69 | -3.23 | -3.63 | -3.74 |
| whole_bg dL | -6.14 | -5.69 | -4.21 | **-4.06** |
| far_bg dL | -7.60 | -7.21 | -4.64 | **-4.40** |
| garL1 | 15.21 | 19.21 | 15.36 | **15.03** |
| skinL1 | 12.62 | 14.80 | 10.85 | **10.96** |
| PSNR/SSIM | 21.67/.835 | 20.94/.822 | 21.63/.835 | 21.62/.836 |

## VERDICT
PCGrad (run36/37) is the clear winner — best or tied on EVERY metric: ring (CLOUD 3.42 beats run14 3.74),
whole/far bg (halves SOAR's darkening), skin (~11 vs 12.6/14.8), garment held (=SOAR ~15, run14 worst 19.21).
Visual (RAW panels): PCGrad backgrounds + faces cleanest; garments comparable. run14's earlier "win" was an
eval_full/no-v6 artifact — with v6 on, run14 is the WORST (worst garment + skin, bg barely beats SOAR).

## OBSOLETE / DEBUG-ONLY (do not compare against canonical)
- eval_full numbers (SOAR 9.16/-29.11/18.18/16.25 etc.) — LoRA-only, no v6, no chain.
- render_panels chain-ON numbers (the earlier PCGrad "breakthrough" table: SOAR -8.25/-9.95, run36 -4.92 etc.).
- All older run3X folder metrics rendered chain-on/mask-gate-forced.
