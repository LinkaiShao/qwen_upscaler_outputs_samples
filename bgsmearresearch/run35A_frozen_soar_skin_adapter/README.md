# run35A — FROZEN SOAR + isolated skin adapter (clean causality test)

Freeze SOAR ENTIRELY (LoRA + v6 + GarmentChain + backbone); train ONLY a small skin adapter (122.9M, 3
DISJOINT blocks 16,24,40, gate=1.0, zero-init out_proj). Fixes run32's three failure modes: co-location
(disjoint blocks), destabilization (frozen base), dead gate (no 0.01 cap). Only the skin adapter trains
(verified: optimizer holds skin_adapter only). Judged by FAITHFUL chain-on render vs SOAR, not in-loop.

## Faithful metrics (5-ID, raw predict_sample) vs SOAR
| metric | SOAR | 35A | result |
|---|---|---|---|
| garL1 | 15.12 | **15.06** | = SOAR ✓ (freeze worked; garment locked) |
| skinL1 | **12.97** | 13.18 | skin slightly WORSE (+0.21) |
| whole_bg | -8.25 | -8.99 | bg darker |
| far_bg | -9.95 | -10.93 | bg darker (-0.98) |
| CLOUD | 5.87 | 6.28 | worse |
| PSNR/SSIM | 21.70/0.834 | 21.68/0.833 | = |

Adapter engaged (out_proj norm 0→0.09 at all 3 sites, gates 1.0 — run32's dead-gate 0.01 FIXED).

## Verdict: REJECT the skin-adapter mechanism (2 things proven)
1. FREEZE PROTECTION WORKS — garment locked identical to SOAR (15.06 vs 15.12), zero destabilization.
   The run32 destabilization is methodologically fixable by freezing. GOOD.
2. But the ISOLATED skin adapter does NOT improve skin — skin got slightly WORSE and its write propagated
   into bg (darker + heavier speckle artifacts, visible in panels). Per the reject criterion (artifacts →
   adapter design/source/mask wrong), the region-owned SKIN ADAPTER mechanism is NOT validated.
→ Do NOT scale adapters (35B not justified). run32 failed for TWO reasons: destabilization (fixable) AND a
  skin adapter that doesn't help skin (design/source/mask). Fixing #1 exposed #2. Skin damage is likely a
  training-recipe problem (SOAR has the best skin; training regresses it), not a missing-write-path problem.
  → makes PCGrad (36/37, no adapters) and temporal routing (39) the more promising directions.
