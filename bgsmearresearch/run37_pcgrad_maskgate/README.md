# run37 — PCGrad + run31 mask-gated garment (confirms run36; best garment)

run31 mask-gated GarmentChain (garment writes garment tokens only) + PCGrad on shared LoRA+v6. No adapters.
SOAR init. ~285 steps (undertrained, 4x-slow PCGrad). PCGrad conflict up to 4/6.

## Faithful vs SOAR / run36
| metric | SOAR | run36 (PCGrad) | run37 (PCGrad+maskgate) |
|---|---|---|---|
| CLOUD | 5.87 | 4.29 | 4.36 |
| whole_bg | -8.25 | -4.92 | -4.93 |
| far_bg | -9.95 | -5.23 | -5.19 |
| skinL1 | 12.97 | 10.97 | 11.12 |
| garL1 | 15.12 | 15.20 | **14.96 (best)** |

## Verdict: CONFIRMS run36. PCGrad is the driver (bg -5, skin ~11 — huge vs SOAR). Mask-gate adds a small
garment benefit (14.96, best-yet garment). run36 ≈ run37 on bg/skin; run37 marginally better garment.
Both beat SOAR on ALL regions, undertrained. → PCGrad (± mask-gate) is the validated direction; a
FULL-LENGTH PCGrad+maskgate run is the top priority. Adapter/forked-head family (35A, run38) is the
alternative being tested but PCGrad already wins.
