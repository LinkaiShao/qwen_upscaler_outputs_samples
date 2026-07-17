# run39 — temporal σ-routed loss (WEAK — ≈ SOAR, temporal routing insufficient)

run31 mask-gate + σ-gated region weights (bg=high-σ, skin=mid-σ, garment=low-σ, floor 0.2). No PCGrad, no
adapters. Well-trained (~900 steps, normal speed).

## Faithful vs SOAR / PCGrad
| metric | SOAR | run39 temporal | run36 PCGrad |
|---|---|---|---|
| CLOUD | 5.87 | 5.43 | **4.29** |
| whole_bg | -8.25 | -8.40 | **-4.92** |
| far_bg | -9.95 | -10.40 | **-5.23** |
| skinL1 | 12.97 | 14.33 | **10.97** |
| garL1 | 15.12 | 15.27 | 15.20 |

## Verdict: WEAK/NEGATIVE. Temporal σ-routing barely changed bg (−8.40 vs −8.25, slightly WORSE far/skin).
Confirms run30's earlier finding that temporal routing is insufficient. Assigning region roles by timestep
does NOT resolve the cross-region conflict — the losses still fight within each σ band. PCGrad (direct
gradient-conflict resolution) remains the unique winner. → temporal direction CLOSED.
