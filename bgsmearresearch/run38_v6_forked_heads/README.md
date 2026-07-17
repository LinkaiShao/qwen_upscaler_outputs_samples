# run38 — frozen-SOAR + v6 forked per-region heads (WEAK — residual-writer family fails, like 35A)

Freeze SOAR LoRA + GarmentChain; train ONLY the v6 forked per-region heads (639K params: to_s/to_b/to_route,
zero-init via V6_REFINE) at the output stage, mask-routed. WELL-TRAINED (~1960 steps, small trainable = fast).

## Faithful vs SOAR / PCGrad
| metric | SOAR | run38 forked heads (~1960 steps) | run36 PCGrad (~285 steps) |
|---|---|---|---|
| CLOUD | 5.87 | 5.71 | **4.29** |
| whole_bg | -8.25 | -8.01 | **-4.92** |
| far_bg | -9.95 | -9.67 | **-5.23** |
| skinL1 | 12.97 | 12.81 | **10.97** |
| garL1 | 15.12 | 15.20 | 15.20 |

## Verdict: WEAK / NEGATIVE. Forked per-region heads barely moved from SOAR (whole_bg -8.01 vs -8.25) DESPITE
7x more training than PCGrad — while PCGrad (undertrained) achieved a massive improvement (whole_bg -4.92).
This confirms 35A: the per-region RESIDUAL-WRITER family (cross-attn adapters OR forked output heads) does
NOT fix the cross-region damage. The problem is shared-GRADIENT CONFLICT, and the fix is PCGrad, not adding
more write paths. → adapter/forked-head direction CLOSED. PCGrad is the answer.
