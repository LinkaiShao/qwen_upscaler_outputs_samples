# run36 — PCGrad on shared LoRA+v6 (BREAKTHROUGH — all 3 regions improve/hold)

Split loss into L_garment/L_skin/L_bg; apply PCGrad (project conflicting gradients) to the SHARED LoRA+v6
params (95M); GarmentChain gets normal grad. Global injection (no mask-gate), no adapters, SOAR init.
Directly tests whether shared-GRADIENT CONFLICT is why fixing one region hurt another.

## PCGrad found REAL conflict: projected_conflicts up to 4/6 pairwise (steps 110-120) — the hypothesis holds.

## Faithful metrics (5-ID raw predict_sample) vs SOAR — ONLY ~290 steps (undertrained, 4x-slow PCGrad)
| metric | SOAR | run36 | result |
|---|---|---|---|
| CLOUD | 5.87 | **4.29** | better |
| whole_bg | -8.25 | **-4.92** | +3.33 MUCH cleaner |
| far_bg | -9.95 | **-5.23** | +4.72 MUCH cleaner |
| skinL1 | 12.97 | **10.97** | -2.00 better |
| garL1 | 15.12 | 15.20 | held (=SOAR) |
| PSNR/SSIM | 21.70/0.834 | 21.69/0.835 | = |

## Verdict: WIN. First run to beat SOAR on bg + skin while HOLDING garment — no cross-region tradeoff.
Panels confirm: background nearly clean white (vs SOAR/35A grey blobs), face clean, garment crisp. PCGrad
resolving the confirmed gradient conflict lets all regions improve together. This is the RIGHT lever — the
cross-region damage was shared-gradient conflict, not a missing write path (35A) or destabilization (run32).
Undertrained at ~290 steps and already this good → a full-length PCGrad run is the priority. run37 = PCGrad
+ run31 mask-gate (adds garment bifurcation on top). CAUTION: undertrained; confirm with a longer run.
