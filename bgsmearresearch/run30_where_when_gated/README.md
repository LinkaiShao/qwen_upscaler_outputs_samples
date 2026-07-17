# run30 — where+when gated (3-tier block LR + sigma sched) from SOAR. RAW, no composite.
3-tier block LR: 0-20 @ 0.25x (bg), 21-44 normal, 45-59 @ 2x (garment) + run20 multiblock 7.5e-5 + sigma-sched (bg high-σ, garment low-σ). run14 bg ratios.
## Faithful vs SOAR (9.16/18.18/16.25), run29 (4.42/17.27/17.90):
| metric | SOAR | run29 | run30 | verdict |
|---|---|---|---|---|
| CLOUD | 9.16 | 4.42 | **3.64** | ✓✓ best SOAR-init bg (~run14 3.34) |
| edit_bg | -5.69 | -3.74 | -2.64 | ✓ |
| garL1 | 18.18 | 17.27 | 18.32 | ✗ regressed |
| skinL1 | 16.25 | 17.90 | 18.15 | ✗ regressed |
## Verdict: gating gave EXCELLENT bg (3.64) but STARVED foreground (garment 18.32 + skin 18.15 both worse than SOAR).
Early blocks 0.25x too frozen + garment sigma-restricted → foreground under-trained. In-loop garment-rising was right.
Block/sigma ALLOCATION just relocates the tradeoff (run29 lost skin; run30 lost garment+skin for bg). Scalar/allocation
approach exhausted — none get all 3. → STRUCTURAL fix needed: mask-gated hidden-state injection (run31+).
