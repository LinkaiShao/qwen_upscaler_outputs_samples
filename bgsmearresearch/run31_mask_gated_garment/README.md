# run31 — mask-gated GARMENT injection (region-owned write). SOAR init, run29 losses, RAW no composition.
GarmentChain injection multiplied by garment token mask (spatial_mask garment=1/boundary=0.4/bg=0) so garment features write ONLY into garment tokens. multiblock 7.5e-5.
## Faithful vs SOAR (9.16/18.18/16.25), run29 (4.42/17.27/17.90), run20-global-7.5e-5 (4.93/15.09/18.20):
| metric | SOAR | run31 | verdict |
|---|---|---|---|
| CLOUD | 9.16 | **4.34** | ✓ held (vs run20 global 4.93 — mask-gating protects bg at same strength) |
| edit_bg | -5.69 | -3.53 | ✓ |
| garL1 | 18.18 | 17.74 | ✓ modest (in-loop 14.48 over-promised) |
| skinL1 | 16.25 | 18.82 | ✗ WORSE (worst yet) |
## Verdict: PARTIAL. Mask-gating garment injection PROTECTS bg (4.34 vs run20 global 4.93 at same 7.5e-5) — region-owned
write stops garment from harming bg. But garment gain modest and SKIN regressed (18.82) — skin has no region-owned path.
Confirms: region-owned garment write works for bg; skin needs its own adapter → run32. (Note in-loop garment proxy
over-promises: 14.48 in-loop vs 17.74 faithful.)
