# run04 — garment share 0.60 (bg 0.30 held) — REJECTED

**Run:** `runs/BGBAND02_090927`   **Warm-start:** soar   **Budget:** 2h (7205s, done)
**Exact shares:** `PCT_FLOW_CORE=0.60 PCT_FLOW_BODY=0.20 PCT_FLOW_BG=0.30 PCT_FLOW_UB=0.04 PCT_FLOW_KEEP=0.01` ; `RPCT_GARMENT=0.60 RPCT_SKIN=0.20 RPCT_BG=0.30 RPCT_BOUNDARY=0.08 RPCT_KEEP=0.02`. Single lever vs run03: garment 0.50→0.60.

## Result — raw predict_sample CLOUD → ⚠ bg WORSE, garment NOT recovered → REJECTED

| ID | CLOUD | Δ vs soar | vs run03 |
|---|---|---|---|
| 00006 | 5.93 | -1.11 | +0.19 |
| 00008 | 7.93 | -0.17 | +1.08 |
| 00013 | 5.17 | -2.93 | +1.16 |
| 00017 | 4.85 | -2.90 | +1.31 |
| 00034 | 4.51 | -5.47 | +0.47 |
| **mean** | **5.68** | **-2.52** | **+0.84 (worse)** |

Garment L1 (deployed): 00006 11.0→10.0, 00013 17.9→28.7 (worse), 00034 8.2→9.9 — mixed, NOT clearly recovered. Skin comparable.

## Visual verdict (00006/00013/00034)
Garments look fine — 00013 tee logo + side stripes intact, close to GT (the ~28 L1 is detail/color difference, not degradation). bg still cleaner than soar but visibly less clean than run03.

## Conclusion
Raising garment share to 0.60 **cost bg** (CLOUD 4.84→5.68) **without visibly improving garment**. The 00013 garment L1 is a persistent artifact across all bifurcated runs (visually fine), not a share-fixable regression. **run04 rejected; run03 (bg 0.30, CLOUD 4.84) remains the best valid run.** Do not keep raising garment share.
