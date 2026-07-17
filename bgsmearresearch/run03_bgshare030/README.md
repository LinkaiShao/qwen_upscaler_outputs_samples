# run03 — bg share 0.30 (back off from 0.35 to protect garment)

**Run:** `runs/BGBAND02_000508`   **Warm-start:** soar   **Budget:** 2 h   **Status:** _training_
**Launcher:** `run_bgband02.sh` with `PCT_FLOW_BG=0.30 RPCT_BG=0.30`

## Why
run02 (bg 0.35) improved bg a lot (CLOUD 8.19→5.16) **but regressed garment** (deployed garment L1 up on 2/3 IDs; rule-3 flag). Per the A/B plan, 0.40 is off (garment not stable). Single-variable change from run02: **bg share 0.35 → 0.30** in both the flow split and the region-%; everything else identical (garment 0.50, skin 0.20, gem+bsab in bundle, SOAR rollout + garment/person protect).

## Hypothesis
A gentler bg share keeps most of run02's bg gain while restoring garment. Interpretable point on the share curve: soar(no framework) 8.19 → run03 bg-0.30 → run02 bg-0.35 5.16.

## Success criteria (raw predict_sample CLOUD + garment)
- **bg:** mean CLOUD still well below soar 8.19 (ideally near run02's 5.16).
- **garment:** deployed garment L1 back to ~soar level (no regression) — this is the gate.
- If bg holds AND garment recovers → 0.30 is the pick. If bg reverts to ~8 → the bg gain needs the higher share, and share-vs-garment is a hard tradeoff → pivot to explicit bias-correction.

## Result — raw `predict_sample` CLOUD → ✅ NEW BEST bg (4.84); garment L1 still drifting

| ID | CLOUD | Δ vs soar | darkening |
|---|---|---|---|
| 00006 | 5.74 | -1.30 | -4.01 |
| 00008 | 6.85 | -1.25 | -4.27 |
| 00013 | 4.01 | -4.09 | -3.56 |
| 00017 | 3.54 | -4.21 | -2.14 |
| 00034 | 4.04 | -5.94 | -3.46 |
| **mean** | **4.84** | **-3.36** | ~-3.5 |

**bg:** best result so far — 8.19 → 4.84, below run02's 5.16, approaching if5's 3.4. Panels clearly cleaner.
**garment:** deployed garment L1 did NOT recover by lowering share — 00013 17.9(soar)→21.1(run02)→27.5(run03), 00034 8.2→12.7→12.1. VISUALLY still acceptable (00013 tee logo/stripes intact), but the L1 trend up with bg-heavy training is a rule-3 signal. Config: PCT_FLOW_CORE=0.50 BODY=0.20 BG=0.30 UB=0.04 KEEP=0.01 ; RPCT_GARMENT=0.50 SKIN=0.20 BG=0.30 BOUNDARY=0.08 KEEP=0.02.

**Decision:** bg is basically solved by the share lever (4.84). Remaining issue = garment drift on detailed garments. Next single lever = raise GARMENT share (rule 3 priority), not bg. → run04.
