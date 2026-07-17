# run29 — SOAR init + run14 recipe (long-form setup CONTROL). Raw single-model, no composition.
Reference: SOAR CLOUD 9.16 / garL1 18.18 / skinL1 16.25 (good fg, bad bg). run14 3.34/19.12/19.63 (good bg, worse fg).
In-loop [DEPLOY] all 3 fell (garment 26.5→18.2, skin 13.2→11.6, bg 8.0→4.6). ~954 steps/2h (slow).

## Faithful eval (RAW) vs SOAR:
| metric | SOAR | run29 | vs SOAR | run14 |
|---|---|---|---|---|
| CLOUD | 9.16 | **4.42** | ✓✓ big | 3.34 |
| edit_bg | −5.69 | −3.74 | ✓ | −1.28 |
| garL1 | 18.18 | **17.27** | ✓ | 19.12 |
| skinL1 | 16.25 | **17.90** | ✗ WORSE | 19.63 |

## Verdict: 2/3 improve — bg hugely (9.16→4.42), garment (18.18→17.27) — but SKIN REGRESSED (16.25→17.90).
The run14 recipe from SOAR does NOT cleanly move all 3: **skin is the tradeoff casualty** (not garment). By the strict
rule (all 3 beat SOAR) this is NOT the long-form setup. Key insight: the shared denoising field improvement (bg+garment)
costs skin — the foreground losses aren't protecting skin. Motivates run30: route skin to mid/late blocks + low-σ so it
isn't collateral. Panels: run29_soar_run14recipe/.
