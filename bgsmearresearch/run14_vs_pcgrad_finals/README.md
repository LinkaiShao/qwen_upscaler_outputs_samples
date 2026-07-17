# run14 vs PCGrad — deployed FINALS (TRUSTED eval_full pipeline, chain-off, no mask-gate)

IMPORTANT: rendered via eval_full (the SAME pipeline that made run14's original panels). The earlier
render_panels versions were WRONG — they force-applied run32's mask-gate + agnostic env to run14/run36/SOAR
which trained without it, softening/distorting the garment and giving bogus metrics. These are the corrected,
consistent, apples-to-apples renders.

## Faithful metrics (eval_full, 5-ID) vs SOAR (9.16 / -19.26 / -29.11 / 16.25 / 18.18)
| metric | SOAR | run14 | run36 PCGrad | run37 PCGrad+mg |
|---|---|---|---|---|
| CLOUD (silhouette ring) | 9.16 | **3.34** | 5.02 | 5.34 |
| edit_bg (revealed bg) | -5.69 | **-1.28** | -5.22 | -5.91 |
| whole_bg | -19.26 | -10.51 | -5.96 | **-5.87** |
| far_bg | -29.11 | -17.12 | -6.95 | **-6.60** |
| garL1 | 18.18 | 19.12 | 18.19 | **18.15** |
| skinL1 | 16.25 | 19.63 | 15.05 | **14.99** |

## Verdict (corrected)
- run14 = best silhouette RING (CLOUD 3.34, edit_bg -1.28) — the crisp edge the eye notices. But worst
  garment (19.12) and worst skin (19.63).
- PCGrad (run36/37) = best whole/far bg, best skin (~15 vs 16.25), garment HELD (=SOAR ~18.15). Ring slightly
  softer than run14 (5.0 vs 3.34).
- Prior "PCGrad garment 15 crushes run14 19" was FALSE (render_panels artifact). On the trusted pipeline all
  garments ~18; PCGrad ties SOAR, run14 slightly worse.
- Net: run14 wins the ring; PCGrad wins bg breadth + skin + ties garment. A ring-vs-rest tradeoff.
