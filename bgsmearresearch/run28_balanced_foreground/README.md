# run28 — BALANCED foreground model (garment + skin) → 2-forward region-routed composite

**Run:** `runs/BGBAND28_151137`  Warm-start run14, unfrozen. run20's garment lever (multiblock LR 7.5e-5) + modest skin
(W_IMG_V6_S=4), WITHOUT run27's over-cooked garment weights. Goal: ONE foreground model good at garment AND skin.
Ran full 2h (7212s) but only 956 steps (~7.5s/it — slower, in-loop DEPLOY-eval + eval contention).

## Quality-first (in-loop [DEPLOY], the check I now do live):
| step | GARMENT L1 | SKIN L1 | BG L1 | chroma |
|---|---|---|---|---|
| 375 | 37.17 | 11.37 | 5.20 | 0.5 |
| 724 | **17.36** | 10.48 | 3.36 | 0.6(=GT) |
Garment IMPROVING (37→17), img_g 0.10→0.085 falling — healthy, NOT run27's over-cook (35→40). Kept it running on this signal.

## Faithful eval + composite (foreground=run28, bg=run14):
| metric | run14 | run28 own | run28 fg + run14 bg | 3-way best |
|---|---|---|---|---|
| CLOUD | 3.34 | 3.98 | **3.46** (bg held) | 3.43 |
| edit_bg | −1.28 | −2.97 | −1.68 | −1.63 |
| garL1 | 19.12 | 16.24 | **16.24** (−15%) | 15.10 |
| skinL1 | 19.63 | 17.69 | **17.70** (−10%) | 17.60 |

## Verdict: ✅ ONE balanced foreground model works. run28 fg + run14 bg = garment 16.24, skin 17.70, bg held (3.46) —
nearly matching the 3-way best (15.10/17.60) at **2 forwards instead of 3**. Confirms region-routing is deployable with
a single foreground model + run14 bg.

## Deployment options (USER DECISION — science is done):
- **2-forward**: run28 (foreground) + run14 (bg), mask-composite. garL1 16.24 / skin 17.70 / CLOUD 3.46. Cheapest good option.
- **3-forward**: run20(gar) + run27(skin) + run14(bg). garL1 15.10 / skin 17.60 / CLOUD 3.43. Best garment, +1 forward.
- **single region-routed LoRA**: fold foreground into one adapter, train foreground-region loss, composite bg from run14 at deploy (2-forward, but one trained model). Cleanest to ship; needs the two-forward LoRA build.
