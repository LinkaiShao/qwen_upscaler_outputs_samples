# run01 — bgband loss (v1)

**Run:** `runs/BGBAND_193303`   **Warm-start:** soar (`soar_noise_20260619_170822/final`)   **Budget:** 60 min (launched before the 2h directive)
**Launcher:** `run_bgband.sh`   **Status:** _training — results pending_

## What I did

Added a **bg-band loss on the RAW model output** (`pred_img`, before the grey-hole paste — the paste hides the defect so optimizing it is cheating). Region = **parse_bg ∩ (8–50 px band outside the person silhouette)**; edit-region intersection left OFF (`BGBAND_EDIT_ONLY=0`) because with pure-noise/SOAR the model paints the whole frame, so the smear lives across the full bg ring, mostly outside the garment-centered edit region.

Three terms (`trainlib/losses/background.py`):
1. `LAMBDA_BGBAND_CLOUD=5` — low-frequency (blurred, σ=20) `|luma(pred)−luma(GT)|` → the coherent cloud
2. `LAMBDA_BGBAND_DARK=10` — `relu(luma(GT)−luma(pred))` → signed anti-darkening (only penalize pred darker than real)
3. `LAMBDA_BGBAND_CHROMA=2` — `|sat(pred)−sat(GT)|` → kills green/pink speckle (grey bg → ~0 saturation)

Isolation: existing bg-soup losses (BG_CHROMA / BG_FIELD / BG_SHELL_AB) set to 0 so this lever is measured alone.

Verified live: band = 138,816 px; losses decreasing (bgb_cloud 0.087→0.041, bgb_dark 0.027→0.010, bgb_chr 0.016→0.011 by step 90).

## Result — raw `predict_sample` CLOUD  →  **REGRESSED**

| ID | CLOUD | Δ vs soar | darkening |
|---|---|---|---|
| 00006 | 10.58 | +3.54 | −9.99 |
| 00008 | 9.18 | +1.08 | −8.09 |
| 00013 | 10.21 | +2.11 | −10.63 |
| 00017 | 9.18 | +1.43 | −8.97 |
| 00034 | 12.77 | +2.79 | −12.87 |
| **mean** | **10.38** | **+2.19** | **worse (−10 vs −7.7)** |

vs soar 8.19, if5 target ~3.4. Training-time bgb losses DID go down (bgb_cloud 0.087→0.041) but the **deployed raw bg got worse** — both CLOUD and darkening. Images in `images/` (`<ID>_final_vs_gt.png` = FINAL deployed | GT) show a heavier, dirtier bg cloud than soar even after the paste.

## Interpretation

Classic **train/deploy divergence**: the single-step x0 losses improved in-training while the multi-step deployed rollout regressed. The darkening is a **rollout/trajectory effect** (a learned bias that compounds over the 20-step sampling), which a single-step band loss on x0 can't fix — and here actively worsened it. Consistent with prior findings (edge-darkening = learned bias, not a single-step mixture; single-step losses blind to the deployed halo).

**Confound (be honest):** this run also *removed* the existing bg-soup (BG_CHROMA/FIELD/SHELL_AB → 0) to isolate the new lever, so part of the +2.19 could be from losing those. But darkening got worse specifically, which points at the anti-darkening loss backfiring through the rollout, not just soup removal.

## Next
- Don't iterate weights on this single-step approach — the lever is wrong for a rollout effect.
- Options: (a) apply the band loss on the **rolled-out** SOAR prediction (not single-step x0); (b) treat darkening as a **bias** correction; (c) A/B to disentangle the soup-removal confound (soar+soup vs soar+soup+bgband).
