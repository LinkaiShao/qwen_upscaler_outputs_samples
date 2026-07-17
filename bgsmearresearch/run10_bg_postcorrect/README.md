# run10 — bg post-correction — ❌ INVALID (GT leak) / inconclusive

## What was tried
Deterministic pixel-space bg correction on the raw generation: replace generated-background pixels (parse-v3 `==0`) with a color field built from "real visible bg", keep person, feather edges. Script: `bg_postcorrect.py`.

## Why the near-0 result is INVALID
The script sourced the field from **`gt` (ground truth)** — `bg_postcorrect.py:37`, `field = gt[...]` — i.e. it pasted the GT background, which deployment does NOT have. That produced bg-masked CLOUD 0.30, but it is a **leak**, not a result.

## Deploy-faithful test (what deploy actually has = the agnostic)
| source | bg-masked CLOUD |
|---|---|
| GT (leak) | 0.30 — invalid |
| agnostic-v3.2 pixels directly | 49.9 — garbage (agnostic masks the band; agn≈128 vs real bg≈200) |
| agnostic far-bg only, interpolated into band | **7.35** vs soar 8.52 — barely better |

## What this proves (narrow) — and what it does NOT
- **PROVEN:** the *specific* method here fails deploy-legally — naive agnostic/visible-bg interpolation can't fix the smear because the near-silhouette background is **masked/unavailable** in the agnostic, and the GT-sourced version was a leak.
- **NOT proven / still open:** post-processing is **not** globally disproven. Untested deploy-legal options remain — a **learned bg-only inpainter**, a **studio-bg prior**, **local color-field fitting from the far bg**, or a **segmentation-aware harmonizer**. Only this leaked-GT + weak-interpolation approach is ruled out.

## Honest standing
- **run03 (LoRA-trained) = best legitimate result: bg CLOUD 4.84** (from soar 8.19).
- **run09** (V6 train==deploy) learned the objective but is **not competitive**: gated-deploy CLOUD **24.87** (worse than soar, far worse than run03).
- **run10** GT-source result is **invalid**; the deploy-legal interpolation is marginal (7.35).
