# run11 — deploy-legal bg-only correction net — ✅ 5-ID overfit PASSES (learnability), generalization TBD

Tiny bg-only correction net, **deploy-legal**, per the plan. NOT a Qwen/V6 module, NOT GT paste.

## Setup
- **Inputs (all deploy-available; NO GT):** raw generated image (soar pred), agnostic image, a **deploy-legal smooth far-bg field** (nearest-fill of the agnostic's REAL bg — grey/masked region excluded — then blur), parse-v3 bg/person masks, edit(grey-hole) mask, distance-to-person map. Audited: `[audit] GT used ONLY as target`.
- **Output:** a small **residual** (`0.5·tanh`) added to the far-bg field, applied ONLY inside generated-bg = (edit ∩ parse_bg). Person/garment/skin untouched.
- **Target (supervision only):** GT background pixels.
- Tiny UNet (32ch, ~3 levels), Adam lr 5e-4, 1500 iters, 5 fixed IDs.

## Result — bg-masked CLOUD (5-ID OVERFIT)
| | bg-masked CLOUD |
|---|---|
| soar raw pred | 8.52 |
| run03 (best trained, generation-time) | 4.84 |
| **run11 corrected** | **0.49** |

Loss L1(gen_bg vs GT) 0.146 → 0.029. Panels (`images/`): smear gone, clean studio grey = GT, person untouched.

## Honest interpretation
- ✅ **Learnability proven:** a deploy-legal bg net CAN reconstruct the near-band studio bg from available context alone (no GT input) and **beats run03** on the 5-ID overfit. Per the plan, this is the "keep going" signal.
- ⚠️ **This is OVERFIT (train == the 5 eval IDs) — memorized, NOT generalization.** 0.49 is train performance. The real test is **scale to full-train + held-out IDs**.
- First key bug fixed: raw-sigmoid output saturated to white (dead gradient); residual-on-a-field is the stable formulation.

## Next
Scale: train the same net on the full dataset (or many IDs) using run03/soar outputs, eval on **held-out** IDs. If it generalizes, it's a real deploy-legal bg fix on top of run03.
