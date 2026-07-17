# run21 (PROBE B) — region residual heads on a frozen run14 base

**Run:** `runs/BGBAND21_234151`  **Warm-start:** run14 (frozen)  **Budget:** 2 h (~2100 steps, ~3.4 s/it)  **Launcher:** `run_bgband21.sh`
**Design:** FREEZE run14 base (LoRA + multi-block = bg authority). Add a garment residual head (`to_g`) to V6Heads;
train linear per-region heads (to_g / to_s / to_b) that **refine the frozen `x0_pred`** per region, composed by masks
(`USE_V6_REFINE`). Heads zeroed at init ⇒ output == run14 exactly at step 0. Each head = independent linear map on
**frozen features** → perfect gradient separation. Eval = matched refine-eval (`/tmp/eval_full_refine.py`).

## Eval validity — self-consistency PASSED
`eval_full_refine.py --zero` on run14 (heads forced to 0) reproduced run14 **exactly** (CLOUD 3.34, edit_bg −1.28,
edit_all −2.37, garL1 19.12, skinL1 19.63; per-ID identical). ⇒ the hook + compose + decode plumbing is correct;
run21 numbers below are trustworthy.

## Result — ✗ FAILED the gate on every axis.
| metric | run14 (locked) | run21 | gate | pass? |
|---|---|---|---|---|
| CLOUD | 3.34 | **7.19** | ≤ 3.5 | ✗ |
| edit_bg_dL | −1.28 | **+7.02** | ~−1.28 (bg no worse) | ✗ overshoot bright |
| edit_nonbg_dL | −2.91 | −4.85 | — | ✗ |
| edit_all_dL | −2.37 | −1.21 | — | ~ |
| whole_bg_dL | −10.51 | −8.55 | — | ~ |
| garL1 | 19.12 | **21.63** | ≤ 19.12 or better garment | ✗ worse |
| skinL1 | 19.63 | **20.40** | ≤ 19.63 | ✗ worse |
| edit PSNR/SSIM | 19.88/0.831 | 19.37/0.807 | — | ✗ |

## Verdict: linear residual heads on frozen features are NOT the separation mechanism. Two failures:
1. **Foreground did not improve** — garL1 19.12→21.63, skinL1 19.63→20.40. The linear heads on FROZEN run14 features
   lack the capacity to improve garment/skin; the useful correction is not linearly decodable from the frozen features.
2. **The bg head degraded bg** — despite the base being frozen, the trainable `to_b` residual **overshot** bg to
   **+7.02** (bright), a visible **bright halo ring** around the person (panels). Learned deltas trained across sigmas,
   applied at the final denoise step, over-correct. So trainable region heads do NOT cleanly preserve bg either.

**This matches the pre-registered fallback:** *"if even small region heads cannot preserve bg while improving
foreground, the issue is deeper in the denoising trajectory."* Cheap output-space residual heads are insufficient.

**Panels:** `run21_region_residual_heads/*_pred_final_gt.png` (00034 shows the bright halo overshoot clearly).

## Recommended next probe
The failure is capacity (frozen-linear can't fix foreground) + deploy overshoot (bg head). Next: **design D —
masked gradient routing**: unfreeze the FULL network (full capacity for foreground) but isolate gradients so
foreground losses cannot update bg-route/params and vice versa. Tests whether optimizer-level separation with full
capacity works, without the linear-head capacity ceiling or the residual-head overshoot. (Cleaner-B variant if
desired: FREEZE `to_b` so bg == run14 exactly (no overshoot) + give foreground heads non-linear capacity.)
