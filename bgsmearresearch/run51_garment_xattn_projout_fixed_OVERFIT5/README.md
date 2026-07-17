# run51 — garment cross-attn at proj_out (CORRECTED). VERDICT: FAIL -> abandon proj_out.

Fixes vs run49: LR_WARMUP 1000->50 (run49/50 never finished warmup), gate init logit -3 -> 0
(sigmoid 0.047 -> 0.5, which had been suppressing the branch output AND its gradient ~20x),
real wrong-garment (a different ID, re-encoded) instead of shuffled tokens, and a DIRECT
garment objective (L1 + edge + sensitivity + residual-outside) instead of run37's flow loss.
5-id OVERFIT, LoRA+v6 frozen -> the branch had exclusive write access.

## Deployed (canonical render, 5 reserved ids)
| | CLOUD | edit_bg | edit_nb | whole_bg | far_bg | PSNR | SSIM | garL1 | skinL1 |
|---|---|---|---|---|---|---|---|---|---|
| xattn OFF (==run37) | 3.42 | -3.74 | -2.03 | -4.05 | -4.40 | 21.61 | .836 | 15.11 | 10.96 |
| xattn ON | 3.46 | -4.10 | -10.08 | -3.91 | -4.09 | 20.87 | .833 | **19.88** | 10.89 |

## Why it fails
1. The GATE LEARNED TO SHUT ITSELF OFF: 0.5 -> 0.462, out_proj absmean 8.9e-5. Given a direct
   garment loss and exclusive write access, gradient descent decided the branch is harmful.
2. In-loop GAIN ~ +0.0000 at every logged step, even overfitting 5 images.
3. It DOES read the garment: wrong-garment reconstructs GT worse than correct, every step
   (e.g. wrong 0.1102 vs gar 0.1063). Conditioning works; the INTERFACE does not.

## The structural reason
proj_out is Linear(3072->64) applied AFTER all 60 blocks. Whatever the branch writes is squeezed
through one per-token linear into 64 channels with NO downstream capacity to interpret it. The
frozen base never gets to "think about" the garment signal.

## CRITICAL side-finding (affects all later runs)
The single-step loss is BLIND to the real damage: GAIN ~ 0 in-loop, yet deployed garL1 went
15.11 -> 19.88. A residual too small to register at one sigma COMPOUNDS over 20 denoising steps.
=> single-step preservation/improvement metrics cannot be trusted; rollout-based loss (Run 5)
may be required earlier than the priority order assumes.

Also: within a forward, proj_out is per-token, so d_bg/d_far are EXACTLY 0; d_skin/d_ring ~1e-4
come only from fractional p_g_tok on boundary tokens (2x2 patch means).

NEXT: Run1 = late-block residual xattn at blocks 50,55,59 (leaves 9 blocks of self-attention for
the frozen base to interpret the signal). See run52_late_xattn_b505559_OVERFIT5.
