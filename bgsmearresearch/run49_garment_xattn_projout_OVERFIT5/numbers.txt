run49 — garment cross-attn at proj_out, Phase A (xattn only, LoRA+v6 frozen), 5-id OVERFIT, 1h
cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1
xattn OFF (==run37)   3.41  -3.70  -1.97  -2.49  -4.05  -4.40  21.61  0.836  15.13  10.97
xattn ON              3.42  -3.94  -5.73  -5.16  -3.90  -4.12  21.42  0.838  15.68  10.78

VERDICT: FAIL. garL1 15.13 -> 15.68 (WORSE). bg/skin roughly held (far_bg -4.40 -> -4.12, skin 10.97 -> 10.78)
but that is meaningless when the garment regressed. Panels: near-identical, marginal stripe change only.

ROOT CAUSE (found later, see run51): the branch was learning at ~2% of nominal rate:
  * LR_WARMUP=1000 (LambdaLR over ALL param groups, lr_lambda(0)=0) and the run only reached 930 steps
    -> warmup NEVER completed, branch LR never realized.
  * gate = sigmoid(-3) = 0.047 multiplies the branch output AND the gradient into out_proj (~20x suppression).
  * objective = run37's inherited single-step flow loss, which barely rewards deployed garment fidelity.
So run49 is NOT a verdict on the architecture. Superseded by run51 (warmup=50, gate=0.5, direct garment loss).

SANITY: xattn OFF reproduces canonical run37 (3.42/-4.40/15.03/10.96) => LoRA truly frozen, delta attributable
to the branch alone.
