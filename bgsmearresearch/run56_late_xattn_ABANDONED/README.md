# run55/56 — late-block residual garment cross-attn. VERDICT: ABANDON this interface.

Base run37 frozen. Garment encoder (2 copied Qwen blocks) -> GarmentCrossAttn -> zero-init
out_proj -> gamma*p_g_tok*gate*A_g added to garment C-tokens at late block(s) 50/55/59.
Bridge objective (garment L1 + edge + real wrong-garment sensitivity + residual-outside),
preservation measured vs frozen-base (branch OFF). LoRA+v6 frozen.

## Two independent failures, across many configs
1. GAIN = +0.0000 on EVERY logged step, in EVERY run (55 weak, 56 fp32+SOAR-off, 56 encoder-frozen).
   The branch never once produced a positive garment gain, even overfitting 5 images with exclusive
   write access. wrong==gar on sensitivity steps -> it barely conditions on the garment either.
2. NaN in the GARMENT ENCODER: DEBUG_XATTN_FINITE named `QwenGarmentEncoder.block` fully non-finite
   at step ~145-161. NOT fixed by fp32, NOT by USE_SOAR=0, NOT by freezing the encoder blocks+temb.
   Root cause: running copied pretrained Qwen blocks on garment latents with image_rotary_emb=None +
   a null/learned temb is off-distribution and diverges; even the trained patch_proj (lr 1e-5) drives
   the frozen block attention to overflow.

## Combined with run51 (proj_out): ALL in-loop residual-injection interfaces are dead
- run51 proj_out: gate LEARNED TO SHUT ITSELF (0.5->0.462), deployed garment 15.11 -> 19.88 (worse).
- run55/56 late block: NaN + GAIN=0.

## Conclusion
In-loop garment residual injection into the frozen run37 denoiser does not work here — the frozen
base has no capacity to interpret an externally-added residual, and the copied-Qwen encoder is
numerically unstable in this use. Per the plan: do NOT keep tuning LR/gate/depth on this interface.

The ONLY interface that has ever improved the garment while preserving bg/skin is the OUTPUT-SPACE
latent refiner: run47 (U-Net) and run48 (Qwen blocks as a POST-denoise refiner) -> garL1 15.2 -> 3.0,
far_bg bit-identical. Next real direction = a different interface (K/V-append, or slot-translator),
or scale the refiner to held-out data.
