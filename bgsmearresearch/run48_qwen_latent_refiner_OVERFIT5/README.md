# run48 — QwenLatentGarmentRefiner on frozen run37 (OVERFIT on 5 ids)

Qwen blocks as a **local latent refiner**, NOT injected into the denoising stream.
(Hidden-state injection was proven invasive: run41/43x/44x all harmed bg/skin via self-attention.)

## Architecture (`qwen_latent_refiner.py`, 690M params)
```
pack(pred_latent, garment_latent, warped_mask) -> (B,3072,132)
in_proj 132->3072 (+learned pos)
2 copied pretrained Qwen blocks (ids 0,1)  [text/pose conditioning DISABLED: learned null_text + null_temb, rope=None]
LayerNorm -> out_proj 3072->64 (ZERO-INIT) -> unpack -> delta
final = pred + warped_mask * delta          # outside garment: latent residual EXACTLY 0
```
- Zero-init head => step 1 `L_gar == base` (identity == run37), verified.
- LayerNorm before the head is REQUIRED: without it delta_absmean jumps to ~1.2 on the first
  non-zero step (latents live in +/-2) and the latent blows up.
- Zero-init costs one dead step (trunk grad = W^T*grad, W=0); trunk gets grad from step 2 (51/58 tensors).

## Results (5 reserved ids; baseline = run37 canonical chain-off, same model state as saved latents)
| | CLOUD | edit_bg | whole_bg | far_bg | PSNR | SSIM | garL1 | skinL1 |
|---|---|---|---|---|---|---|---|---|
| run37 baseline | 3.75 | -3.70 | -5.21 | -6.04 | 21.51 | .832 | 15.21 | 11.54 |
| U-Net refiner (run47, 200k steps) | 3.54 | -3.15 | -5.11 | -6.04 | 25.95 | .931 | 3.14 | 9.99 |
| **Qwen refiner (31k steps)** | 3.59 | -3.29 | -5.14 | **-6.04** | 25.54 | .929 | **3.02** | 10.15 |

Qwen beats U-Net on garment (3.02 vs 3.14) with **6x fewer steps** and a **3x smaller residual**
(delta_absmean 0.016-0.050 vs 0.078-0.102). far_bg bit-identical; firewall `outside_max = 0.00e+00`.

## Garment sensitivity probe (garment latent-L1, lower=better) — the key check
| id | base | correct | WRONG | ZERO | SHUFFLED |
|---|---|---|---|---|---|
| 00006 | .1345 | **.0149** | .1181 | .0960 | .1013 |
| 00008 | .1637 | **.0176** | .1413 | .1330 | .1158 |
| 00013 | .1307 | **.0136** | .1184 | .1050 | .0944 |
| 00017 | .1609 | **.0156** | .1495 | .1178 | .1094 |
| 00034 | .1569 | **.0173** | .1424 | .1048 | .0952 |

Correct garment ~9x better than baseline; wrong/zero/shuffled collapse back to ~baseline.
=> the refiner genuinely CONDITIONS ON THE GARMENT; it is NOT memorizing pred->GT per image.

## CAVEAT
Still an OVERFIT on 5 ids. "Uses the garment" here could be an *index lookup* over 5 memorized
garments rather than garment understanding. Only a held-out test separates these.
**NEXT: train on several hundred ids with these 5 held out; measure surviving garL1 gain + rerun this probe.**

Checkpoint `runs/qwen_refiner_OVERFIT5/refiner.pt` · log `logs/qwen_refiner_OVERFIT5.log`
Repro: `REFINER=qwen QWEN_BLOCK_IDS=0,1 REFINER_LR=5e-5 python train_refiner.py 40000 runs/qwen_refiner_OVERFIT5`
