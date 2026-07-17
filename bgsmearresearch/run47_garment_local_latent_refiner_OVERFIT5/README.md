# run47 — Garment-local LATENT refiner on frozen run37  (OVERFIT on 5 ids)

**The first garment intervention that improves the garment without harming bg/skin.**

## Architecture (`garment_refiner.py`, 3.39M params)
```
delta = zero_init_unet(pred_latent, garment_latent, warped_mask)
final = pred_latent + warped_mask * delta        # outside garment: latent residual EXACTLY 0
```
- base = **frozen run37**; its canonical deployed pred latent is the input (no denoising-trajectory change)
- zero-init output conv → at step 0 `final == pred` (identity == run37), then it learns a controlled residual
- **no Qwen hidden-state injection** — this is the spatial firewall the residualized-chain runs (run41/43x/44x) could not provide

## Losses (`train_refiner.py`)
masked garment L1 · masked edge/gradient L1 · wrong-garment sensitivity hinge · boundary-ring |delta| (no seam) · TV.
Outside-garment zero is **by construction**, not a loss.

## Result (5 reserved ids; baseline = run37 canonical, same model state as the saved latents)
| | CLOUD | edit_bg | whole_bg | far_bg | PSNR | SSIM | garL1 | skinL1 |
|---|---|---|---|---|---|---|---|---|
| run37 baseline | 3.75 | -3.70 | -5.21 | -6.04 | 21.51 | 0.832 | 15.21 | 11.54 |
| **refiner** | **3.54** | **-3.15** | **-5.11** | **-6.04** | **25.95** | **0.931** | **3.14** | **9.99** |

**Every metric improves or holds. far_bg bit-identical.** Latent firewall verified: `outside_max = 0.00e+00` on all 5.
Pixel-space composite is unnecessary (VAE decode bleed only ~0.15-0.21/255; clipping it slightly *hurts*).

## CAVEAT — this is MEMORIZATION, not generalization
200k steps on 5 images drove garment latent-L1 0.129 -> 0.005. It proves the **mechanism** (garment is improvable
behind a hard spatial firewall while bg/skin stay safe), and nothing about held-out performance.
**Next:** train on a few hundred ids, hold out these 5, measure surviving garL1 gain.

## Files
`ALL_pred.png` (all 5 stacked) · `{id}_pred.png` = [run37 baseline | refiner | GT] · `numbers.txt`
Checkpoint: `runs/garment_refiner_OVERFIT5/refiner.pt` · log: `logs/refiner_OVERFIT5.log`
Inputs: `runs/BGBAND37_012549/deploy_imgs/{sid}_pred_latent.pt` + `my_vton_cache/latents/{sid}_{garment_latent,warped_mask_128,person_latent}.pt`

## GOTCHA
`render_panels.sh` runs `train.py` with `MAX_STEPS=6`, so the LoRA drifts between val passes and `deploy_imgs`
is rewritten. The baseline rawpreds MUST come from the same val pass as the saved `*_pred_latent.pt`, else bg
metrics shift ~1.5 and you will wrongly conclude the refiner darkened the background.
