# v6b — Qwen-Image-Edit-2511 VTON with specialized heads + parse routing

Run: `vton_20260424_152039` (1-hour training, 5 VITON-HD samples, paired)

## What's new in v6b

Architecture additions on top of the tight-agnostic baseline:

1. **Specialized heads** (`V6Heads` module) — tokenwise Linear projections from Qwen's pre-proj features (3072-dim/token, captured via forward_hook on `norm_out`):
   - `to_s: Linear(3072, 64)` → packed skin residual δ_s
   - `to_b: Linear(3072, 64)` → packed bg residual δ_b
   - `to_route: Linear(3072, 16)` → 4-class routing logits (4 classes × 2×2 patch)
   - δ heads are zero-init so `x_repair = x_src` at the start
   - Total: ~442k params, tiny compared to the main LoRA (47M)

2. **Gradient-isolated losses** — each head's gradient flows through its own parameters + shared transformer features only:
   - `L_flow_main` (unchanged from baseline): region-weighted flow MSE → trains main transformer (LoRA)
   - `L_repair_v6 = W_S · L1(δ_s, person−agn)|_M_s + W_B · L1(δ_b, person−agn)|_M_b∪M_other + W_KEEP · ||δ||₁|_outside` → trains δ heads only
   - `L_route_v6 = CE(route_logits, Y_route)` where `Y_route ∈ {garment, skin, bg, keep}` → trains routing head only

3. **Parse-based routing labels** (training only, never seen by model):
   - `M_g = warped_mask` (inference-reproducible)
   - `M_s = ring ∩ parse_skin`, `M_b = ring ∩ parse_bg`, `M_other = ring fallback`
   - `M_k = 1 − dilate(warped, 7 latent px)`
   - Parse used only to subdivide the repair ring, never to define garment geometry

4. **V6_ZERO_G_CORE** — zero agnostic inside `erode(warped_mask, 2)` after USE_AGNOSTIC_INPAINT. Removes the "torso template" the model was learning to denoise.

5. **Composition at inference** (warped-only, no parse needed):
   ```
   x_final = M_g · C_lat + ring · (agn + δ) + M_k · agn
   ```

## Config (matches baseline 133455 + v6)

```
USE_PURE_NOISE=1  USE_AGNOSTIC_INPAINT=1  SLOT_ORDER="agnostic,garment,silhouette"
EARLY_BLOCK_CUTOFF=-1  LR_EARLY_MULT=1.0                    # uniform LR, no 10× early-block boost
AGNOSTIC_FILE=_tight_agnostic_latent.pt
AGNOSTIC_MASK_FILE=_tight_agnostic_mask_latent.pt
IMG_LOSS_WEIGHT=0.15  LAMBDA_ANTISLUDGE=0  LAMBDA_TV=0
USE_V6=1  V6_ZERO_G_CORE=1  V6_R_OUT=7  V6_R_IN=2
W_FLOW_CORE=1.0  W_FLOW_REPAIR=0.1  W_FLOW_UB=0.3  W_FLOW_KEEP=0.05
W_V6_DELTA_S=0.5  W_V6_DELTA_B=0.5  W_V6_DELTA_KEEP=1.0  W_V6_UB=2.0
LAMBDA_V6_REPAIR=1.0  LAMBDA_V6_ROUTE=0.5
TIME_BUDGET=3600
```

## No inference cheating

At training we use target parse to build M_s, M_b masks for the repair-head CE/L1 loss — supervision only, never fed to the model.

At inference the model sees exactly:
- `C_t` (noise init)
- `agnostic_latent` (tight, with V6_ZERO_G_CORE applied — uses eroded warped, no parse)
- `garment_latent`
- `silhouette slot` (warped_mask)
- text/pose embeddings

Composition at inference uses `warped_mask` only. No parse at test time. See `diag/v6_inputs/` for proof of what the model actually sees.

## Files

- `code/` — training + inference + diagnostic scripts
- `results/panel/` — per-sample GT, pred, heatmap, panel.png
- `results/gt_vs_pred_v6b.png` — 5-sample GT | pred strip
- `results/inference.py` — the saved inference script for this run
- `results/v6_heads.pt` — trained V6Heads state_dict (442k params)
- `results/config.json`, `train.log`, `loss_comparison.md` — run provenance
- `diag/v6_inputs/` — visualized inputs the model actually receives (col 4 is model input)
- `diag/parse_v6/` — 4-class routing visualization (M_g / M_s / M_b / M_k)
