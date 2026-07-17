# Loss comparison across runs — exp395 → exp409

Planning document for exp409 (the first run with a region-separated loss structure).
Will be copied into `runs/vton_<exp409_run_name>/final/` at launch time.

## exp395 (original best) — "3-region weighted MSE + image L1"

```python
weight_map = 0.05 + 0.95*M_core + 0.25*M_repair + 1.0*core_boundary

L_latent = mean( ||pred_v - v_target||² * weight_map_packed )
L_img    = mean( |VAE_decode(x0_pred) - person_img| * weight_map_img )

L = L_latent + 0.1 * L_img
```

Key characteristic: **one fused weight map** applied to BOTH the flow MSE in latent and
the image L1 after decode. Everything mixes into the same gradient.

eval.py metrics: main 35.549, PSNR 19.58, SSIM 0.798, pose_use +5.89, pose_spec +6.56

## exp401 / 402 / 403 / 404 (conditioning-order study) — same loss as exp395

No loss change. Those 4 runs tested slot order with identical 3-region weighted loss.

| run | slot order | main |
|---|---|---|
| exp401 | [C, ag, pose, rough, gar] | 35.694 |
| exp402 | [C, gar, ag, pose, rough] | 35.137 |
| exp403 | [C, ag, gar, pose, rough] | 35.693 |
| exp404 | [C, ag, pose, gar, rough] | 35.454 |

## exp405 (edge loss attempt)

```python
L = L_latent + 0.1 * L_img + 0.05 * L_edge_sobel
```

Regressed (main 34.151, −1.54) — Sobel term too aggressive, dragged PSNR down.

## exp406 (w_repair rebalance, killed early at step 503)

Same loss as exp395, but `w_repair: 0.25 → 0.5`. Incomplete; never evaluated.

## exp407 (w_repair 0.10 + agnostic noise)

Same loss structure as exp395 but `w_repair: 0.25 → 0.10`. Agnostic neutralized
inside edit region (core→0, repair→noise). Regressed: main 35.318. Visual fuzz worse
because the low w_repair gave the model no loss pressure in the repair ring.

## exp408 (dimmed agnostic, w_repair restored to 0.25)

Loss identical to exp395. Only the agnostic tensor changed (core→0, repair→0.25×real).
Best SSIM in the study (0.7979, +0.010 over exp401), but main 35.418 (−0.28) and
pose_use dropped. Still uses GT target_mask to define M_core — not ideal for inference.

## **exp409 (new region-separated loss + rough-derived proxy)**

### Conditioning change (no GT leak at inference)

```python
# Proxy core: pixels in M_full where rough visibly differs from agnostic
diff_mag = |rough - agnostic|.mean_over_channels()
diff_in_agn = diff_mag * M_full
max_diff = per_sample_max_of(diff_in_agn)
M_core_proxy   = ((diff_in_agn > 0.4 * max_diff) & (M_full > 0.5))
M_repair_proxy = M_full - M_core_proxy

# Dimmed agnostic
agnostic = (
    agnostic * (1 - M_full)            # outside: real body unchanged
  + 0        * M_core_proxy            # proxy core: zero
  + 0.25     * agnostic * M_repair_proxy   # proxy repair: dimmed
)
```

### Loss (replaces the exp395-style fused weight map)

```python
# Region partitioning for the LOSS (still uses GT target_mask at training time only)
M_full        = agnostic_mask > 0.5
M_core        = target_mask > 0.5
M_repair      = M_full - M_core
M_repair_inner= (max_pool(M_core, 15) & M_repair)    # transition band near core
M_repair_outer= M_repair - M_repair_inner            # far-from-core body area
M_out         = 1 - M_full

# Per-token velocity MSE at the packed sequence level
sq_err = (pred_v - v_target)^2

L_flow_core         = mean_in(sq_err, pack(M_core))
L_flow_repair_inner = mean_in(sq_err, pack(M_repair_inner))
L_flow_repair_outer = mean_in(sq_err, pack(M_repair_outer))
L_flow_out          = mean_in(sq_err, pack(M_out))

# Direct latent-space x0 L1 reconstruction — NEW
x0_pred = C_t - sigma * unpack(pred_v)
diff_l1 = |x0_pred - person|.mean(channels)
L_recon_repair_inner = mean_in(diff_l1, M_repair_inner)
L_recon_repair_outer = mean_in(diff_l1, M_repair_outer)

# Composite latent loss (user-specified weights)
L_latent = L_flow_core
         + 0.25 * L_flow_repair_inner
         + 0.10 * L_recon_repair_inner
         + 0.05 * L_flow_repair_outer
         + 0.25 * L_recon_repair_outer
         + 0.05 * L_flow_out

# Image-space L1 preserved with original 3-region map (for the rough VAE-decode check)
L_img = mean( |VAE_decode(x0_pred) - person_img| * old_weight_map_img )

# Total
L = L_latent + 0.1 * L_img
```

## Structural diff

| Aspect | exp395 / exp408 | exp409 |
|---|---|---|
| Region partitioning | 3 regions (outside / core / repair) with fused weight map | 5 regions (out / core / repair_inner / repair_outer, boundary via L_img) |
| Repair ring | Single zone, single weight | Split into inner (near-core) and outer (far-from-core), different flow/recon weights |
| Flow MSE form | One scalar on the whole token budget | Four scalars, each a region-conditional mean, each normalized by its own token count |
| x0 reconstruction | Only via VAE-decoded pixel L1 | Direct latent-space L1 added on the repair ring, plus the existing image L1 |
| Keep signal | Implicit via low weight outside | Explicit direct L1 saying "repair should match person" |
| Small regions | Drowned by large-region pixel count | Each region normalized by its own pixel count (tiny transitions not drowned) |
| Agnostic input | Real agnostic with grey blob inside edit region | Proxy-core zeroed, proxy-repair dimmed 0.25×, derived from rough (no GT leak) |
| Target_mask at inference | Not needed for inference | Not needed — proxy computed from rough |

## Semantic intent per term in exp409

| Term | Weight | Says to model |
|---|---|---|
| `L_flow_core` | 1.00 | Generate the garment here — full flow matching signal |
| `0.25 * L_flow_repair_inner` | 0.25 | Transition zone near garment edge — generate but less aggressively |
| `0.10 * L_recon_repair_inner` | 0.10 | Stay close to real body here — light direct pull |
| `0.05 * L_flow_repair_outer` | 0.05 | Far from garment — barely any generation pressure |
| `0.25 * L_recon_repair_outer` | 0.25 | Strong keep signal — this area MUST match the real body latent |
| `0.05 * L_flow_out` | 0.05 | Background should not change much |
| `0.1  * L_img` | 0.10 | Whole image should look right after VAE decode |

## Hypotheses being tested in exp409

1. **Splitting repair into inner/outer should reduce sleeve-fuzz artifacts.** Previous
   runs treated the whole ring uniformly at weight 0.25. That gave the model enough
   freedom to fill the far-from-core pixels with noisy textures without penalty.
   With inner getting 0.25/0.10 (flow + light recon) and outer getting 0.05/0.25
   (weak flow + strong recon), the far body region is pulled hard toward the real
   person latent, so fuzz there gets aggressively penalized.

2. **Direct latent-space x0 L1 gives a cleaner gradient than image L1.**
   Image L1 requires VAE decode which is expensive and stochastic at high sigma.
   Latent x0 L1 operates directly in the model's output space, so gradients are
   local, fast, and sigma-stable. Expected to improve training efficiency.

3. **Rough-derived proxy core removes a GT-leak path.** Previous runs used
   target_mask directly for the agnostic modification, which at inference would
   require the evaluator to pass GT info to the model. exp409 derives the core
   location from |rough − agnostic|, which is a noisy but GT-free signal.
   Expected side effect: some samples may have a slightly wrong proxy core (because
   rough doesn't always match the GT garment silhouette exactly), so model sees a
   slightly different agnostic than at "perfect" neutralization. Hopefully robust.

## Open questions / risks

1. The `0.25 * L_recon_repair_outer` term is a DIRECT latent L1 against person. For
   samples where person and agnostic already differ (non-agnostic-mask changes like
   hair, shadows), this could pull x0_pred toward a slightly wrong target. Shouldn't
   matter because M_repair_outer is confined to the agnostic mask.

2. The inner/outer dilation kernel is 15 px (≈7 px radius) at the 128×96 latent
   resolution, which corresponds to ~56 px at image resolution. That's a fairly
   wide transition band — might be tunable.

3. If exp409 regresses, the next move is probably to split the conditioning and
   loss changes into two separate runs so we can isolate which is responsible.

## Score to beat

exp401: main 35.694, PSNR 19.93, SSIM 0.788, pose_use +4.98, pose_spec +4.94.

exp395: main 35.549, PSNR 19.58, SSIM 0.798, pose_use +5.89, pose_spec +6.56 (strongest pose).
