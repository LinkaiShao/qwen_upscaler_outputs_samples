# VTON Training Regime, GAN, Recent Trainings & Diagnostics — Detailed Report
_Compiled 2026-06-16. Covers: (1) the original v01 model, (2) inference/deployment, (3) the
garment-conditioned discriminator, (4) the GAN rollout branch, (5) the band-loss experiments,
(6) the full diagnostic investigation into why the boundary/background defect won't train out._

---

## 0. Cast of artifacts (paths)

| Artifact | Path |
|---|---|
| Original model (v01) | `runs/vton_20260608_175115/final/` |
| v01 launcher (recipe) | `versions/v01_clean_losses/run.sh` |
| Refactored training code | `trainlib/` (entry `train.py` → `trainlib.run.main`) |
| Inference (one shared path) | `runs/vton_20260608_175115/final/inference.py` (`predict_sample`) |
| Deployed-paste pipeline | `runs/vton_20260608_175115/make_pred.py`, `precompute/generate_panel.py` |
| Edit-region definition | `editmask.py` (`grey_hole`) |
| Discriminator (critic) | `discriminator.py` (`GarmentDiscriminator`), weights `runs/discriminator_v01/D_best.pt` |
| Critic trainer | `train_discriminator.py`; pair producer `pred_producer.py` |
| GAN rollout + loss | `trainlib/gan_rollout.py`, `trainlib/gan_loss.py` |
| Band loss | `trainlib/forward.py` (`USE_BAND_LOSS`) |
| Recent runs | `runs/cont_ganband_2ep/` (pixel band+GAN), `runs/latent_band_test/` (latent band) |
| Continuation launchers | `run_cont_ganband.sh`, `run_latent_band_test.sh` |

---

## 1. Original model — v01 (`vton_20260608_175115`)

### 1.1 Base & adapters
- **Base:** Qwen-Image-Edit-2511 (flow-matching, v-prediction). VAE 8× downsample: image 1024×768 ↔ latent 16×128×96.
- **Trainable adapters (three):**
  1. **LoRA** — rank 64 / α 128, targets `to_k, to_q, to_v, to_out.0`, on **all 60 transformer blocks**. **94,371,840 params.**
  2. **v6 heads** — `to_s + to_b + to_route`, **442,512 params**, lr 3.0e-4. The "paint-vs-keep" routing + repair mechanism (`USE_V6=1`, `V6_R_IN=2`, `V6_R_OUT=7`). Applied via a forward hook on `transformer.norm_out`. **Essential — disabling v6 destroys the model.**
  3. **Multi-block GarmentChain injection** — **FULL Qwen blocks** at 8 sites (blocks **12,20,28,36,44,50,55,59**), **2.74B params**, lr 5e-5. (`USE_MULTI_BLOCK_INJ=1 USE_MULTI_BLOCK_FULL=1`.)

### 1.2 Optimizer / schedule (runtime, from train.log header)
- **Per-block LR:** blocks 0–28 at **1.5e-4**, blocks 29–59 at **3.0e-5** (base 3e-5 × `LR_EARLY_MULT=5.0` for early blocks, `EARLY_BLOCK_CUTOFF=28`). 45.6M early / 48.8M late params.
  - _Note: `run.sh` exports `LR=2e-5`; the saved `config.json` and runtime header show base **3e-5**. The runtime header is authoritative for what produced 175115._
- **Schedule:** cosine, warmup 1000, total 23000, min_frac 0.1. AdamW, wd 0.01, β(0.9,0.999), grad clip 1.0.
- **Batch:** size 1, grad_accum 1. **Dataset:** VITON-HD `test` split, full-train, **11,647 samples** → ~11,647 steps/epoch.
- **Trained:** fresh (no warm-start), **2 epochs ≈ 23,000 steps, ~34 h**. Seed 42, gradient checkpointing on. Transformer on cuda:0, VAE on cuda:1.

### 1.3 Conditioning / input construction
- **Slots:** `slot_order = [agnostic, garment, silhouette]` (the noised latent C is slot 0; these are the conditioning slots).
- `USE_PURE_NOISE=1` (C starts from pure noise in masked region), `USE_AGNOSTIC_INPAINT=1`, raw agnostic+rough (no neutralization).
- **Soft masks:** `garment_prior` (from `warped_mask_128`) + `uncertain_band` (thin ring at the true garment contour = dilate(target_mask) − erode(target_mask)) + `keep`.
- **σ sampling:** Beta(1,1) = **uniform** on [0,1]. `USE_SIGMA_SCHED=1` (`SIGMA_SCHED_LO=0.6`, `HI=1.4`) scales conditioning slots by σ (structure slots ↑ at high σ, detail slots ↑ at low σ).
- `DILATE_M_FULL=3` (edit-region dilation in latent cells).

### 1.4 Loss (the "clean losses" recipe — verified active terms only)
Authoritative banner: `loss = L_flow(w_keep=0.05, w_gar=1.0, w_ub=0.3) + 0.5·L_img + 1.0·L_recon_ub + 0.0·L_antisludge + 0.0·L_tv`, plus v6 + tv_edge.

| Term | Weight | What it is |
|---|---|---|
| **L_flow** | 1.0 (w_keep 0.05 / w_gar 1.0 / w_ub 0.3) | flow-matching v-pred MSE; the backbone |
| **L_img** | **0.5**, region-split (`USE_V6_IMG_STAGE=1`) | image-space L1 on VAE-decoded x0. Region weights: garment `W_IMG_V6_G=4.0` (G_ID 4.0 @ low σ, G_STRUCT 0.3 @ high σ), boundary `W_IMG_V6_UB=1.2` (UB_LO 0.3 / UB_HI 1.2), skin `W_IMG_V6_S=3.0`, bg `W_IMG_V6_B=5.0`, other 0.7, keep 0.3 |
| **L_recon_ub** | **1.0** (`LAMBDA_RECON`) | direct latent L1 `|x0_pred − person|` in the **uncertain band** (garment contour ring). "ub" = uncertain band, NOT upper band |
| **L_v6_repair** | 0.8 (`LAMBDA_V6_REPAIR`) | v6 repair head |
| **L_v6_route** | 0.5 (`LAMBDA_V6_ROUTE`) | v6 routing head |
| **L_tv_edge** | 0.4 (`LAMBDA_TV_EDGE`) | TV smoothness on x0_pred at silhouette boundary |
| **antisludge, tv** | **0.0** | explicitly OFF |
| ~30 other terms (chroma/ab/halo family/percep/bg_chroma/...) | 0.0 | OFF (the "overbuild" was disabled) |

`x0_pred = C_t − σ·v_pred_lat` (the predicted clean latent). L_recon_ub and the band losses below all key off this `x0_pred`.

---

## 2. Inference / deployment pipeline

- **Sampler:** `predict_sample` — 20-step FlowMatch Euler. With `RAW_FULL_PRED=1` it returns the **raw full-frame generation** latent `C_lat` (pre-paste).
- **Decode + paste (the deployed image):**
  1. VAE-decode `C_lat` → raw RGB.
  2. **Edit region = the grey hole of `agnostic-v3.2.jpg`** (`editmask.grey_hole`): detect grey [128,128,128]/low-chroma → keep **largest blob only** → fill holes → **dilate 8 px**. (This replaced `agnostic_mask_latent`, whose stray single-cell specks dilated into face/crotch blobs.)
  3. **Hard binary paste:** `m·raw + (1−m)·agnostic_jpg` (no soft alpha blend — soft blend caused the grey band).
- **`make_pred.py`** is the canonical implementation; `pred_producer.py` mass-produces deployed `gt.png`/`pred.png` pairs.

---

## 3. The discriminator (critic) — `discriminator_v01`

### 3.1 Architecture (`GarmentDiscriminator`, `discriminator.py`, 2.75M params)
- **Judges the entire composite**, not just the edit region — the kept real skin/body **outside** the grey hole anchors judgment of the generated skin **inside** it.
- Inputs: full image (3ch) **+ edit-mask channel (1ch)** so it knows generated-vs-kept; garment branch on the **cloth image** (3ch).
- Image patches **cross-attend** to garment features. Heads: **patch head** (47×35 over the whole frame) + **global head**. Spectral-norm throughout.

### 3.2 Training (`train_discriminator.py`)
- Data: **4,921 matched pairs** `gt.png` (real garment) vs `pred.png` (generated), both deployed via the same grey-hole paste → differ **only** inside the edit region. Cloth from VITON-HD `cloth/`, edit mask from `grey_hole`.
- bs **256**, 16-worker DataLoader, hinge loss on both heads, 87/13 train/val split, **4,000 steps** (~2 h, GPU0).
- **Result:** margin climbed monotonically (through transient spectral-norm dips) to **+7.16 patch / +5.92 global** at step 4000. Held-out: **prefers real in 100% of 639 val pairs** (both heads), mean margin +5.92, worst-case still positive. `D_best.pt`.
- Caveat established later: its signal is **whole-image / global** (large receptive field) — a strong "contains a generated garment" detector more than a precise per-pixel locator.

---

## 4. The GAN rollout branch — `gan_rollout.py` + `gan_loss.py`

**Purpose:** supervise the **deployed multi-step output** (not the single-step x0_pred), using the frozen critic.

### 4.1 The rollout (one shared denoising path)
- Reuses the **real `predict_sample`** with a new `grad_start_step` param (the only change to `inference.py`): per-step `set_grad_enabled(i ≥ grad_start)`. Default `None` → byte-identical inference; rollout passes **`grad_start_step=12`** → prefix steps 0–11 detached, **tail 12–19 differentiable** (8 grad steps).
- Decode `C_lat` → grey-hole hard paste → `pred_final`. GT composite built identically from the real person latent.
- **Validated:** faithfulness (rollout no-grad vs `make_pred` pred.png = **0.10/255**), differentiable tail (**480/480 LoRA params receive gradient**), memory **87.7 GB peak** with gradient checkpointing (fits the 95 GB GPU0; VAE/critic on cuda:1).

### 4.2 The loss (`gan_loss.py`, spec §3,4,6)
- Paired **soft-margin gap**: `gap = D(gt).detach() − D(pred)`; `gan_map = softplus((gap−0.5)/0.5)·0.5` (stops pushing once near GT — safer than −D(pred).mean()).
- Pool inside the generated region at critic resolution: `L_gan = 0.5·mean + 0.5·worst-25%`.
- **`lambda_gan` calibrated from gradient magnitudes** so the critic contributes ~10% of the generator gradient (raw GAN grad ≈ 28–30× base → λ ≈ 0.0026), ramped 0→full over 250 updates.
- Critic **frozen** (§7: no joint training initially).

### 4.3 Cadence (wired into `run.py`)
- 1 rollout update per **`GAN_EVERY=4`** base updates; rollout has its own backward + optimizer step; LR scheduler driven by base steps only; OOM on a rollout is caught and that update is skipped (GPU1 shared with a game).

---

## 5. The band loss (boundary-background defect) — two versions

**Target defect:** the revealed background just outside the person silhouette, inside the edit region, where GT is true background — the model renders it slightly **warm/off-white** with **low-frequency blotchiness**.

### 5.1 Pixel band (first version — `cont_ganband_2ep`)
On the **VAE-decoded RGB images** (1024×768). Band = `(0 < d_out ≤ 12 px from person) & edit & GT-bg`, soft weight `exp(−d/4)`. Four terms:
`L_band = 1.0·Charbonnier(pred−gt) + 0.25·Lab-mean-color + 0.25·masked-low-freq-blotch + 0.10·one-sided-variance`.
High-pass texture kept **diagnostic-only** (the defect is low-freq, not high-freq grain).

### 5.2 Latent band (second version — `latent_band_test`, replaced the pixel band)
On the **latent** (128×96): `Charbonnier(x0_pred − person_latent)` over the boundary band (`parse_bg & edit & dist ≤ BAND_LAT_MAXDIST=8 cells`, weight `exp(−d/BAND_LAT_SCALE=2)`). Direct latent target, no VAE decode. ~1,000 band cells/sample.

### 5.3 σ-gate & σ-bias (both versions)
- **σ-gate:** `g_band = max(gaussian peak@0.30, 0.25 if σ<0.20)` — concentrate where the defect forms.
- **σ-bias:** half of training samples drawn `Uniform(0.20, 0.42)` (the formation window), half from the normal Beta(1,1).

---

## 6. Recent trainings

### 6.1 `cont_ganband_2ep` — v01 continuation + pixel band + GAN rollout (STOPPED ~step 4,000)
- Resumed from v01 (LoRA + v6 + GarmentChain, `missing=0 unexpected=0`). `LAMBDA_BAND=0.5`, GAN rollout `GAN_EVERY=4`.
- **Speed: ~19 s/step** — the rollout (20-step sample through the 2.74B GarmentChain, every 4 steps, ~55 s each) was **~70% of wall-clock**.
- **Pixel band stayed flat:** 0.133 → 0.146 over 4k steps (window-averaged). Did not drive the defect down.
- **GAN dynamic:** the generator **closed and overshot** the frozen-critic gap in ~500 rollout updates (gap +5.4 → negative; D(pred) > D(gt)); `L_gan` collapsed 6.96 → ~0.1. Per spec §7, frozen-critic signal consumed.
- Stopped to pivot to a clean isolation test.

### 6.2 `latent_band_test` — v01 base + latent band, GAN OFF (STOPPED ~step 8,970)
- Same v01 recipe & base; **GAN fully off** (no rollout), pixel band killed, **latent band `LAMBDA_BAND=20`** (effective ~12× flow with the gate). ~5.5 s/step.
- **Latent band also stayed flat:** 0.072 → 0.069 over 8,500 steps despite the dominant weight. Val img fine (0.519→0.488), flow stable — the model trains; the band term specifically is stuck.
- Checkpoints saved: `best_val` (=step 4000), `step_5800`, `latest` (=step 8000).

---

## 7. Diagnostics

### 7.1 σ-trajectory (when the bg garbage forms)
Per-step decode of one 20-step sample (00034): the revealed-background region is noise until ~step 11, then **commits at steps 14–17 (σ ≈ 0.42 → 0.20)**. Residual texture std **plateaus at ~13** (vs ~2–3 for real flat white) by step 16–17 and the final low-noise steps **never clean it up**. → the model bakes in the off-white + blotch mid-sampling and can't walk it back.

### 7.2 VAE-floor test (is the ring a decode artifact?) — **NO**
Band metrics of **decode(GT latent) vs the real original image** (the VAE round-trip floor) vs **model pred vs real**:

| | VAE floor: pixel / color / blotch | Model: pixel / color / blotch |
|---|---|---|
| 00006 | 0.014 / 0.12 / **0.0014** | 0.34 / 4.68 / **0.124** |
| 00034 | 0.010 / 0.07 / **0.0013** | 0.16 / 0.89 / **0.055** |

The VAE reconstructs the GT background **near-perfectly** (blotch ~0.0015); the model is **40–100× worse**. → **the ring is a LATENT generation error, not a decode artifact.** Backgrounds are exactly what an 8× VAE reconstructs best. ~50–100× headroom to the floor.

### 7.3 Loss-space audit (pixel vs latent)
- Pixel band = on VAE-decoded images; gradient reaches the latent only **indirectly** through the (frozen) VAE-decoder Jacobian → weak teacher for a latent error. Motivated the latent band.
- Latent band = direct `x0_pred` vs GT latent; but it **also stayed flat** (§6.2) → the problem is not the pixel-vs-latent framing.

### 7.4 Check B — deployed 20-step output across checkpoints — **no sustained improvement**
Deployed band metrics (vs real original) for ckpts 0 / 4k / 5.8k / 8k:

| ckpt | band_pixel | band_color | band_blotch | D(pred) |
|---|---|---|---|---|
| 0 (v01) | 0.282 | 4.01 | 0.115 | −2.29 |
| 4000 | 0.263 | 3.58 | 0.106 | −0.39 |
| 5800 | 0.267 | 3.34 | 0.103 | −1.82 |
| 8000 | **0.295** | **4.52** | **0.128** | +0.88 |

Transient dip through 5.8k then **regresses past baseline by 8k**. (Rising D(pred) is the *frozen* critic being fooled as the model drifts, not a real fix.)

### 7.5 Gradient-conflict probe (σ=0.30, fixed noise) — **conflict REFUTED**
`||g_band|| = 3.55`, `||g_flow|| = 0.085`, `||g_recon|| = 0.035` → band gradient **~42× larger** than flow.
`cos(band,flow) = +0.42`, `cos(band,recon) = +0.55`, `cos(flow,recon) = +0.53` → **positively aligned.** The band loss is not fighting the flow objective; it's a large, aligned gradient that doesn't move the output.

### 7.6 Check C — one-sample fixed-noise overfit — **no floor, no capacity limit**
Train ONE fixed (sample, noise, σ=0.30) instance, LR 2e-4 (note: cosine warmup makes the scheduler LR 0 at step 0 — pin it):

```
step 1: band=0.0705   step 40: 0.0368   step 80: 0.0195   step 160: 0.0095   step 220: 0.0076
```

Band drops **0.070 → 0.0076 (9.3×)** and keeps falling toward the ~0.001 floor. → the LoRA **has ample capacity** to fix the boundary when there's a single unambiguous target.

### 7.7 Synthesis — why the band loss can't train out the defect
Putting §7.5 + §7.6 together (large aligned gradient + fits a single instance + can't move the full-data average) is the signature of being **already at the conditional-mean optimum**:
- The one-step loss penalizes `x0_pred = C_t − σ·v_pred` at σ≈0.3, **averaged over noise**. Its Bayes-optimal value is `E[x0 | C_t, cond]`, whose irreducible error is the **conditional variance** — many clean backgrounds are compatible with the same noised state, so the optimal one-step predictor is a blurry average. That floor is **~0.07**; no weight beats it in expectation.
- On a single fixed instance there's no ambiguity → one target → overfits to ~0.
- The **visible defect lives in the multi-step, free-running (exposure-biased) deployed trajectory** (§7.1, §7.4) — a different object the one-step loss never touches.

**Conclusion:** a band loss on the **one-step prediction** — pixel or latent — is structurally incapable of fixing this defect. The structurally-correct fix is to **supervise the deployed endpoint** (`pred_final` from the rollout): a deterministic per-sample target with no per-step conditional-mean averaging. That is the rollout's domain (the GAN already operates there; a deterministic band loss on `pred_final` is the direct next experiment).

---

## 8. Status & next step
- All training/diagnostic runs **stopped**; GPU free.
- Diagnostic hooks left in `forward.py`/`run.py` behind flags: `FIXED_SIGMA`, `RETURN_LOSS_TENSORS`, `RUN_DIAGNOSTIC` / `DIAG_LR` / `DIAG_OVERFIT_STEPS`.
- GAN rollout + critic intact and validated.
- **Recommended next experiment:** endpoint band loss computed on the rollout's `pred_final` (vs GT composite), replacing/augmenting the one-step band loss.
