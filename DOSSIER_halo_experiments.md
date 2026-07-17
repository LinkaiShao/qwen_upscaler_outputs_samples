# VTON Halo-Elimination — Experiment Dossier (2026-06-24)

## The problem
A **dark edge-ring halo** at the person/garment silhouette, visible **only in 20-step
from-noise deployment** — invisible to the training forward. Goal: kill it without
dirtying the rest of the background.

## The metric (validated)
**Deployed ring-contrast** = `ring(0–8px) dL − surround(20–50px) dL`, averaged over the 5
test IDs (dL = generated − GT luminance on the inner background). More negative = more
visible halo. Validated: reproduces the eye, and independently reproduces the known
110824 < 143106 regression. Tool: `/tmp/bg_quality.py` / `/tmp/contrast_measure.py`
(run on `<run>/final` via the V01 inference pipeline, same seeds for all).

## Results (deployed ring-contrast; closer to 0 = better)

| run | design | contrast | verdict |
|---|---|---|---|
| **110824** (vton_20260623_110824) | gem + bg-color soup, **45-min** snapshot | **−1.35** | **best so far** |
| 143106 (vton_20260623_143106) | 110824 trained to 3.75h | −1.48 | regressed |
| **195835** (vton_20260623_195835) | region-separated + harmony, bg→white, no soup | −2.01 | cleanest bg; **undertrained, still improving** |
| exp1 / 235951 (vton_20260623_235951) | + punish entire latent | −2.91 | **worst** |
| exp2 / 050243 (vton_20260624_050243) | + multistep full-inference rollout | −2.49 | **backfired** |

Per-image ring-contrast (110824 vs 195835): 110824 wins 4/5; **00008 (Adidas striped
shirt) is the universal worst case** for every variant.

Per-image (110824 | exp1 | exp2):
```
  00006  -0.79 | -1.25 |  -1.77
  00008  -4.52 | -10.35|  -9.39    <- worst case
  00013  +0.57 | +0.40 |  +0.97    (clean, no ring)
  00017  -1.78 | -2.53 |  -1.85
  00034  -0.20 | -0.85 |  -0.41
  MEAN   -1.35 | -2.91 |  -2.49
```

## What worked
- **Region separation (bifurcation)** → the **cleanest background field** of any run
  (clean white; far-bg dL +0.72 vs 110824's −0.83). The keeper architecture.
- **gem** (concentrated garment-edge ring → GT) → directly tightens the halo ring;
  it is *why* 110824's ring is tight.
- **The metric + 45-min snapshot tracking** → the methodology that let us judge by the
  deployed halo instead of the misleading val_img / thumbnails.

## What didn't
- **Full-latent supervision (exp1)** → worse edge.
- **Multistep rollout punishment (exp2)** → **globally darkened** the whole bg (far dL
  −4.16); backfired.
- **brs / model's-own-bg self-target** → abandoned (the model's generated bg is garbage;
  the reference must come from the real agnostic/GT).
- **Loss soup** (stacking chroma/field/shell/brs) → confirmed: adding losses didn't help.

## Key insights
1. **val_img is misleading** — improves while the deployed halo regresses (train/deploy
   divergence).
2. **The halo is a multi-step artifact** — it compounds over the 20 inference steps;
   single-step losses (gem included) are partly blind to it. 110824 is a 45-min
   sweet-spot snapshot before continued training regrows it (gem stays flat in training
   while the deployed halo regrows).
3. **110824's "good" is partly cheap** — its bg is uniformly slightly-dark, so the ring
   blends in (low contrast), not a pristine edge. 195835's genuinely clean bg *exposes*
   its residual ring.
4. **Field vs ring are different problems**: `img_b` (diffuse, out→in) cleans the bg
   *field*; `gem` (concentrated, in→out) fixes the *ring*. 195835 fixed the field, left
   the ring; 110824 fixed the ring, hid it under dimness. **The synthesis = both.**

## Current state (2026-06-24 20:25)
- **RUNNING:** `160743` (vton_20260624_160743) — 195835 continued (8h, from 195835/final)
  to test whether more training crosses 110824. Step ~1410, ends ~00:07. Snapshots every
  45 min in `runs/.../snapshots/`.
- **BUILT, HELD (not queued):** `run_region_gem.sh` — base 110824 + full bifurcation +
  harmony + **gem** (the field+ring synthesis). GPU will idle after 160743 finishes until
  manually launched.

## Current/planned losses (run_region_gem — verified from code)

8 active loss terms (everything else in the sum is weight-0 / off):

| loss | Garment | Skin | Background | Other | Boundary |
|---|---|---|---|---|---|
| flow (denoise) | ✓✓ strong | ✓ | ✓ | ✓ | ✓ |
| reconstruction (img) | →GT (×4) | →GT (×3) | →**WHITE** (×5) | →GT (×0.7) | →GT (×1.2) |
| harmony | →GT | →GT | →GT | →GT | (covers) |
| v6 (route + repair δ) | ✓ | ✓ | ✓ | ✓ | — |
| edge-smooth (tve) | — | — | — | — | ✓ |
| recon | — | — | — | — | →GT |
| **gem (NEW)** | — | — | →**edge ring → GT** | — | — |

Notes:
- Every region trained to **GT except background → flat white**; gem pulls the bg edge
  ring back to GT.
- bg loss **region = GT `parse_bg`**, **target = real GT bg** (white sampled from real bg
  outside the hole; gem = GT per-pixel). The v6 route head's *prediction* is NOT used to
  place the loss (`brs` is off) — the head is only *trained* to match GT for inference.
- Background = the busiest region (img→white, harmony→GT, gem→GT, repair δ→GT): one region
  pulled toward white by one loss and GT by three. Watch for conflict there.
- "Other" = hair / pants / unknown (hole − garment − skin − bg). Reconstructed →GT weakly.
- "Boundary" = thin ring on the outer person/bg silhouette (the halo line), heavily but
  thinly supervised. The internal garment/skin/bg *interface* boundary loss (`L_v6_boundary`,
  `W_V6_UB`) is OFF.
- Largest single contributor is actually **`rep`** (v6 repair-head δ, ×0.8), then gem.

## Lineage
soar_noise → 110824 (gem+soup, best) → [region-separation pivot] → 195835 (region+harmony)
→ {exp1 full-latent, exp2 multistep} (both worse) → 160743 (195835 continued, RUNNING)
→ run_region_gem (built, held).

## Update 2026-06-25 — gem + bifurcation overnight results (offline ring-contrast)
```
110824 (baseline)            -1.35   <- STILL BEST
gem early ~step253 (peak)    -1.71   gem helps (beats no-gem bifurcation) but < 110824
195835 bifurc 4h             -2.01
gem final 8h                 -2.37   regressed past peak
160743 bifurc CONVERGED 8h   -3.21   <- training bifurcation longer made it WORSE
```
- gem WORKS (gem-region -1.71 beats no-gem bifurcation -2.01/-3.21) — it tightens the ring like it did for 110824. NOT broken.
- gem-region does NOT beat 110824: 110824 wins partly by MASKING the ring under a uniformly-dim bg (low contrast); the bifurcation's clean bg EXPOSES the residual ring.
- Both designs regress with more training (multi-step train/deploy divergence): gem -1.71->-2.37, bifurc -2.01->-3.21. 195835's "still improving" reversed on convergence.
- VERDICT: nothing this round beat 110824 (-1.35). The deployed halo remains a multi-step problem no single-step loss (gem/brs/region-white) durably fixes; the clean-bg designs trade ring-masking for ring-exposure.

## Update 2026-06-25 (later) — BG RING ISOLATION: first win over 110824
Fix: partition bg into ring(0-8px, owned by gem->GT + tve) and field(>8px, owned by img_b->white);
harmony/recon/img_ub excluded from ALL bg. (BG_RING_ISOLATION=1, GARMENT_EDGE_RING_PX=8, forward.py.)
Deterministic offline ring-contrast (run vton_20260625_112845, seeded eval, 45min val):
```
110824 base                -1.35
iso snap1218 (~step500)     -1.33   tied
iso snap1303 (~step730)     -1.04   <- BEATS 110824 (+0.31). New best. Genuinely tighter ring
                                       (ring -1.72 vs -2.02) at same surround (-0.68), NOT a dim-bg trick.
iso snap1348 (~step960)     -2.46   regressed
iso snap1518 (last)         -1.54
```
- Mechanism (b) the white-vs-GT edge CONFLICT was real -> isolation fixed it -> better peak.
- Mechanism (a) the single/multi-step ROLLOUT GAP still real -> still peaks-then-diverges; isolation can't fix it.
- KEEPER: isolation + early-stop at the peak (seeded validation finds it). Best checkpoint = snap1303 (-1.04).
- Divergence fix still untried: backprop-through-rollout OR post-hoc refiner.
- Also fixed this round: seeded the in-loop halo eval (deterministic); val cadence -> 45min.
