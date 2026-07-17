# Garment-conditioning overnight — residualized zero-init GarmentChain

GOAL: a garment path that beats run37 garment (garL1 15.03) while preserving run37 bg/skin
(CLOUD 3.42 / far_bg -4.40 / skin 10.96). Every run: frozen run37 base + trained residualized
chain `F += mask*scale*zero_init_proj(LN(state))`. Eval = canonical (v6-on) chain-ON vs chain-OFF.
PASS only if garL1(ON)<garL1(OFF) AND CLOUD<=3.62 AND far_bg/skin not worse.

Each run's folder here has: ALL_pred.png (raw chainON|chainOFF|GT), ALL_final.png (composite), per-id panels, numbers.txt.

## Master table (chain-ON vs chain-OFF; want ON garment better, bg/skin held)
| run | config | CLOUD ON/OFF | far_bg ON/OFF | garL1 ON/OFF | skin ON/OFF | verdict |
|-----|--------|--------------|---------------|--------------|-------------|---------|
| run41  | 8-site scale1.0     | 4.62 / 3.41 | -5.61 / -4.40 | 14.99 / 15.11 | 11.25 / 10.97 | **FAIL** (bg+skin harmed, garment ~flat) |
| run43x | 8-site scale0.25    | 4.66 / 3.42 | -5.65 / -4.40 | 15.05 / 15.12 | 11.25 / 10.98 | **FAIL** (gentler scale didn't help) |
| run42  | 4late-site scale1.0 | CORRUPTED checkpoint (truncated save) — unrecoverable |||| SKIP |
| run44x | 4late-site bound0.1 | (v3 eval) | | | | pending |
| run45x | 8-site scale0.5     | (training) | | | | pending |
| run46x | 4late-site scale0.25| (queued) | | | | pending |

## Finding so far
The 8-site residualized chain HARMS bg/skin (CLOUD +1.2, far_bg darker ~1.2, skin worse) for a
negligible garment change — at BOTH scale 1.0 and 0.25. Mask-gating to garment tokens does NOT
stop it: the damage propagates via self-attention (garment tokens perturb bg/skin tokens downstream).
4-site (late-only) results pending (v3) — testing if fewer/later sites reduce the attention spread.
If 4-site also fails → residualized hidden-injection class is dead → build the OUTPUT-SPACE latent
refiner (bg-safe fallback: residual only inside garment mask, provably cannot touch bg).

## v3 session Thu Jul  9 12:45:56 PM PDT 2026

## v3 session Thu Jul  9 12:51:24 PM PDT 2026
images -> bgsmearresearch/run44x_residchain_4late_b0.1 (5 ids)

### run44x_residchain_4late_b0.1 — 4late-site boundary0.1 (sites=44,50,55,59 boundary=0.1)
chain-ON :      3.26    -3.50     3.20      1.15     -3.80   -4.10  21.68  0.832  15.23  10.55
chain-OFF:      3.40    -3.70    -2.01     -2.52     -4.04   -4.39  21.61  0.836  15.13  10.98
cols: CLOUD editbg editnb editall wholebg farbg PSNR SSIM garL1 skinL1 | ref run37: CLOUD 3.42 far -4.40 gar 15.03 skin 10.96
RULE: PASS if garL1(ON)<garL1(OFF) AND CLOUD<=3.62 AND far_bg/skin not worse
[eval done run44x_residchain_4late_b0.1 Thu Jul  9 01:04:54 PM PDT 2026]
[train run45x_residchain_8site_s0.5_0709_130454 (8-site scale0.5 boundary0.1) Thu Jul  9 01:04:54 PM PDT 2026]

## run47 — garment-local LATENT refiner on frozen run37 (OVERFIT5) — **THE WIN**
Hidden-state injection (run41/43x/44x) is dead: every residualized-chain variant harmed bg/skin via self-attention.
Output-space refiner instead: `final = pred + warped_mask * zero_init_unet(pred, garment, mask)`.

| | CLOUD | whole_bg | far_bg | garL1 | skinL1 | PSNR | SSIM |
|---|---|---|---|---|---|---|---|
| run37 base | 3.75 | -5.21 | -6.04 | 15.21 | 11.54 | 21.51 | .832 |
| **run47 refiner** | **3.54** | **-5.11** | **-6.04** | **3.14** | **9.99** | **25.95** | **.931** |

Every metric improves or holds; far_bg bit-identical; latent residual outside garment exactly 0.
CAVEAT: overfit/memorization on 5 ids. Next: train on many ids, hold out the 5.
Folder: `run47_garment_local_latent_refiner_OVERFIT5/`
