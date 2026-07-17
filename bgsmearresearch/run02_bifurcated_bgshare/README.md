# run02 — bifurcated region-% with bumped bg share (rule-compliant fix)

**Status: NOT YET RUN — setup for review.**
**Launcher:** `run_bgband02.sh`   **Warm-start:** soar (`soar_noise_20260619_170822/final`)   **Budget:** `TIME_BUDGET=7200` (2 h)

## Why run01 failed and what changes here

run01 bolted **fixed-weight, single-step** band losses (DARK=10/CLOUD=5/CHR=2) onto the raw x0 and regressed (8.19 → 10.38). Reading `TRAINING_RULES.md`, that broke rules 1 (fixed weights, not detach-%), 2/5 (3 extra terms + gem = soup/patch), and 4 (single-step is blind to the deployed rollout — which is exactly where the darkening lives).

run02 does **not** add any new loss. It uses the existing **rule-compliant bifurcated framework** and moves the one lever the rules allow: the **bg region's detach-normalized gradient share**.

## The setup (maps to the 5 rules)

| Rule | How run02 satisfies it | Config |
|---|---|---|
| **1 True bifurcation** | bg is a detach-normalized `%` of gradient (`p·L/(L.detach()+eps)`), not a fixed weight. **Bumped bg share 0.25 → 0.35 in BOTH the flow split and the region-%** (they must move together), skin 0.25 → 0.20 to make room, garment stays 0.50 dominant. Verify it fires via `RPCT_DEBUG`. | flow: `PCT_FLOW_CORE=0.50 PCT_FLOW_BODY=0.20 PCT_FLOW_BG=0.35 PCT_FLOW_UB=0.04 PCT_FLOW_KEEP=0.01`  ·  region-%: `RPCT_GARMENT=0.50 RPCT_SKIN=0.20 RPCT_BG=0.35 RPCT_BOUNDARY=0.08 RPCT_KEEP=0.02` |
| **2 No soup** | bg bundle = **one** target (→ real GT): `m_bg` (flow) + 0.5·`L_img_b` (img) + `gem` + `bsab`, all pulling to real bg. No counteracting terms. My run01 bolt-on losses OFF. | `W_GARMENT_EDGE_MATCH=8.0` (gem, in bundle), `LAMBDA_BG_SHELL_AB=1.0` (bsab, in bundle), `LAMBDA_BGBAND_*` unset (0) |
| **3 Lower-level priority** | garment kept dominant (0.50 > bg 0.40); garment/person **protected** during the SOAR rollout so bg correction can't smear it. | `SOAR_PROTECT_GARMENT=1 SOAR_PROTECT_PERSON=1` |
| **4 Deployed eval** | the bg bundle acts on the **rolled-out** SOAR prediction (multi-step), so it reaches the darkening single-step couldn't; select/score on the deployed from-noise metric. | `USE_SOAR=1 SOAR_PROB=0.25 SOAR_KSTEPS=4 SOAR_FORCE_START=1`; `DEPLOY_HALO_EVAL=1 DEPLOY_EVAL_HOURS=0.7`; final eval = raw `predict_sample` CLOUD |
| **5 Keep it simple** | removed 3 bolt-on losses; no new terms — only re-weighted an existing region share. | — |

## Hypothesis
The bg darkening is a rollout/trajectory bias. Giving the **bg region a larger detach-normalized share (0.35)** of the gradient — moved together in the flow split and region-%, computed on the SOAR-rolled-out prediction, pulling to real GT — should reduce deployed bg CLOUD toward if5's ~3.4 **without** garment regression (garment stays 0.50 dominant + protected). Start at 0.35 so bg increases without becoming competitive with garment; only go to 0.40 if 0.35 helps and garment stays stable.

## A/B plan (interpretable, no new losses)
1. **run00** — soar baseline (done: 8.19)
2. **run02** — soar + bg share **0.35** (this run)
3. **run03** — if run02 improves and garment is stable, soar + bg share **0.40** (`RPCT_BG=0.40 PCT_FLOW_BG=0.40`)

Everything else stays fixed across all three, so any CLOUD change is attributable to the bg share alone.

## Verification before trusting results (rule 1 + 4)
1. `RPCT_DEBUG` line must show `flow_bg_present=True` and a nonzero `B=` bundle magnitude — else the bg branch silently didn't fire (needs v6 masks; `USE_V6=1`). I will confirm this in the first ~20 steps before letting it run.
2. Score on **deployed raw `predict_sample` CLOUD** (5 IDs), not single-step val.

## Eval / success criteria
Run `/tmp/eval_bgband.sh runs/<BGBAND02>/final bgsmearresearch/run02_bifurcated_bgshare` on completion → fills the table below + `images/<ID>_final_vs_gt.png` (FINAL deployed | GT, raw-CLOUD label).

- **Success:** mean raw CLOUD **< 8.19** (soar) with no garment regression (watch deployed GARMENT metric); ideally toward ~3.4 (if5).
- **No change / regress:** the bg-share lever isn't enough on a 2 h warm-start → next is a from-scratch bifurcated foundation or an explicit bias-correction term.

## Result — raw `predict_sample` CLOUD → ✅ IMPROVED (bg), ⚠ garment regressed

| ID | CLOUD | Δ vs soar | darkening |
|---|---|---|---|
| 00006 | 5.89 | −1.15 | −4.38 |
| 00008 | 7.49 | −0.61 | −5.13 |
| 00013 | 3.91 | −4.19 | −3.27 |
| 00017 | 4.18 | −3.57 | −2.69 |
| 00034 | 4.32 | −5.66 | −3.62 |
| **mean** | **5.16** | **−3.03** | ~−3.8 (from −7.7) |

**bg: real win** — mean CLOUD 8.19 → 5.16, toward if5's 3.4; darkening roughly halved. The compliant approach (detach-% bg share on the rolled-out SOAR pass) works where run01's single-step bolt-on failed.

**garment: regressed (rule-3 flag).** Deployed garment L1 rose on 2/3 checked IDs (00013 17.9→21.1, 00034 8.2→12.7; 00006 fine), and in-loop `[DEPLOY]` GARMENT L1 climbed 25→34. Visually the garment is only *mildly* softer (collar/pockets intact), but it is not stable.

**Decision:** per the A/B plan (0.40 only if garment stable) and rule 3 (garment priority), do **not** escalate to 0.40. Next = **run03 at bg 0.30** (back off the share to keep the bg gain without the garment cost). If 0.30 restores garment while holding most of the bg win, that's the sweet spot; else the lever is share-vs-garment-bound and the real fix is an explicit bias-correction that doesn't touch garment.
