# BG-Smear / Region-Separation Experiment Ledger

Purpose: track every 2h overnight experiment so Codex can review progress every wake-up without reconstructing history from logs.

## Baseline Gate

Current accepted reference: `run14_bgweight` / `runs/BGBAND14_033138/final`.

| Metric | run14 |
|---|---:|
| CLOUD | 3.34 |
| edit_bg_dL | -1.28 |
| edit_all_dL | -2.37 |
| garL1 | 19.12 |
| skinL1 | 19.63 |

A run is accepted only if bg/ring is no worse than run14 and garment and/or skin improves in metrics and panels.

## Rejected But Informative

| Run | Lever | Result |
|---|---|---|
| run18 | skin share up, bg share down | garment/skin improved, bg regressed |
| run19 | garment identity image weight up | garment/skin improved, bg badly regressed |
| run20 | multi-block LR up | best garment, bg regressed |
| run21 | frozen run14 base + LINEAR per-region residual heads (to_g/to_s/to_b) refine x0_pred | garment/skin did NOT improve (garL1 19.12→21.63, skinL1 19.63→20.40); bg overshot bright (edit_bg −1.28→+7.02, CLOUD 3.34→7.19, visible bright halo). |

Conclusion: scalar foreground strengthening trades off against bg (18/19/20). Cheap LINEAR output-space residual heads on frozen features (21) also fail — foreground not improvable from frozen features, and the trainable bg head overshoots. Entanglement is deeper. Next: lock bg exactly (no to_b) + NONLINEAR foreground heads (run23); if still no gain → region-routed LoRA/adapters with true parameter separation.

## run21 — region residual heads (design B) — full entry
- Parent: run14 (frozen LoRA+multiblock). Heads: linear to_g/to_s/to_b refine x0_pred; to_route from run14. USE_V6_REFINE.
- Eval: `/tmp/eval_full_refine.py` (matched); self-consistency `--zero` == run14 EXACTLY → plumbing valid.
- Metrics vs run14: CLOUD 3.34→7.19 ✗ | edit_bg −1.28→+7.02 ✗ | edit_all −2.37→−1.21 | garL1 19.12→21.63 ✗ | skinL1 19.63→20.40 ✗ | PSNR/SSIM 19.88/0.831→19.37/0.807 ✗
- Visual: bright halo ring around person (bg head overshoot). Panels: `run21_region_residual_heads/`.
- Verdict: REJECTED. Foreground not improved + bg degraded.
- Lesson: linear-on-frozen heads lack capacity; trainable bg residual overshoots at deploy.
- Next: run23_bgfrozen_fg_nonlinear (lock bg = run14, no to_b, NONLINEAR garment/skin heads).

## Required Entry Format

Copy this for every new run:

```md
### runNN_name

- Status:
- Launcher:
- Parent checkpoint:
- Hypothesis:
- Changed files/modules:
- Changed env vars:
- Start/end:
- Checkpoint:
- Eval path:
- Panels:
- Metrics vs run14:
  - CLOUD:
  - edit_bg_dL:
  - edit_nonbg_dL:
  - edit_all_dL:
  - whole_bg_dL:
  - far_bg_dL:
  - garL1:
  - skinL1:
  - edit PSNR/SSIM:
- Visual verdict:
- Acceptance verdict:
- Lesson:
- Next proposed run:
```


## run23 — bg-locked + nonlinear FG heads (probe B2) — FIRST separation success
- Parent run14 frozen. bg IMMOVABLE (to_b+to_route frozen, V6_BG_LOCK hard skin-gate → bg=x0_pred exactly, no leak). Train only nonlinear MLP to_g/to_s (3072-512-64). Eval eval_full_refine V6_FG_NONLINEAR=1 V6_BG_LOCK=1; self-consistency --zero == run14 exactly.
- Metrics vs run14: CLOUD 3.34→3.31 ✓held | edit_bg −1.28→−1.25 ✓held | whole_bg −10.51→−10.50 ✓ | edit_all −2.37→−1.39 ✓ | garL1 19.12→18.34 ✓ | skinL1 19.63→19.62 ~flat | PSNR 19.88→19.99 ✓
- Verdict: ✓ SEPARATION WORKS (modest). bg provably immovable + garment improved (first non-Pareto result). Skin flat. Gain small (garment ~4%) — frozen-feature nonlinear heads have limited reach.
- Lesson: region separation IS achievable (frozen bg + separate FG path improves FG without bg cost). Magnitude capped by frozen features.
- Next: run24 = scale FG capacity (V6_FG_HIDDEN 512→1024). If gain grows → scale heads/long-form; if plateaus → foreground needs own params = region-routed LoRA.

## run24 — scale FG head capacity 512->1024 (probe B3)
- Parent run14 frozen, bg locked. One change: nonlinear FG head hidden 512->1024 (heads 3.2M->6.5M). Self-consistency --zero == run14.
- Metrics vs run14 / run23(512): CLOUD 3.34/3.31/3.34 ✓bg held | edit_bg -1.28/-1.25/-1.30 ✓ | garL1 19.12/18.34/**19.36** | skinL1 19.63/19.62/19.79 | edit_all -2.37/-1.39/-2.26
- Verdict: ✗ capacity did NOT grow the gain — it ERASED it (garL1 18.34->19.36 ≈ run14). Bigger head trains slower, under-converged in 2h. bg immovable throughout.
- Lesson: frozen-feature heads cap the garment gain at run23's ~4% and it does NOT scale with head capacity. The FROZEN run14 features are the ceiling. To improve foreground meaningfully, foreground needs its own params that MODIFY features (LoRA adapter), not a head reading frozen features.
- Next: region-routed LoRA (design A) — foreground LoRA adapter on frozen base, trained on garment/skin regions, bg masked out, composited at deploy. Needs design decision → HELD for user direction.

## run25 — 512 FG head + higher LR (5e-4) (probe B4)
- Parent run14 frozen, bg locked. One change vs run23: LR 2e-5->5e-5 (FG heads 5e-4). Self-consistency --zero == run14.
- garL1: run14 19.12 | run23(512,2e-4) 18.34 | run24(1024) 19.36 | run25(512,5e-4) 19.14. bg held everywhere (CLOUD 3.34, edit_bg -1.31).
- Verdict: higher LR did NOT reproduce run23's 18.34 → run23 gain was 5-ID NOISE. Frozen-feature FG heads do NOT reliably improve foreground (any capacity/LR ~ run14 ±noise).
- CONCLUSION (21/23/24/25): bg-locking works (bg immovable); frozen-feature heads can't improve foreground. Next = region-routed foreground LoRA (trainable feature-modifying capacity, bg locked). = run26.

## run26 — REGION-ROUTED COMPOSITE (offline, no training) — ✅ WORKS
- Composite foreground(garment∪skin)=run20 + bg=run14, soft-mask. /tmp/eval_composite.py.
- Result: CLOUD 3.43 (run14 3.34), edit_bg -1.66 (-1.28), garL1 **15.09** (run14 19.12, -4!), skinL1 18.22 (19.63). Clean seam.
- Verdict: REGION-ROUTING WORKS. garment from garment-best + bg from bg-best = best of both, bg held. Pareto tradeoff is intra-model only; cross-model composite dissolves it.
- Deploy: two-forward composite (run14 bg + foreground-model foreground, warped+skin masks). Next: run27 train dedicated foreground (push garment hard, bg composited away) to beat 15.09.

## run27 — dedicated foreground model + 3-way composite
- run27: warm-start run14 unfrozen, garment MAXed. In-loop garment DEGRADED (L1 35->40, over-cooked, run17 pattern — flagged late). Own eval: garL1 17.06 (< run14 19.12 but > run20 15.09), skinL1 17.58 (best), CLOUD 3.98.
- Composites (fg from model + bg run14): 2way fg20 garL1 15.09/skin 18.22 | 2way fg27 garL1 17.04/skin 17.62 | **3way (gar=run20,skin=run27,bg=run14) garL1 15.10/skin 17.60/CLOUD 3.43**.
- VERDICT: run27 = SKIN source (not garment; run20 stays garment). BEST = 3-way composite: garL1 15.10 (-21% vs run14), skinL1 17.60 (-10%), bg held (CLOUD 3.43). REGION-ROUTING fully validated.
- Deploy = N-forward region-routed composite. Productionizing (ship composite / single region-routed FG LoRA / distill) = USER DECISION.

## run28 — BALANCED foreground model → 2-forward composite ✅
- run20 garment lever (multiblock 7.5e-5) + modest skin (S=4), no run27 over-cook. In-loop garment IMPROVING (L1 37→17, healthy — continued on this signal). 956 steps/2h.
- Own eval: garL1 16.24, skinL1 17.69, CLOUD 3.98. Composite (run28 fg + run14 bg): garL1 16.24, skinL1 17.70, CLOUD 3.46 (bg held).
- Verdict: ONE balanced foreground model gives good garment AND skin → 2-forward deploy (vs 3-way 15.10/17.60 at 3 forwards). Region-routing deployable cheaply.
- Deploy options (USER): 2-forward (run28+run14) | 3-forward (run20+run27+run14, best garment) | single region-routed LoRA.

## run29 — SOAR init + run14 recipe (control). RAW, no composite.
- Faithful vs SOAR (9.16/18.18/16.25): CLOUD 4.42 ✓✓ | garL1 17.27 ✓ | skinL1 17.90 ✗ WORSE. In-loop all 3 fell but faithful shows skin regressed.
- Verdict: 2/3 improve (bg+garment), SKIN is the tradeoff casualty. run14 recipe NOT the long-form setup (skin fails).
- Next: run30 (where+when gated) — route skin/garment to mid/late blocks + low-σ, bg to early/high-σ, so skin isn't collateral.

## run30 — where+when gated (block+sigma) from SOAR. RAW.
- Faithful: CLOUD 3.64 ✓✓ (best SOAR-init, ~run14) | garL1 18.32 ✗ | skinL1 18.15 ✗. Great bg but foreground starved.
- Verdict: block/sigma gating over-protects bg, starves foreground. Allocation approach exhausted (run29 lost skin, run30 lost garment+skin). → STRUCTURAL: mask-gated hidden-state injection next (run31).

## run31 — mask-gated GARMENT injection (region-owned). SOAR init, RAW.
- Faithful: CLOUD 4.34 ✓ (vs run20 global-7.5e-5 4.93 — mask-gating protects bg at same strength) | garL1 17.74 (SOAR 18.18) | skinL1 18.82 ✗ worst.
- Verdict: PARTIAL — region-owned garment write PROTECTS bg (confirmed vs run20), but garment gain modest + skin regressed (no skin path). → run32 skin adapter. In-loop garment over-promised (14.48 vs faithful 17.74).

## run31b — mask-gated GARMENT injection @ MULTI_BLOCK_LR=5e-5 (clean A/B vs run31's 7.5e-5). SOAR init, RAW.
- Only diff vs run31: MULTI_BLOCK_LR 7.5e-5 → 5e-5.
- In-loop [DEPLOY] (FAITHFUL — hooks applied; eval_full is LoRA-only/unfaithful for injection): step378 GARMENT 21.79 / SKIN 13.41 / BG 7.71 (BAND -3.14); step724 GARMENT 38.47 / SKIN 12.94 / BG 11.29 (BAND -5.36).
- Verdict: NEGATIVE. Garment DIVERGES at 5e-5 (21.79→38.47, dim +0.6→-7.32 = garment darkening/collapse); bg also worsens (7.71→11.29). Lower LR is WORSE than run31's 7.5e-5 (garment 14.48). → 7.5e-5 mask-gate is the correct garment base. run32 (skin adapter) built on run31's 7.5e-5 config, NOT run31b.

## run32 — region-owned SKIN adapter (311M, 8 sites, skin-token-gated xattn, src=body/agnostic skin) on run31's GOOD 7.5e-5 mask-gate garment base. SOAR init, RAW single model.
- In-loop [DEPLOY] (FAITHFUL): step383 GARMENT 21.65 / SKIN 14.34 / BG 5.53 (BAND -1.83); step747 GARMENT 15.27 / SKIN 12.45 / BG 4.07 (BAND -1.56). Stopped at TIME_BUDGET (2 deploys).
- Verdict: POSITIVE — as skin-adapter gates ramp (init 0.01), ALL THREE improve together: SKIN 14.34→12.45 (adapter engaging, ~run31 level), GARMENT 21.65→15.27 (recovers toward run31 14.48, NO divergence unlike run31b 38.47), BG 4.07/BAND -1.56 (healthy, near target -1). Region-owned skin write does NOT break garment/bg. First run with garment~15 + skin~12.5 + bg~4 all healthy at one deploy. → keep skin adapter for combined. Next: run33 bg adapter (isolate), then combined all-3.

## run33 — region-owned BG adapter (236M, 6 sites @ EARLY/MID blocks 4,8,12,16,20,28, bg-token-gated xattn, src=agnostic bg) on 7.5e-5 mask-gate garment base. SOAR init, RAW. NO skin adapter (isolate bg lever).
- In-loop [DEPLOY]: step375 GARMENT 28.15 / SKIN 13.45 / BG 9.78 (BAND -4.13); step724 GARMENT 23.65 / SKIN 11.45 / BG 6.77 (BAND -2.96). Stopped at TIME_BUDGET (2 deploys).
- Verdict: bg adapter ENGAGES (as gates ramp: BG 9.78→6.77, BAND -4.13→-2.96 toward target -1; all metrics improving). BUT weaker trajectory than run32 at same step (run32 s747: GAR 15.27/SKIN 12.45/BG 4.07/BAND -1.56 vs run33 s724: GAR 23.65/SKIN 11.45/BG 6.77/BAND -2.96). run33 garment notably higher (23.65 vs 15.27) + bg worse than run32 — likely early-block bg writes perturbing shared feats, or SOAR run-to-run variance. bg adapter works directionally but skin adapter (run32) gave cleaner all-3. → COMBINED run34 (all 3) to see if together beats either alone.

## run35A — FROZEN SOAR + isolated skin adapter (causality test). Only skin adapter trains (disjoint blocks 16,24,40, gate=1.0, zero-init).
- Faithful vs SOAR: garL1 15.06 (=SOAR 15.12 ✓ freeze works) | skinL1 13.18 (WORSE vs 12.97) | far_bg -10.93 (darker vs -9.95) | CLOUD 6.28 (vs 5.87).
- Adapter ENGAGED (out_proj 0→0.09, gates 1.0 — run32 dead-gate fixed). But skin NOT improved + bg speckle artifacts (panels).
- Verdict: REJECT skin-adapter mechanism. Freeze PROTECTION validated (garment locked); but isolated skin adapter doesn't help skin + adds bg artifacts → design/source/mask wrong. Do NOT scale adapters (no 35B). Skin damage = training-recipe problem, not missing path. → PCGrad (36/37) + temporal (39) more promising.

## run36 — PCGrad on shared LoRA+v6 (global injection, no adapters). SOAR init. ~290 steps (undertrained, 4x-slow).
- PCGrad found REAL conflict (projected_conflicts up to 4/6). 
- Faithful vs SOAR: CLOUD 4.29 (vs 5.87) | whole_bg -4.92 (vs -8.25, +3.33) | far_bg -5.23 (vs -9.95, +4.72) | skinL1 10.97 (vs 12.97, -2.0) | garL1 15.20 (=SOAR 15.12).
- Verdict: **WIN / BREAKTHROUGH**. First run to improve bg + skin while HOLDING garment (no cross-region tradeoff). Panels: bg nearly clean white, face clean, garment crisp. Gradient conflict WAS the cause; PCGrad fixes it. Undertrained & already best → full-length PCGrad is priority. → run37 = PCGrad + mask-gate.

## run37 — PCGrad + run31 mask-gated garment (no adapters). SOAR init. ~285 steps.
- Faithful vs SOAR: CLOUD 4.36 | whole_bg -4.93 | far_bg -5.19 | skinL1 11.12 | garL1 14.96 (BEST garment).
- Verdict: CONFIRMS run36 (PCGrad WIN). bg dramatically cleaner + skin better, garment best-yet (14.96 via mask-gate). run36≈run37 on bg/skin. PCGrad is the driver; mask-gate adds small garment edge. → FULL-LENGTH PCGrad+maskgate is top priority.

## run38 — frozen-SOAR + v6 forked per-region heads (639K, to_s/to_b/to_route). WELL-TRAINED ~1960 steps.
- Faithful vs SOAR: CLOUD 5.71 | whole_bg -8.01 | far_bg -9.67 | skinL1 12.81 | garL1 15.20. ≈ SOAR (barely moved).
- Verdict: WEAK/NEGATIVE. Forked heads barely improve DESPITE 7x more training than PCGrad; PCGrad (undertrained) crushes it. Confirms residual-writer family (35A adapter + run38 forked heads) does NOT fix cross-region damage. PCGrad (gradient-conflict resolution) is the unique winner. → adapter/forked direction CLOSED.

## run39 — temporal σ-routed loss (run31 maskgate + σ-gated region weights). Well-trained ~900 steps.
- Faithful vs SOAR: CLOUD 5.43 | whole_bg -8.40 | far_bg -10.40 | skinL1 14.33 | garL1 15.27. ≈ SOAR (slightly worse bg/skin).
- Verdict: WEAK/NEGATIVE. Temporal routing insufficient (confirms run30). PCGrad remains unique winner. → temporal CLOSED.
