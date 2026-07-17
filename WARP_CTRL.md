# WARP_CTRL — warper experiment control file

## ★ FINAL DIRECTION SCOREBOARD (all PROPER, equal 1.5h / ~4200-5000 steps, no early-stop)
| run | latent L1 (correct) | PIX garL1 (correct) | EDGE/detail (correct) |
|---|---|---|---|
| **D baseline** | 0.177 | **17.63** (best) | **4.641** (best) |
| A residual   | 0.177 | 18.27 | 4.905 |
| B high-pass  | 0.181 | 20.20 | 4.680 |
| C copied-init| 0.207 | 23.46 | 5.123 (worst) |
| exp03 ref (2h/9805 steps) | 0.176 | 14.67 | — |

**VERDICT: none of the 3 modifications beat the plain baseline (D) at equal budget.** D has the best pixel AND best edge. Residual = no help (worse edge). High-pass = hurt pixel (20.2 vs 17.6), edge ~tied — the ×3 edge weight did NOT actually sharpen. Copied-init = actively worse (fresh random blocks were not the bottleneck). Identity separation held for all (correct ≪ wrong/zero). D at 1.5h (17.63) vs exp03 at 2h (14.67) = same config, just less training — consistent.
**Conclusion: the blur/no-detail limit is NOT fixable by residual-target, high-pass loss, or copied init. It's the MSE-regression conditional-mean floor — needs a non-MSE objective (adversarial / diffusion decode) to add real detail.**

---


Single source of truth. I (the agent) read this every ~30 min, act, and update it.
No detached queue orchestrators — just one run at a time + polling.

## PROTOCOL
- On each check: read this file, check `nvidia-smi` + the RUNNING job, update STATE below.
- If GPU idle AND there is a PENDING experiment → launch it (single `nohup ... &` run, survives).
- When a run finishes → run its eval, move it to DONE with results, advance to next PENDING.
- Individual `nohup train_warper.py` runs SURVIVE teardown; the agent is the durable orchestrator via 30-min self-wakeups.

## STATE
- updated: 2026-07-14 11:21
- **CHAIN COMPLETE — all A/B/C/D done. GPU idle. Wakeups stopped. See scoreboard at top.**

## PENDING
(none)

Launch template (single nohup, survives):
```
nohup env WARP_EXP=3 WARP_FULL_TRAIN=1 WARP_NO_EARLYSTOP=1 WARP_BATCH=8 WARP_VAL_EXTRA=20 \
  TIME_BUDGET=5400 WARP_EVAL_EVERY=200 WARP_EVAL_FIRST=50 <EXTRA_ENV> \
  WARP_OUT=bgsmearresearch/warp_<TAG>_FULLDATA python -u train_warper.py > logs/warp_<TAG>_proper.log 2>&1 &
```

## DONE
- **D_baseline PROPER**  : 4257 steps (1.5h) | latentL1 c=0.177 | PIX garL1 c=17.63 | EDGE c=4.641. BEST of the four on both pixel and edge — the modifications did not beat it.
- **A_residual PROPER** : 4229 steps (1.5h, no early-stop) | latentL1 c=0.177 w=0.431 z=0.415 | PIX garL1 c=18.27 w=101.96 | EDGE c=4.905 w=5.147. Identity strong; but PIX 18.27 > exp03 14.67 and EDGE not improved — residual target did NOT sharpen. Compare vs D_baseline at equal 1.5h.
- **B_highpass PROPER** : 4242 steps (1.5h) | latentL1 c=0.181 w=0.428 z=0.437 | PIX garL1 c=20.20 w=99.59 | EDGE c=**4.680** w=4.934. Best EDGE/detail so far (< A 4.905) but PIX worse (20.20 vs A 18.27) — high-pass trades color for edge, as expected.
- **C_copiedinit PROPER**: 4998 steps (1.5h) | latentL1 c=0.207 | PIX garL1 c=23.46 | EDGE c=5.123. WORST of the four — copied Qwen init did NOT help; fresh random blocks were not the bottleneck.
- direction A_residual  : 900 steps  PIX 20.62 EDGE 4.821  (undertrained, superseded)
- direction B_highpass  : 1500 steps PIX 18.24 EDGE 4.679  (best of the four, but undertrained)
- direction C_copiedinit: 3600 steps PIX 21.32 EDGE 5.036  (worst)
- direction D_baseline  : 1500 steps PIX 19.24 EDGE 4.732
- exp03 full-data 2h    : 9805 steps PIX 14.67 (converged reference; blurry, no detail)
