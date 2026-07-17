# run09 — V6 heads trained through the DEPLOYED gated composite (train==deploy) — LEARNS, not converged

**Run:** `runs/BGBAND02_185043`. Frozen LoRA+garment, train only v6 heads, 5-ID overfit, `USE_V6_DEPLOY_COMPOSE=1` (score the gated V6 latent composite as pred_img), `LAMBDA_V6_REPAIR=0` (aux δ-loss dropped — it was the bad objective; `rep` now a meaningless diagnostic), route CE kept. 30 min / 305 steps.

## Result → ✅ LEARNABLE (refutes run08), quality not yet converged
- **val_img improving monotonically:** 2.68 → 2.36 → 2.07 → 1.97 (judge by this, NOT `rep`).
- **train img (on the composite) monotonic:** 2.58 → 2.19, still falling at the cutoff (not plateaued).
- **Gated-deploy raw CLOUD = 24.87** (soar 8.19, run07 58.8, run08 53.4) — more than halved vs the broken/untrained runs, and NOT converged.
- **Panels:** garment + outside-edit preserved; repair ring still confetti (undertrained — 305 steps is far too few).

## Conclusion
The corrected **train==deploy composite objective learns** — the heads reduce the composed image loss and the deploy CLOUD drops (59→25) while training. run08's "V6 not learnable" was an artifact of scoring raw x0_pred + the bad aux δ-loss. **The V6 region path is NOT closed.** Next: (a) lock the deploy/eval inference to this exact composite (self-contained), (b) train much longer to converge the ring, (c) then judge final CLOUD/panels.
