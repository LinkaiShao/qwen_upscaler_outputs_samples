# run08 — RETRACTED (broken setup: train != deploy)

**⚠️ CONCLUSION RETRACTED.** This run did NOT train the deployed composition. Training decoded/scored raw `x0_pred` (latent.py old path), the v6 loss trained `δ_s`/`δ_b` independently vs `(person−agnostic)` + zero-keep penalties, and val was raw train_step img (frozen `val_img=0.4176`). So `rep` staying flat proves the **auxiliary residual objective is bad**, NOT that the V6 region strategy is impossible. Superseded by run09 (real train==deploy composite via `USE_V6_DEPLOY_COMPOSE=1`).

---

# run08 — 5-ID overfit of v6 heads (train==deploy composition) — ❌ NOT LEARNABLE

**Run:** `runs/BGBAND02_174447`  frozen LoRA + garment, train only v6 heads, `OVERFIT_SIDS`=5 fixed IDs, v6 loss (bg/skin δ→(person−agnostic) in M_repair + route CE), lr 2e-4, ~15 min / 465 steps.

## Result → ❌ V6 head capacity/target is WRONG (per the decision rule)
- **`rep` (repair reconstruction) FLAT at ~0.55** the entire run (step 10: 0.543 → step 460: 0.565). The bg/skin heads never reduced the reconstruction loss — they cannot fit `(person−agnostic)` even on 5 fixed images.
- **`rou` (route CE) = 0.03** — routing DID learn (a Linear classifies gar/skin/bg fine).
- **Gated-deploy CLOUD = 53.4** (soar 8.19, run03 4.84) — confirms the heads produce no valid residual; deploy composition still garbage.

## Conclusion
The V6 route is fine; the **residual-reconstruction head is not learnable**. A single per-token `Linear(3072→64)` can route but cannot regress the full latent residual, even overfitting 5 imgs. → **Stop chasing frozen-V6 for bg.** Return to **LoRA-deployed fixes only** — the real progress was there: run03 (bg-share 0.30, deployed) = CLOUD **4.84**, the best valid result. The frozen-head path is closed by this probe.
