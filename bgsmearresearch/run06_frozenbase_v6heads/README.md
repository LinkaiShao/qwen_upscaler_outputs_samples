# run06 — frozen try-on base, train ONLY the v6 bg/skin/route heads

**Status:** QUEUED — launches automatically when run05 finishes.  **Launcher:** `run_bgband06.sh`  **Warm-start:** soar.  **Budget:** 2h.

## The question this answers (user-directed)
Can a **region-specialized bg path** reduce raw CLOUD **without hurting garment/skin**? — by fixing bg through the v6 heads while the try-on model is frozen, instead of retraining the whole LoRA (which drifts garment).

## Design (one change: WHAT trains, not the loss)
- **Freeze the try-on base:** `FREEZE_LORA=1` (main LoRA frozen) + `FREEZE_MULTI_BLOCK=1` (garment injection frozen). → garment/skin try-on **cannot be destroyed** by construction (rule 3 satisfied structurally, not by protection weights).
- **Train ONLY the v6 heads:** `USE_V6=1 FREEZE_V6=0` → `to_s` (skin), `to_b` (bg residual), `to_route` (paint-vs-keep) at lr×10, warm-started from soar's `v6_heads.pt`. These read pre-proj features on a separate gradient path.
- **Same detach-% region losses** (`USE_REGION_PCT_LOSS=1 USE_FLOW_BG_SPLIT=1`, bg 0.30) drive the heads toward real GT bg; garment/skin terms compute but flow only into the (frozen→no-op) base + the trainable skin head.
- `USE_V6_IMG_STAGE=1`, `SOAR_PROTECT_GARMENT=1`, garment/person protected in rollout (moot since frozen).
- Exact shares recorded on eval.

## Eval (required)
Fixed 5-ID raw `predict_sample` CLOUD + garment/skin L1 + **both** `pred_vs_gt` (inference) and `final_vs_gt` (paste) panels. Compare to soar 8.19, run03 4.84 (best), if5 3.4.

## Decision
- **If bg CLOUD drops with garment/skin L1 unchanged (frozen) →** the region-specialized bg path works; then **gradually unfreeze more of the main LoRA** (low-LR) to push further.
- **If bg CLOUD does NOT drop →** the routing/region *target* is wrong (not the whole model); rethink the bg head's objective, don't unfreeze.

## Result — raw `predict_sample` CLOUD
_pending (runs after run05)._

## Result → ❌ NO CHANGE (8.19) — and a deployment-path problem

Raw `predict_sample` CLOUD = **8.19, pixel-identical to soar** (Δ=−0.00 every ID).

**Root cause (important):** run06 trained `v6_heads.pt` (Linear `to_s/to_b/to_route`, applied via a norm_out hook in the *training* forward with `USE_V6_IMG_STAGE`). But the deployment `predict_sample` (0608 inference) loads `repair_head.pt`/`routing_head.pt` (Conv2d) — which run06 does NOT produce — so **the trained v6 heads have NO deployment path**. With the LoRA frozen (=soar) and heads not applied, the eval literally regenerates soar.

**Cross-check (head-applying path):** the in-loop `[DEPLOY]` halo_eval DOES apply the v6 heads, and it was **flat** (BG L1=13.0 at steps 891 and 1782) → the heads didn't move bg even where applied.

**Answer to the run's question:** can a region-specialized bg path (v6 heads, frozen base) reduce raw CLOUD? **No** — on two counts: (1) the heads have no path in the faithful deployment inference, and (2) even in the head-applying path they didn't reduce bg. Per the plan, this means the **routing/region target is wrong**, not the whole model — do NOT unfreeze on this basis.

**Lesson:** verify a trained component actually deploys in the eval inference BEFORE spending a 2h run. The v6_heads/multi_block are training-only in this lineage; only the LoRA deploys via 0608 predict_sample (which is why runs 02–05's gains came from the LoRA, not the heads).
