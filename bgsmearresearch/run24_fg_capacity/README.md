# run24 (PROBE B3) — scale FG head capacity (V6_FG_HIDDEN 512→1024)

**Run:** `runs/BGBAND24_083437`  **Warm-start:** run14 (frozen)  **Budget:** 2 h (~2086 steps)  **Launcher:** `run_bgband24.sh`
**One change vs run23:** nonlinear FG head hidden 512→1024 (heads 3.2M→6.5M params). bg still LOCKED (to_b+to_route
frozen, V6_BG_LOCK). Eval `eval_full_refine` with `V6_FG_NONLINEAR=1 V6_FG_HIDDEN=1024 V6_BG_LOCK=1`; self-consistency
`--zero` == run14 EXACTLY (valid).

## Result — ✗ capacity did NOT grow the gain; it regressed it.
| metric | run14 | run23 (512) | run24 (1024) |
|---|---|---|---|
| CLOUD | 3.34 | 3.31 | 3.34 (bg held ✓) |
| edit_bg_dL | −1.28 | −1.25 | −1.30 (bg held ✓) |
| whole_bg_dL | −10.51 | −10.50 | −10.51 (identical ✓) |
| edit_all_dL | −2.37 | −1.39 | −2.26 |
| **garL1** | 19.12 | **18.34** | **19.36** |
| skinL1 | 19.63 | 19.62 | 19.79 |
| edit PSNR | 19.88 | 19.99 | 19.83 |

## Verdict: capacity is NOT the lever. Doubling the head erased run23's garment gain (18.34→19.36 ≈ run14) — the
larger head trains slower and did not converge in the same 2h, so it under-delivered. bg stayed immovable throughout
(no leak). Combined with run23:
- **Frozen-feature heads give at most a small, fragile garment gain (run23 ≈ 4%), and it does NOT scale with head
  capacity.** The ceiling is the FROZEN run14 features — the foreground-improvement signal isn't richly present in
  them, so no read-off head (bigger or smaller) can extract much.
- ⇒ **This is the "plateau → region-routed LoRA" branch.** To improve foreground meaningfully while keeping bg
  separated, foreground needs its OWN trainable params that MODIFY features (a LoRA adapter), not a head reading
  frozen features. bg stays protected by masking the adapter's effect to foreground regions.

**Panels:** `run24_fg_capacity/*_pred_final_gt.png`.
## Next: region-routed LoRA (design A) — a foreground LoRA adapter on the frozen run14 base, trained on garment/skin
regions only, bg = frozen base (adapter masked out of bg), composited by warped/route masks at deploy. Needs a design
decision (region-gating mechanism + matched twice-forward/blended eval) → held for user direction.
