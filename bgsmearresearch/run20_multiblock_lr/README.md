# run20 (C) — run14 + multi-block injection LR (5e-5→7.5e-5)

**Run:** `runs/BGBAND20_201250`  **Warm-start:** run03  **Budget:** 2 h (~1000 steps)  **Launcher:** `run_bgband20.sh`
**Single lever vs run14:** `MULTI_BLOCK_LR` 5e-5→7.5e-5 (garment conditioning path strength). Else = run14. Standard eval.

## Result — ✗ REJECT (bg regressed), but BEST garment of the whole queue.
| metric | run14 | run20 | acceptance | pass? |
|---|---|---|---|---|
| CLOUD | 3.34 | 4.93 | ≤ ~3.5 | ✗ |
| edit_bg_dL | −1.28 | −3.54 | (bg protect) | ✗ bg darker |
| edit_nonbg_dL | −2.91 | +2.20 | — | ✓ |
| edit_all_dL | −2.37 | +0.52 | ≥ −2.5 | ✓ |
| whole_bg_dL | −10.51 | −15.68 | — | ✗ raw bg darker |
| far_bg_dL | −17.12 | −24.77 | — | ✗ |
| **garL1** | 19.12 | **15.09** | ≤ 19.12 | ✓✓ best-ever |
| skinL1 | 19.63 | 18.20 | ≤ 19.63 | ✓ |
| edit PSNR/SSIM | 19.88/0.831 | 20.40/0.834 | — | ✓ |

**Reading:** Strengthening the garment conditioning path (multi-block LR) gave the **best garment of the queue**
(garL1 15.09, −4.0 vs run14) and better skin — but **still regressed the bg** (CLOUD 4.93, whole_bg −15.68).
**Refutes the hypothesis that the conditioning path spares the bg.** Combined with run18 (share) and run19 (img
weight), the conclusion is robust: **garment/skin improvement and bg cleanliness are ANTI-CORRELATED at run14's
operating point — run14 sits on a Pareto frontier.** Any single scalar that helps garment/skin darkens the bg,
whether via loss share, img weight, or conditioning strength.
**Panels:** `run20_multiblock_lr/panel_all.png`.
**Implication:** to get garment/skin gains WITHOUT the bg cost you need to move the frontier, not push a scalar —
e.g. longer training (the long-form baseline) so both are learned to convergence, or an explicit bg-protection
term that holds the bg while garment strengthens. Single 2h levers can only trade along the frontier.
