# run19 (B) — run14 + garment identity (W_IMG_V6_G_ID 4.0→5.0)

**Run:** `runs/BGBAND19_180718`  **Warm-start:** run03  **Budget:** 2 h (~1000 steps)  **Launcher:** `run_bgband19.sh`
**Single lever vs run14:** `W_IMG_V6_G_ID` 4.0→5.0 (garment identity img weight). Else = run14. Noise-fill ⇒ standard eval.

## Result — ✗ REJECT (bg badly regressed; worst CLOUD of the queue).
| metric | run14 | run19 | acceptance | pass? |
|---|---|---|---|---|
| CLOUD | 3.34 | **6.11** | ≤ ~3.5 | ✗✗ (00008 outlier 11.4) |
| edit_bg_dL | −1.28 | −5.47 | (bg protect) | ✗ bg much darker |
| edit_nonbg_dL | −2.91 | −0.09 | — | ✓ |
| edit_all_dL | −2.37 | −1.54 | ≥ −2.5 | ✓ |
| whole_bg_dL | −10.51 | −18.28 | — | ✗ raw bg darker |
| far_bg_dL | −17.12 | −28.75 | — | ✗ |
| garL1 | 19.12 | 17.25 | ≤ 19.12 | ✓ better |
| skinL1 | 19.63 | 18.20 | ≤ 19.63 | ✓ better |

**Reading:** Raising garment-identity img weight improved garment (17.25) and skin (18.20) but **badly regressed bg**
(CLOUD 3.34→6.11, edit_bg −1.28→−5.47, whole_bg −10.51→−18.28). Same failure mode as run17 — **any foreground
img-weight boost steals from bg.** Confirms run14's region balance is a knife-edge: garment/skin gains via img
weights come at the bg's expense. Not accepted.
**Panels:** `run19_garment_id/panel_all.png`.
**Takeaway:** garment/skin recovery must come from region-SHARE (run18 direction, milder) or the conditioning path
(run20 multi-block LR / run21 early LoRA), NOT from raising the foreground image-loss weights.
