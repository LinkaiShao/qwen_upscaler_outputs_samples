# run25 (PROBE B4) — run23 512-head config + higher FG LR (5e-4)

**Run:** `runs/BGBAND25_104621`  **Parent:** run14 frozen, bg locked  **One change vs run23:** LR 2e-5→5e-5 (FG heads 2e-4→5e-4).
Eval `eval_full_refine` V6_FG_NONLINEAR=1 V6_FG_HIDDEN=512 V6_BG_LOCK=1; self-consistency --zero == run14 exactly.

## Result — higher LR did NOT reproduce run23's gain.
| run | config | CLOUD | edit_bg | garL1 | skinL1 |
|---|---|---|---|---|---|
| run14 | baseline | 3.34 | −1.28 | 19.12 | 19.63 |
| run23 | 512, lr2e-4 | 3.31 | −1.25 | **18.34** | 19.62 |
| run24 | 1024, lr2e-4 | 3.34 | −1.30 | 19.36 | 19.79 |
| **run25** | **512, lr5e-4** | 3.34 | −1.31 | **19.14** | 19.59 |

## Verdict: run23's 18.34 does NOT replicate → it was 5-ID eval NOISE, not a real gain.
Neither more capacity (run24=19.36) nor faster training (run25=19.14) reaches run23's 18.34; both land at run14-level
(~19.1–19.4). bg stayed locked/immovable throughout (CLOUD 3.34, edit_bg −1.31 — no leak). ⇒ **frozen-feature FG heads
(linear or nonlinear, any capacity/LR) keep garment at ~run14 ±noise and do NOT reliably improve foreground.**

## Campaign conclusion (runs 21/23/24/25): bg-LOCKING WORKS (bg provably immovable in every refine run), but reading
FROZEN run14 features cannot extract foreground improvement — the ceiling is the features, not the head. This confirms
the structural conclusion: **foreground needs its own trainable feature-MODIFYING capacity that cannot touch bg**
= region-routed foreground LoRA/adapter with bg locked. That is the committed next direction (run26).
