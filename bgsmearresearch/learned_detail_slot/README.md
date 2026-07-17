# learned_detail_slot — Learned Detail Slot Replacement

**Mode** `GARMENT_ADAPTER_MODE=detail_slot`. NO mid-block path. REPLACE the standard garment conditioning SLOT with learned_detail_slot(aligned garment). Forced-handoff zeros old garment, keeps learned slot. Train slot net + LoRA, v6 frozen. Closest to what Qwen expects.

**Common protocol (all 6).** Pose/agnostic/silhouette/bg context unchanged. The NEW garment path always gets the
CLEAN aligned garment (warped_rgb 3 + garment_latent 16 + warped_mask 1). Asymmetric starvation (GARMENT_SLOT_CORRUPT=zero)
zeros ONLY the standard garment+rough slots fed to run37. Per-step schedule (random): 50% forced-handoff (std starved,
adapter ON), 30% both ON, 10% base-solo (adapter OFF = run37), 10% both OFF. FULL data, 2h, warm-start run37, zero-init
adapter so step0 == run37.

**Pieces.** `model.py` = garment_adapter.py (unified module, this mode selected). `run.sh` = the launcher (exact env).
`WIRING.py` = verbatim trainlib hooks (state, forward schedule/holder/inject, run.py build/save). Live path = guarded
hooks in trainlib importing root garment_adapter.py.

**Eval (4 conditions, 5-panel [baseline run37 | correct | wrong | branch-off | GT] on 5 held-out reserved).**
Branch-off = run37's OWN output (state._GARMENT_ADAPTER_BYPASS=True; NOT v_garment=0). Metrics garL1 + edge/detail.
REJECT if correct/wrong/branch-off look the same, or bg/skin degrade.
