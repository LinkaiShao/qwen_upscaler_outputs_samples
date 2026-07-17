# KV_CLEAN — clean co-train + WRONG-garment contrastive test (BUILT, NOT RUN)

Rebuilt after the previous KV run was found untrustworthy (inert credit, no real wrong-contrastive,
muddy layered config, broken eval). This is the honest test: **does an UNFROZEN run37 learn to READ
an injected garment K/V?**

## What's new / fixed vs the muddy run
1. **Real WRONG-garment contrastive** (the reading test) — `trainlib/forward.py` (branch-credit block):
   when `USE_KV_WRONG_CONTRAST=1`, run a second pass with the garment_latent ROLLED across the batch
   (sample i gets sample i-1's garment), re-encode the K/V, and penalize `relu(gar(correct) − gar(wrong) + margin)`
   so the correct garment MUST reconstruct the garment region better than a wrong one. Metrics logged:
   `bc_on`=gar(correct), `kvc_wrong`=gar(wrong), `bc_off`=gar(no-KV). **Read the log: WANT bc_on < kvc_wrong
   (reads CONTENT) AND bc_on < bc_off (K/V helps at all).** Requires `BATCH_SIZE=2`.
2. **Clean launcher** `run_kv_clean.sh` — every value HARD-SET (no `${VAR:-}` layering, no accreted overrides),
   so what's written is what runs.
3. **KV-aware eval** `eval_kv.sh` — no `garment_adapter.pt` requirement; loads `garment_kv_branch.pt` +
   `garment_kv_procs.pt`; renders baseline / KV-correct / **KV-WRONG** / KV-bypass. The WRONG condition
   (`GARMENT_KV_DEPLOY_WRONG`, forward.py:~1035) swaps the K/V branch's garment_latent to a different
   reserved id → **deployed, image-space `garL1(correct)` vs `garL1(wrong)` specificity check.**
4. **σ-gated contrastive** (`CONTRAST_SIGMA_MIN=0.6`, forward.py:~1363) — the credit only trains at σ≥0.6
   where C_t doesn't leak the GT garment; the model still trains across all σ. Fixes the low-σ leak.
5. **Pure starvation** (`GARMENT_SLOT_CORRUPT_P=1.0`) — the standard garment+rough slot is zeroed EVERY step.

NOTE on metrics: `bc_on`=gar(correct), `bc_off`=gar(NO-KV / bypass), `kvc_wrong`=gar(wrong garment). There is
no separate "zero garment encoded through K/V" metric — `bc_off` is the K/V-OFF (bypass) baseline, which is
the meaningful comparison. The training contrastive is garment-token velocity-MSE (cheap gradient); the
DEPLOYED wrong eval is the image-space (final-pixel) garL1 arbiter.

## Config (all explicit in run.sh)
pure USE_GARMENT_KV append @ blocks 44,50,55,59 (gate init 0.0 = real contribution) + GLOBAL LoRA co-train
(FREEZE_LORA=0, EARLY_BLOCK_CUTOFF=-1, LR 3e-5) + garment/rough zero-starvation (P=0.9) + branch-credit
(zero contrastive, W=3) + WRONG-garment contrastive (W=5) + image losses ON (PURE_LATENT=0, protects bg/skin)
+ MULTI-GARMENT (full 11647, no memorization) + BATCH_SIZE=2.

## To run (when you're ready)
`RUN_NAME=KV_CLEAN_$(date +%H%M%S) TIME_BUDGET=7200 RESDIR=bgsmearresearch/KV_CLEAN bash run_kv_clean.sh`
then `bash eval_kv.sh runs/KV_CLEAN_<...> bgsmearresearch/KV_CLEAN/kveval` and VIEW IDENTITY_panel.png.

## Success criterion
Training: `bc_on` consistently BELOW `kvc_wrong` (correct garment beats wrong = reads content) by a
MEANINGFUL margin (not ~0.1% noise), with `bc_off` also above `bc_on` (K/V helps), bg/skin held (image
losses). Eval: KV-correct garment closer to GT than baseline/bypass, bg/skin clean.

## Optional next (only if the first run is ambiguous)
- image-space contrastive IN TRAINING (decode correct+wrong garment crops, σ-gated so cost is bounded) —
  currently the training gradient is latent-velocity; the deployed wrong eval already gives the image-space check.
- caveat: `GARMENT_KV_DEPLOY_WRONG` uses `_load_wrong_adapter` (a fixed wrong reserved id per sample); fine at
  eval batch size 1. If eval batch is raised, the wrong-garment pick becomes batch-shared / less clean.

## Judging criteria (user)
Training: **bc_on < kvc_wrong** (reads content) AND **bc_on < bc_off** (K/V helps), by a meaningful margin.
Deployed (eval_kv panel): **garL1(KV-CORRECT) < garL1(KV-WRONG)** and/or **< garL1(KV-BYPASS)**, bg/skin (CLOUD/skinL1) NOT worse than baseline.

**Verdict:** validated; RUNNING (user green-lit).
