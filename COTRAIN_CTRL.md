# CO-TRAIN CONTROL — run37 LoRA + OOTD garment reference net (FULL data, 2h, PURE-LATENT)

STATUS: **RUNNING**  (relaunched PURE-LATENT 2026-07-14)

## Process
- wrapper: `run_cotrain_2h.sh` — PID + RUN_NAME in `/tmp/cotrain_2h.out`
- train log: `logs/COTRAIN_<mmdd_HHMM>.log`
- ckpt out: `runs/COTRAIN_<...>/final/` (tryon_lora.safetensors + garment_branch.pt + ootd_injectors.pt)
- results dir: `bgsmearresearch/cotrain_ootd_FULLDATA_2h/` ; DONE marker: `.../COTRAIN_DONE.txt`

## PURE-LATENT change (this relaunch)
- `PURE_LATENT=1`: the per-step VAE decode of the model prediction is SKIPPED (latent.py). ALL image-space
  losses no-op (region/edge/garment_adv/background early-return; total.py None-coalesces the sum + metrics).
  KEPT (latent, supervise garment detail): W_FLOW, LAMBDA_RECON, region-pct in LATENT mode (RPCT_IMG_W=0),
  USE_FLOW_BG_SPLIT, LAMBDA_V6_ROUTE. Also off: IMG_LOSS_WEIGHT, USE_V6_IMG_STAGE, LAMBDA_V6_REPAIR,
  LAMBDA_PERCEPTUAL, all W_IMG_V6_*, W_GARMENT_EDGE_MATCH, DEPLOY_HALO_EVAL.

## Recipe (unchanged clean co-train; run69 destabilizers out)
- Warm-start LoRA from run37; FREEZE_LORA=0 (co-train) + gnet trainable; FREEZE_V6=1; USE_PCGRAD=1.
- OOTD K/V garment reference at blocks 36,44,50,55; to_v_g zero-init; warped-mask gate; CLEAN garment input.
- NO CFG_DROPOUT / garment-slot corruption / HF-bridge soup. FULL data (11647 ids), BATCH_SIZE=1, grad_accum=1.

## Smoke verified (pure-latent, before launch)
- 0 errors / no NaN.
- BOTH grads nonzero: LoRA lora_gradnorm=1 (480 params) + gnet to_v_g.gradnorm=0.018 (co-train intact).
- flow/region stable: rpct bundles G~0.06 S~0.04 B~0.09, total~0.97-1.08 (not exploding).

## IMPORTANT — the decode was NOT the bottleneck
- pure-latent steady step ~= 24 s/sample (batch 2 = ~49 s/step) vs 27 s/step with the decode.
- The decode was only ~3 s. Real cost = PCGrad (1 backward + 4 autograd.grad = 5x gradient passes) each step.
- => pure-latent is ~10-12% faster (~300 steps in 2h vs ~267), NOT 5-10x. To get 5-10x more steps the lever is
  PCGrad (drop / fewer tasks) or model size, NOT the decode. Kept PCGrad ON per instruction (run37 stability).

## Eval plan (auto after train)
Deploy render on 5 reserved held-out ids: run37 BASELINE vs co-train CORRECT vs co-train WRONG (OOTD_DEPLOY_WRONG)
-> metric_rawpred.py garL1 + IDENTITY_panel.png [run37 | correct | WRONG | GT].
WIN iff garL1(correct) < garL1(baseline) AND garL1(correct) < garL1(wrong).

## Poll
- tail bgsmearresearch/cotrain_ootd_FULLDATA_2h/numbers.txt
- tail /tmp/cotrain_2h.out   (RUN_NAME + progress)
- cat bgsmearresearch/cotrain_ootd_FULLDATA_2h/COTRAIN_DONE.txt   (done?)
