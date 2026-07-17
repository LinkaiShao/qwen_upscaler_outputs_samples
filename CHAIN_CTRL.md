# CHAIN_CTRL — tonight's 6 garment-net runs (sequential, FULL-DATA 2h each)

All share the UNIFIED adapter (`garment_adapter.py`, mode-selected) + common protocol: aligned garment
(warped_rgb+garment+mask) always CLEAN into the adapter; asymmetric starvation of the STANDARD garment slot
(GARMENT_SLOT_CORRUPT=zero); per-step 50/30/10/10 schedule; warm-start run37; zero-init adapter (step0==run37);
FULL data (11647), TIME_BUDGET=7200. Each run has a self-contained folder `bgsmearresearch/<dir>/`
(model.py + run.sh + WIRING.py + README.md).

## RUNNING
- **Run1 — Input Adapter, FROZEN run37** — LAUNCHED
  - RUN_NAME=**RUN1_INPUTADAPTER_0714_2326**  PID=**1774730**  log=**bgsmearresearch/run_inputadapter_frozen/RUN1_INPUTADAPTER_0714_2326.log**
  - ETA ~2h (~12 s/step, ~600 steps). Smoke passed: full-data, adapter_gradnorm>0, lora_gradnorm=0 (frozen), 0 errors.

## PENDING (built + folders ready; SMOKE each ~150s before its 2h, then launch)
Order = Run2, Run3, Run4, Run5, Run6.

| # | dir | launcher | mode | notes |
|---|-----|----------|------|-------|
| 2 | inputadapter_lora   | run_inputadapter_lora.sh   | input_hidden  | + LoRA co-train (LR 5e-6), v6 frozen |
| 3 | latenttoken_adapter | run_latenttoken_adapter.sh | latent_token  | 64-d packed inject in _fwd, +LoRA |
| 4 | learned_detail_slot | run_learned_detail_slot.sh | detail_slot   | replace garment slot, +LoRA |
| 5 | spatial_adaln       | run_spatial_adaln.sh       | spatial_adaln | FiLM @12,20,28,36, +LoRA |
| 6 | early_controlnet    | run_early_controlnet.sh    | controlnet    | residual @8,16,24,32, +LoRA |

### Per-run commands (X = launcher, DIR = folder)
SMOKE (~150s, verify 0 errors + adapter_gradnorm>0):
```
RUN_NAME=<TAG>_SMOKE TIME_BUDGET=150 POSE_DEBUG=1 ADAPTER_LOG=bgsmearresearch/<DIR>/smoke.log bash <X>
grep -E "garment_adapter: mode|adapter_dbg" bgsmearresearch/<DIR>/smoke.log ; grep -c Error bgsmearresearch/<DIR>/smoke.log
```
2h LAUNCH (single nohup, log in the folder):
```
TS=$(date +%m%d_%H%M); RN=<TAG>_$TS
RUN_NAME=$RN TIME_BUDGET=7200 ADAPTER_LOG=bgsmearresearch/<DIR>/$RN.log nohup bash <X> > bgsmearresearch/<DIR>/nohup.out 2>&1 &
```
EVAL (after ckpt `runs/$RN/final/garment_adapter.pt`): render 4 conditions — baseline(run37) / correct / wrong /
**branch-off = run37's OWN output** (set state._GARMENT_ADAPTER_BYPASS via GARMENT_ADAPTER_BYPASS=1 env in the render);
5-panel + garL1/edge. REJECT if correct/wrong/branch-off look identical or bg/skin degrade.

## DONE
- (none yet)

## CRITICAL EVAL FIX honored: branch-off = run37 velocity/output (state._GARMENT_ADAPTER_BYPASS -> hooks return run37 unchanged), NOT adapter-output=0.
