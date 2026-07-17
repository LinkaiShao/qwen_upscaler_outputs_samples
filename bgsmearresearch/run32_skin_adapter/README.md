# run32 — region-owned SKIN adapter. SOAR init, RAW single model, no composition.

Adds a **region-owned skin adapter** (311M params, 8 sites) on top of run31's GOOD 7.5e-5 mask-gated garment base.
A light per-block cross-attn (`GarmentCrossAttnNoGate`) writes a skin residual INTO **skin tokens ONLY** (gated by
`parse_skin` token mask), sourced from the person's own body/agnostic skin latent. Garment writes into garment tokens
(mask-gate), skin writes into skin tokens (adapter) — **separate routed residual paths** so skin recovers WITHOUT skin
gradients rewriting the bg/garment path. Adapter gates init 0.01 → ramp over training.

## Code / config
- Launcher: `../../run_bgband32.sh`
- Adapter module: `../../skin_adapter.py` (`SkinAdapter` + `install_skin_adapter_hooks`)
- Wiring: `trainlib/state.py` (`_SKIN_ADAPTER/_SKIN_HOOKS/_SKIN_HOLDER`), `trainlib/run.py` (instantiate+hooks+param group @ `SKIN_ADAPTER_LR=1e-4`), `trainlib/forward.py` (holder: skin_src + skin_mask + N_C)
- Env: `USE_SKIN_ADAPTER=1 SKIN_ADAPTER_LR=1e-4 SKIN_ADAPTER_TARGETS="12,20,28,36,44,50,55,59"` + `USE_MASK_GATED_INJECTION=1` (garment) + `MULTI_BLOCK_LR=7.5e-5`
- Output weights: `runs/BGBAND32_080449/final/` (`skin_adapter.pt` + `skin_adapter_targets.txt` + tryon_lora + v6_heads + multi_block_injection)
- Log: `logs/BGBAND32_080449.log`

## In-loop [DEPLOY] (FAITHFUL — hooks applied; eval_full is LoRA-only/unfaithful for adapters)
| step | GARMENT | SKIN | BG | BAND_OFFSET |
|---|---|---|---|---|
| 383 (gates ~0) | 21.65 | 14.34 | 5.53 | -1.83 |
| 747 (ramped) | **15.27** | **12.45** | **4.07** | **-1.56** |

Baselines: run31 garment 14.48; run31b (5e-5) DIVERGED garment 38.47.

## Verdict: POSITIVE
As skin-adapter gates ramp, **ALL THREE improve together**: SKIN 14.34→12.45 (adapter engaging), GARMENT 21.65→15.27
(recovers toward run31 14.48 — NO divergence), BG 4.07/BAND -1.56 (healthy, near target -1). The region-owned skin
write does NOT break garment/bg. **First run with garment~15 + skin~12.5 + bg~4 all healthy at one deploy.**
→ keep skin adapter for the combined run34.
