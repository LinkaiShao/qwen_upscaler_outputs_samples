# run34 — COMBINED: all 3 region-owned adapters. SOAR init, RAW single model, no composition.

The deliverable: **garment mask-gate (7.5e-5) + SKIN adapter (311M) + BG adapter (236M)**, all region-owned paths on
one shared Qwen denoiser. Garment writes into garment tokens (mask-gate), skin adapter writes into skin tokens only,
bg adapter writes into bg tokens only (early/mid blocks). Three separate routed residual paths so garment/skin/bg each
OWN their write and do NOT fight through a shared path.

GOAL: in-loop [DEPLOY] GARMENT + SKIN + BG **all improve together** vs SOAR/run31, ideally beating run32 (skin-only best).

## Code / config
- Launcher: `../../run_bgband34.sh`
- Adapter modules: `../../skin_adapter.py` + `../../bg_adapter.py`
- Env: `USE_MASK_GATED_INJECTION=1` + `USE_SKIN_ADAPTER=1 SKIN_ADAPTER_LR=1e-4 SKIN_ADAPTER_TARGETS="12,20,28,36,44,50,55,59"` + `USE_BG_ADAPTER=1 BG_ADAPTER_LR=1e-4 BG_ADAPTER_TARGETS="4,8,12,16,20,28"` + `MULTI_BLOCK_LR=7.5e-5`
- Output weights (on completion): `runs/BGBAND34_121343/final/` (both `skin_adapter.pt` + `bg_adapter.pt` + tryon_lora + v6_heads + multi_block_injection)
- Log: `logs/BGBAND34_121343.log`

## Reference (prior best) — head-to-head target
| run | GARMENT | SKIN | BG | BAND |
|---|---|---|---|---|
| run31 base | 14.48 | — | — | — |
| run32 skin-only (BEST) | 15.27 | 12.45 | 4.07 | -1.56 |
| run33 bg-only | 23.65 | 11.45 | 6.77 | -2.96 |

## In-loop [DEPLOY]
_RUNNING (BGBAND34_121343, started 12:13:43, frees GPU ~14:13). Results filled in on completion._

## Verdict
_pending — must beat run32 on all 3 to confirm the combined region-owned architecture._
