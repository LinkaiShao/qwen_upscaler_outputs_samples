# run33 — region-owned BG adapter. SOAR init, RAW single model, no composition.

Adds a **region-owned bg adapter** (236M params, 6 sites @ EARLY/MID blocks 4,8,12,16,20,28) on the 7.5e-5 mask-gated
garment base. Light per-block cross-attn writes a bg residual INTO **bg tokens ONLY** (gated by `parse_bg` token mask),
sourced from the real surrounding bg (agnostic latent). Targets early/mid blocks because bg is low-frequency and
resolved early in the denoiser. **NO skin adapter** here — isolate the bg lever. Gates init 0.01 → ramp.

## Code / config
- Launcher: `../../run_bgband33.sh`
- Adapter module: `../../bg_adapter.py` (`BgAdapter` + `install_bg_adapter_hooks`)
- Wiring: `trainlib/state.py` (`_BG_ADAPTER/_BG_HOOKS/_BG_HOLDER`), `trainlib/run.py` (instantiate+hooks+param group @ `BG_ADAPTER_LR=1e-4`), `trainlib/forward.py` (holder: bg_src=agnostic bg + bg_mask=parse_bg + N_C)
- Env: `USE_BG_ADAPTER=1 BG_ADAPTER_LR=1e-4 BG_ADAPTER_TARGETS="4,8,12,16,20,28"` + `USE_MASK_GATED_INJECTION=1` + `MULTI_BLOCK_LR=7.5e-5`
- Output weights: `runs/BGBAND33_100927/final/` (`bg_adapter.pt` + `bg_adapter_targets.txt` + tryon_lora + v6_heads + multi_block_injection)
- Log: `logs/BGBAND33_100927.log`

## In-loop [DEPLOY] (FAITHFUL)
| step | GARMENT | SKIN | BG | BAND_OFFSET |
|---|---|---|---|---|
| 375 (gates ~0) | 28.15 | 13.45 | 9.78 | -4.13 |
| 724 (ramped) | 23.65 | 11.45 | **6.77** | **-2.96** |

## Verdict: bg adapter ENGAGES, but weaker overall than run32
As gates ramp: **BG 9.78→6.77, BAND -4.13→-2.96 toward target -1** (bg adapter working directionally); all metrics
improving. BUT weaker trajectory than run32 at the same step — run32 s747 (GAR 15.27/SKIN 12.45/BG 4.07/BAND -1.56)
beats run33 s724 (GAR 23.65/SKIN 11.45/BG 6.77/BAND -2.96). run33 garment notably higher (23.65 vs 15.27) and bg worse
than run32 — likely early-block bg writes perturbing shared features, or SOAR run-to-run variance.
→ bg adapter works, but skin adapter (run32) gave the cleaner all-3. Test both together in combined run34.
