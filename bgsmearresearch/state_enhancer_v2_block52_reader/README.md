# state_enhancer_v2_block52_reader

**Thesis.** Can the main transformer learn to be a *reader* for the garment net — pulling real garment detail from the block-52 state_enhancer — instead of the old LoRA solving the garment alone at low sigma (where the noised target latent `C_t` still leaks GT garment identity and lets the main net cheat)? We move the enhancer to an EARLY-ish block (52) so blocks 53-59 can interpret it, freeze the 0-51 LoRA and co-train only 52-59, and force the handoff by (a) starving the target latent at mid/high sigma and (b) quarantining bg/skin from reading garment tokens so the enhancer's contribution can't smear.

**The 5 changes (all env-gated; existing runs byte-identical when unset):**
1. **Schedule moved before sigma** (`STARVE_HIGH_SIGMA=1`): the starve/branch decision is made BEFORE sigma sampling so a starve step can be noised high.
2. **Mid/high-sigma starvation** (`STARVE_SIGMA_LO=0.5`): starve steps resample `sigma ~ U[0.5, 1.0]`, corrupting GT-garment detail in `C_t` across the range where it would otherwise leak. This IS the "corrupt garment detail in C_t", done cleanly via noise.
3. **60/25/10/5 schedule** (`ENH_SCHED_V2=1`): branch-only / both / base-solo / both-off (vs the old 50/30/10/10).
4. **Attention quarantine** (`USE_ENH_QUARANTINE=1`, blocks 53-59): a (query,key)-pair additive attn bias (-1e4) blocks bg/skin C-queries from attending garment C-keys, so injected garment features can't smear downstream. Only applied when the enhancer is active (`garment_adapter._active()`), so base-solo/branch-off stay == run37.
5. **Block-52 reader** (`GARMENT_ADAPTER_BLOCKS=52`, `EARLY_BLOCK_CUTOFF=51 LR_EARLY_MULT=0.0`): enhancer injects at block 52; the 0-51 LoRA is frozen (LR 0.0) and 52-59 co-train at `args.lr` so the late stack learns to read the enhancer.

**Files.** `model.py` = copy of `garment_adapter.py` (the state_enhancer mode is unchanged — this experiment EXTENDS it, it does not rewrite it). `enh_quarantine.py` = the quarantine bias + processor. `run.sh` = copy of `run_state_enhancer_v2.sh`. `WIRING.py` = verbatim snapshot of the new/changed hook blocks in `trainlib/forward.py`, `trainlib/run.py`, `trainlib/state.py`, each labeled with its source site.

**Acceptance test (verdict PENDING — not yet run for 1h).**
- `correct` garment garL1 < `branchoff` garL1 (the enhancer helps beyond the frozen base);
- `wrong` garment does NOT drop like `correct` (real identity read, not a memorized average);
- bg/skin do NOT regress vs run37 (quarantine holds; no smear);
- panels show REAL garment detail (not sludge/mean).
