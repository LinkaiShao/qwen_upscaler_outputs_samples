# Run B — Block-55 state-aware enhancer

Identical architecture to Run A (see `../state_enhancer_A_block59/README.md`) but injects the
cross-attention enhancer **after block 55** instead of 59.

```
Delta = CrossAttn(Q = H_55[:, :N_C], K = V = QwenEncoder([warped_rgb, garment_latent, warped_mask]))
H_55' = H_55 + M_garment * ZeroOut(Delta)
```

**What it tests.** Blocks 56..59 of run37's self-attention remain downstream of the edit, so run37 gets a
few blocks to INTERPRET / integrate the garment correction into the drawing. The question: do those downstream
blocks make the correction more useful (better identity) — or do they re-introduce bg/skin speckle by spreading
garment content outward?

**REJECT if** bg/skin speckle returns (CLOUD/farBG/skinL1 regress vs baseline, or panel shows speckle).
Run ONLY after Run A shows the enhancer carries identity at all.

**Launch:** `ENH_BLOCK=55 RESDIR=bgsmearresearch/state_enhancer_B_block55 bash run_state_enhancer.sh`

**Verdict:** _pending — gated on Run A._
