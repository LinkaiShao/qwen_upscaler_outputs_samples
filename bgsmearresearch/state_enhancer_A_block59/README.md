# Run A — Final-block state-aware enhancer (block 59)

**Thesis.** Not "garment net predicts garment" but "garment net EDITS what run37 is already drawing."
After the LAST transformer block, cross-attend run37's own garment-region target tokens (Q) to garment-encoder
features (K/V) and add a zero-init, mask-gated residual:

```
G      = QwenEncoder([warped_rgb, garment_latent, warped_mask])      # garment detail features
Delta  = CrossAttn(Q = H_59[:, :N_C], K = G, V = G)                   # H_59 = run37 hidden after block 59
H_59'  = H_59 + M_garment * ZeroOut(Delta)                            # out_proj zero-init; M gates to garment only
```

**Why block 59 (safest POC).** The first N_C tokens are the noised target latent run37 denoises (velocity =
`_o[:, :N_C]`). Editing them after the final block means there is NO downstream self-attention to spread garment
content into skin/bg — the correction flows only into final norm/proj → velocity. bg-smear risk minimized.

**Guarantees (CPU-verified).** out_proj zero-init → `max|delta|=0.0` at init → step0 == run37 exactly.
M_garment (warped-mask gate) → delta is EXACTLY 0.0 on non-garment tokens → bg/skin untouched by construction.

**Recipe.** Warm-start run37 (BGBAND37_012549). Co-train LoRA (FREEZE_LORA=0, so branch-off = trained-LoRA-alone
bar) + enhancer (727M, GARMENT_ADAPTER_LR=1e-4). 50/30/10/10 starvation schedule (GARMENT_SLOT_CORRUPT=zero →
under forced-handoff the enhancer is the ONLY garment path). Pure-latent flow recipe (img losses off, as chain).
2h full-data. Launch: `ENH_BLOCK=59 RESDIR=bgsmearresearch/state_enhancer_A_block59 bash run_state_enhancer.sh`.

**Acceptance test (user).**
1. CORRECT enhancer improves garment detail over branch-off (LoRA-alone).
2. WRONG enhancer does NOT get the same improvement.
3. Branch-off vs correct shows improvement from the ENHANCER, not just co-trained LoRA.
4. CLOUD/skin do not regress vs baseline.
5. Panel shows ACTUAL detail improvement — not green speckle or generic sharpening.
6. EXTRA: deploy with standard garment slot ZEROED + enhancer on — if correct still beats wrong there, the
   enhancer genuinely carries identity.

**Verdict:** _pending run._
