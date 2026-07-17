# maskedlora_latenttoken — SPATIAL LoRA GATING

**Hypothesis.** Runs 2/3 produced a REAL garment signal (correct beats wrong on garL1) but the
**global** tryon LoRA leaked identity paint into face/hands/bg → halos. Fix: physically
**quarantine the LoRA to the garment region** so it is mathematically forbidden from altering
anything outside it.

**Mechanism.** Wrap every PEFT LoRA `Linear` so its update is masked:

```
X_new = base_layer(X) + ΔW(X) · M
```

`M` = per-token gate over the full image sequence (`3072 · NUM_SLOTS`), = the warped-mask C-slot
garment token gate on the first `N_C` tokens, **0** on the C-slot face/hands/bg tokens **and** all
conditioning slots. Outside `M`, `ΔW` contributes **exactly 0**. Inside `M`, full LoRA capacity to
learn identity. Text-stream LoRA layers (different seq-len) are left full-strength (prompt path).

**Arch.** Run3's `latent_token` garment adapter (`GARMENT_ADAPTER_MODE=latent_token`, zero-init,
step0 == run37) + the 50/30/10/10 schedule + asymmetric starvation of the standard garment slot
(`GARMENT_SLOT_CORRUPT=zero`). **Co-train the now-quarantined LoRA** (`FREEZE_LORA=0`) with the
adapter. **Image losses ON** (`PURE_LATENT=0`) because the halo is a pixel-space artifact and must
be decoded to be supervised; in-loop `DEPLOY_HALO_EVAL` watches it directly.

**Files (self-contained snapshot).**
- `model.py`      = root `garment_adapter.py` (latent_token adapter).
- `masked_lora.py`= root `masked_lora.py` (the LoRA-gating wrapper).
- `run.sh`        = root `run_maskedlora_latenttoken.sh` (launcher).
- `WIRING.py`     = verbatim trainlib hook blocks (run.py install + forward.py mask-set).

**Guard.** `USE_MASKED_LORA=1`. Default off → every other run byte-identical.

**BUGFIX (soft-gate leak).** First runtime VERIFY showed `bg(M==0)=38.25` (should be 0): `mask_tok`
returns a per-token FRACTION, so boundary tokens had `Md∈(0,0.5]` and `delta*=Md` left non-zero LoRA
on them. Fix = **binarize M to strict {0,1}** in forward.py (`_cgate = (_cgate > MASKED_LORA_THRESH).float()`,
default 0.5). Boundary folds to bg (protected → kills the halo); true bg is EXACTLY 0.

**Verified (CPU, mirrors runtime shape 12288 tok / 292 garment, no GPU — Run8 owns GPU0).** Real PEFT
LoRA + `install_masked_lora` + soft→binarized mask: `max|ΔW·M|` bg(M==0) = **0.000e+00** on BOTH
wrapped image-stream layers (out=3072 and out=512), garment(M>0) full-strength, text-stream (len 77)
falls back unmasked. GPU re-smoke command ready; run once Run8 frees GPU0:
`RUN_NAME=MLORA_SMOKE2 TIME_BUDGET=150 MASKED_LORA_DEBUG=1 nohup bash run_maskedlora_latenttoken.sh > bgsmearresearch/maskedlora_latenttoken/smoke2.out 2>&1 &`

**REJECT if:** halo (deploy bg/face dL) not reduced vs Run2/3, OR garment signal lost
(correct no longer beats wrong/branch-off) — i.e. the mask starved the identity path.
