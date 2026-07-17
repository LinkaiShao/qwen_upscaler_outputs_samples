"""WIRING.py — VERBATIM snapshot of the live trainlib hook blocks for the MASKED-LoRA run.
This is a READ-ONLY record of how run.sh's env flags are wired into the ROOT executed path
(trainlib/run.py, trainlib/forward.py, masked_lora.py). The LIVE path is at ROOT; this file
documents it so the run is self-contained. Guard flag: USE_MASKED_LORA=1 (default off elsewhere).

──────────────────────────────────────────────────────────────────────────────
1) trainlib/run.py — install the wrapper right after the LoRA freeze block:
──────────────────────────────────────────────────────────────────────────────
    if int(os.environ.get("USE_MASKED_LORA", "0")):
        from masked_lora import install_masked_lora
        _n_ml = install_masked_lora(transformer)
        log.info(f"MASKED LoRA (spatial gating) on {_n_ml} LoRA Linear layers")

──────────────────────────────────────────────────────────────────────────────
2) trainlib/forward.py — build the per-token gate right after `hidden` is assembled
   (M length == hidden.size(1); 1 only on C-slot garment tokens, 0 on C-slot face/hands/bg
   + ALL conditioning slots). Static in token positions -> set once per forward-batch:
──────────────────────────────────────────────────────────────────────────────
    hidden = torch.cat([C_p_] + cond_seq, dim=1)   # (B, 3072 * NUM_SLOTS, 64)
    img_shapes = ...
    if int(os.environ.get("USE_MASKED_LORA", "0")):
        import garment_adapter as _ga_ml
        _cgate = _ga_ml.mask_tok(batch, device, weight_dtype, B, C, H, W)   # (B, N_C, 1) SOFT fraction
        # BINARIZE -> strict {0,1} (BUGFIX): mask_tok is a per-token FRACTION; boundary tokens have
        # Md in (0,0.5] so `delta *= Md` left NON-ZERO LoRA on them (runtime VERIFY bg=38.25). Hard
        # threshold => bg (M==0) delta EXACTLY 0; boundary folded to bg (protected -> kills the halo).
        _cgate = (_cgate > float(os.environ.get("MASKED_LORA_THRESH", "0.5"))).to(weight_dtype)
        _n_c = _cgate.size(1); _n_full = hidden.size(1)
        if _n_full > _n_c:
            _pad = torch.zeros(B, _n_full - _n_c, 1, device=device, dtype=weight_dtype)
            state._MASKED_LORA_MASK = torch.cat([_cgate, _pad], dim=1)       # (B, N_full, 1)
        else:
            state._MASKED_LORA_MASK = _cgate
    else:
        state._MASKED_LORA_MASK = None

──────────────────────────────────────────────────────────────────────────────
3) masked_lora.py — the wrapper (see the sibling masked_lora.py snapshot). Each PEFT LoRA
   Linear's forward becomes:  X_new = base_layer(X) + ΔW(X) * M   (vanilla path mirrors
   peft==0.18.1 lora.Linear.forward exactly; falls back to the original forward for
   disabled/merged/mixed-batch/variant paths). _get_mask returns None for the text stream
   (seq-len != M's len) => text-stream LoRA stays full-strength (it processes the prompt).
   One-time VERIFY print (MASKED_LORA_DEBUG=1): max|ΔW·M| on bg tokens (M==0) MUST be 0.0.

──────────────────────────────────────────────────────────────────────────────
4) trainlib/state.py — singleton:
──────────────────────────────────────────────────────────────────────────────
    _MASKED_LORA_MASK = None   # (B, N_full, 1) per-token gate; LoRA delta *= M

CPU unit-test proof (a real PEFT LoRA linear + install_masked_lora + a garment gate):
    max|ΔW·M| bg(M==0) = 0.000e+00 ; garment(M==1) = 8.319e+00 ;
    garment-region delta == full unmasked LoRA delta (True) ; text-stream (len 77) unmasked.
"""
