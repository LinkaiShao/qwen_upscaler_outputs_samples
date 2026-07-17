"""masked_lora.py — SPATIAL LoRA GATING (USE_MASKED_LORA=1).

Physically quarantine the tryon LoRA to the garment region. The tryon LoRA is a global
PEFT adapter whose delta ΔW(X) is added to every image token — so a co-trained LoRA can
(and did, Runs 2/3) leak identity paint into face/hands/bg as halos. Here we wrap each
PEFT LoRA Linear so its forward is:

    X_new = base_layer(X) + ΔW(X) * M

where M (state._MASKED_LORA_MASK) is a per-token gate, length = the FULL image sequence
(3072 * NUM_SLOTS), that is 1 ONLY on the C-slot garment tokens (warped-mask token gate)
and 0 everywhere else — the C-slot face/hands/bg tokens AND all conditioning slots. Outside
the mask ΔW contributes EXACTLY 0, so the LoRA is mathematically forbidden from altering
face/hands/bg. Inside the mask it keeps full capacity to learn garment identity.

Only IMAGE-stream Linear layers are masked (matched by seq-len == M's len). Text-stream LoRA
layers (encoder_hidden_states projections, a different seq len) get no mask -> full LoRA,
since they process the prompt, not the output image.

Guarded: install_masked_lora is only called when USE_MASKED_LORA=1; otherwise every run is
byte-identical (PEFT forward untouched). Mirrors peft==0.18.1 lora.Linear.forward exactly for
the vanilla path; falls back to the original forward for disabled/merged/mixed-batch/variant
paths so behaviour is preserved whenever masking cannot apply.
"""
import os
import torch
from trainlib import state

_INSTALLED = False
_DBG_COUNT = 0                 # masked-path VERIFY prints emitted
_FALLBACK_SEQS = set()         # seq-lens that hit the shape-mismatch (unmasked) fallback — should be text-stream only


def _get_mask(x):
    """Return the per-token gate M for input x, or None (=> unmasked full LoRA)."""
    M = getattr(state, "_MASKED_LORA_MASK", None)
    if M is None:
        return None
    # only the image stream matches (B, N_full, .); text stream / any other length -> unmasked.
    if x.dim() != 3 or x.shape[0] != M.shape[0] or x.shape[1] != M.shape[1]:
        if x.dim() == 3 and int(os.environ.get("MASKED_LORA_DEBUG", "1")) and x.shape[1] not in _FALLBACK_SEQS:
            _FALLBACK_SEQS.add(x.shape[1])
            print(f"[masked_lora] fallback (UNMASKED full LoRA) on seq_len={x.shape[1]} "
                  f"(M spans {M.shape[1]}) -> should be TEXT stream only", flush=True)
        return None
    return M


def _is_lora_linear(mod):
    return (hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base_layer")
            and isinstance(getattr(mod, "lora_A", None), torch.nn.ModuleDict))


def _make_masked_forward(layer, orig_forward):
    def masked_forward(x, *args, **kwargs):
        global _DBG_COUNT
        # fall back to the exact original forward whenever masking cannot / should not apply.
        if (getattr(layer, "disable_adapters", False) or getattr(layer, "merged", False)
                or kwargs.get("adapter_names", None) is not None or len(args) > 0):
            return orig_forward(x, *args, **kwargs)
        M = _get_mask(x)
        if M is None:
            return orig_forward(x, *args, **kwargs)
        layer._check_forward_args(x, **kwargs)
        result = layer.base_layer(x, **kwargs)
        torch_result_dtype = result.dtype
        lora_A_keys = layer.lora_A.keys()
        delta = None
        for active_adapter in layer.active_adapters:
            if active_adapter not in lora_A_keys:
                continue
            if active_adapter in getattr(layer, "lora_variant", {}):
                # non-vanilla LoRA variant present -> preserve exact behaviour, no masking.
                return orig_forward(x, *args, **kwargs)
            lora_A = layer.lora_A[active_adapter]
            lora_B = layer.lora_B[active_adapter]
            dropout = layer.lora_dropout[active_adapter]
            scaling = layer.scaling[active_adapter]
            xc = layer._cast_input_dtype(x, lora_A.weight.dtype)
            d = lora_B(lora_A(dropout(xc))) * scaling
            delta = d if delta is None else delta + d
        if delta is not None:
            Md = M.to(delta.dtype)
            delta = delta * Md
            # ── bg-quarantine VERIFY (max|ΔW·M| on bg tokens must be 0) — first N wrapped image-stream
            #    layers so the smoke confirms bg==0 on ALL of them, not just the first. ──
            if _DBG_COUNT < int(os.environ.get("MASKED_LORA_DEBUG_N", "8")) and int(os.environ.get("MASKED_LORA_DEBUG", "1")):
                _DBG_COUNT += 1
                with torch.no_grad():
                    m0 = (Md <= 0.0)           # strict bg (binary mask -> exactly 0): face/hands/bg/conditioning
                    m1 = (Md > 0.0)            # garment region
                    bg_max = float(delta[m0.expand_as(delta)].abs().max()) if m0.any() else float("nan")
                    fg_max = float(delta[m1.expand_as(delta)].abs().max()) if m1.any() else float("nan")
                    print(f"[masked_lora] VERIFY#{_DBG_COUNT} seq={x.shape[1]} out={delta.shape[-1]}  "
                          f"max|deltaW*M| bg(M==0)={bg_max:.3e}  garment(M>0)={fg_max:.3e}  "
                          f"n_bg_tok={int(m0[0,:,0].sum())} n_gar_tok={int(m1[0,:,0].sum())}  "
                          f"(bg MUST be 0.0)", flush=True)
            result = result + delta
        return result.to(torch_result_dtype)
    return masked_forward


def install_masked_lora(transformer):
    """Wrap every PEFT LoRA Linear so its delta is gated by state._MASKED_LORA_MASK. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return 0
    n = 0
    for mod in transformer.modules():
        if _is_lora_linear(mod) and not getattr(mod, "_masked_lora_wrapped", False):
            mod._orig_forward_pre_mask = mod.forward
            mod.forward = _make_masked_forward(mod, mod._orig_forward_pre_mask)
            mod._masked_lora_wrapped = True
            n += 1
    _INSTALLED = True
    print(f"[masked_lora] installed spatial LoRA gating on {n} LoRA Linear layers "
          f"(delta *= M; M=1 only on C-slot garment tokens)", flush=True)
    return n
