"""enh_quarantine.py — ENH v2 (B): attention QUARANTINE for the block-52 state_enhancer.

Goal: on the blocks downstream of the block-52 garment injection (default 53-59), bg/skin
QUERY tokens among the first N_C (the noised target tokens run37 denoises) must NOT attend to
GARMENT KEY tokens among the first N_C. That way the learned garment features the enhancer
edits into the garment tokens cannot smear into bg/skin via later self-attention.

Mechanism — the STANDARD (query,key)-pair additive attention bias:
  bias[b, 0, q, k] = -1e4   iff  q in first-N_C AND gate(q)==0 (bg/skin/keep)
                                AND k in first-N_C AND gate(k)>0 (garment)
  bias = 0 everywhere else (all text tokens, all conditioning-slot tokens, garment queries,
  bg/skin keys). gate = the warped-garment per-token mask (deploy-available), NOT the GT mask.

A key-only bias was rejected: an additive per-key -1e4 on garment keys would ALSO block
garment queries from attending garment keys (self-consistency), which is wrong. The pair mask
is the correct object; at BATCH_SIZE 1-2 overfit a full (B,1,S,S) bf16 bias (~300MB/sample) is
memory-safe, so we use it. It is built ONCE per forward (build_quarantine_bias, stored in
state._ENH_QUARANTINE_HOLDER["bias"]) and shared across all quarantine blocks.

The processor only applies the bias when the enhancer is active (garment_adapter._active()), so
base-solo / branch-off / eval passes stay byte-identical to run37.
"""
import os, sys
import torch
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
from trainlib import state

NEG = -1e4


def build_quarantine_bias(M, N_C, seq_txt, total_img, device, dtype, neg=NEG):
    """Additive attention bias (B, 1, total_seq, total_seq).

    Args:
      M         : (B, N_C, 1) per-token garment gate (warped_mask fraction); >0.5 == garment.
      N_C       : number of target (C-slot) tokens (the first N_C image tokens).
      seq_txt   : number of text tokens prepended to the image tokens in the joint sequence.
      total_img : total image tokens across ALL slots (== hidden.size(1)).
    Returns:
      bias with `neg` at [bg/skin query in first-N_C, garment key in first-N_C], 0 elsewhere.
    """
    B = M.shape[0]
    total_seq = seq_txt + total_img
    gar = (M.reshape(B, N_C) > 0.5)                       # (B, N_C) garment tokens
    bg  = ~gar                                            # (B, N_C) bg/skin/keep (non-garment)
    pair = (bg[:, :, None] & gar[:, None, :])             # (B, N_C, N_C): bg-query -> garment-key
    block = torch.where(
        pair,
        torch.tensor(neg, device=device, dtype=dtype),
        torch.zeros((), device=device, dtype=dtype),
    )                                                     # (B, N_C, N_C)
    bias = torch.zeros(B, 1, total_seq, total_seq, device=device, dtype=dtype)
    qs = seq_txt
    bias[:, 0, qs:qs + N_C, qs:qs + N_C] = block          # text + conditioning slots stay 0
    return bias


class EnhQuarantineProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Adds the shared quarantine bias to attention_mask, then runs the base joint attention.
    Identity (== base processor) when no bias is present or the enhancer is inactive."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None, image_rotary_emb=None):
        import garment_adapter as _ga
        bias = getattr(state, "_ENH_QUARANTINE_HOLDER", {}).get("bias")
        if bias is not None and _ga._active():
            attention_mask = bias if attention_mask is None else attention_mask + bias
        return super().__call__(attn, hidden_states, encoder_hidden_states,
                                encoder_hidden_states_mask, attention_mask, image_rotary_emb)


def install_quarantine_processors(transformer, blocks):
    """Install EnhQuarantineProcessor on the attention of each block index in `blocks`.
    Returns the number of attention processors replaced."""
    _inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    tblocks = _inner.transformer_blocks if hasattr(_inner, "transformer_blocks") else _inner.blocks
    n = 0
    for b in blocks:
        blk = tblocks[b]
        for mod in blk.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = EnhQuarantineProcessor()
                n += 1
    return n
