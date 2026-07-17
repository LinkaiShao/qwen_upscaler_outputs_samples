"""run57 — Qwen garment K/V branch.

Interface (the DESIRED one, not residual injection):
    main image tokens (queries) attend to GARMENT MEMORY appended as extra K/V inside the
    frozen block's own joint attention.  NO `main_hidden += garment_residual`.

  garment_latent --patchify--> tokens
  run through copied REAL Qwen blocks WITH CORRECT ROPE (built from transformer.pos_embed,
    single 3072-token garment image) + real temb + text  -> garment features G   (stable:
    the run56 NaN came from image_rotary_emb=None; here rope is real)
  at selected main blocks, a GarmentKVAttnProcessor:
      Kg = to_k_g(G), Vg = to_v_g(G)                  (dedicated, trainable projections)
      joint_key   = cat([txt_key, img_key,   gate*Kg])
      joint_value = cat([txt_value, img_value, gate*Vg])
      joint_query unchanged  ->  garment tokens are MEMORY, not queries
  gate is a small fp32 scalar; at gate~0 the block is bit-identical to run37.

Trainable: garment branch (optional) + to_k_g/to_v_g + gate.  Frozen: main LoRA/v6/base.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from safetensors import safe_open
import trainlib.state as state

BASE = "/home/link/Desktop/Code/fashion gen testing"
QWEN = os.environ.get("PRETRAINED", f"{BASE}/Qwen-Image-Edit-2511")


def _finite(name, x):
    if int(os.environ.get("DEBUG_XATTN_FINITE", "0")) and torch.is_tensor(x) and not torch.isfinite(x).all():
        raise FloatingPointError(f"[GKV-NAN] {name} non-finite shape={tuple(x.shape)} dtype={x.dtype}")
    return x


class GarmentKVBranch(nn.Module):
    """Copied real Qwen blocks over the garment latent, run with CORRECT rope."""
    def __init__(self, transformer, block_ids, dim=3072):
        super().__init__()
        blocks = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks
        self.block_ids = list(block_ids)
        self.blocks = nn.ModuleList([deepcopy(blocks[b]) for b in self.block_ids])
        self.n_text = int(os.environ.get("GARMENT_KV_NTEXT", "8"))
        self.patch_proj = nn.Linear(16 * 4, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 * 48, dim)); nn.init.normal_(self.pos_embed, std=0.01)
        self.temb = nn.Parameter(torch.zeros(1, dim)); nn.init.trunc_normal_(self.temb, std=0.02)
        # dummy text at the BLOCK dim (3072), not the raw text-embed dim (3584 pre-txt_in).
        self.dummy_text = nn.Parameter(torch.zeros(1, self.n_text, dim)); nn.init.normal_(self.dummy_text, std=0.02)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, garment_latent, image_rotary_emb):
        garment_latent = garment_latent.to(self.patch_proj.weight.dtype)
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, (H // 2) * (W // 2), C * 4)
        x = self.patch_proj(x) + self.pos_embed[:, : (H // 2) * (W // 2)]
        _finite("branch.in", x)
        text = self.dummy_text.expand(B, -1, -1).to(x.dtype)
        text_mask = torch.ones(B, self.n_text, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).to(x.dtype)
        for i, blk in enumerate(self.blocks):
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=image_rotary_emb)
            _finite(f"branch.block{i}", x)
        return _finite("branch.out", self.out_norm(x))


class GarmentKVAttnProcessor(nn.Module):
    """Wraps the Qwen joint-attention, appending gated garment K/V. Query unchanged."""
    def __init__(self, base_processor, dim=3072, heads=24, head_dim=128, gate_init=-4.0):
        super().__init__()
        self.base = base_processor
        self.to_k_g = nn.Linear(dim, heads * head_dim, bias=True)
        self.to_v_g = nn.Linear(dim, heads * head_dim, bias=True)
        self.heads = heads
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None, **kw):
        gh = state._GARMENT_XATTN_HOLDER
        G = gh.get("Gkv") if gh else None
        import os as _os_kv
        if G is None or getattr(state, "_XATTN_BYPASS", False) or int(_os_kv.environ.get("XATTN_BYPASS", "0")):
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask,
                             image_rotary_emb=image_rotary_emb, **kw)
        # replicate the base joint attention, but append garment K/V
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        seq_txt = encoder_hidden_states.shape[1]
        iq = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        ik = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        iv = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        tq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        tk = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        tv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None: iq = attn.norm_q(iq)
        if attn.norm_k is not None: ik = attn.norm_k(ik)
        if attn.norm_added_q is not None: tq = attn.norm_added_q(tq)
        if attn.norm_added_k is not None: tk = attn.norm_added_k(tk)
        if image_rotary_emb is not None:
            imf, tmf = image_rotary_emb
            iq = apply_rotary_emb_qwen(iq, imf, use_real=False)
            ik = apply_rotary_emb_qwen(ik, imf, use_real=False)
            tq = apply_rotary_emb_qwen(tq, tmf, use_real=False)
            tk = apply_rotary_emb_qwen(tk, tmf, use_real=False)
        # garment K/V (no query, no rope on memory -> content-based attention)
        Kg = self.to_k_g(G).unflatten(-1, (attn.heads, -1))
        Vg = self.to_v_g(G).unflatten(-1, (attn.heads, -1))
        gate = torch.sigmoid(self.gate_logit).to(Kg.dtype)
        # run59 late-timestep schedule: weak/off at high sigma, stronger at low sigma
        # (early garment perturbations have more steps to spread into bg/skin).
        _shi = float(os.environ.get("GARMENT_KV_SIGMA_HI", "0"))
        if _shi > 0:
            _slo = float(os.environ.get("GARMENT_KV_SIGMA_LO", "0.2"))
            _snow = gh.get("sigma_now")
            if _snow is not None:
                _w = max(0.0, min(1.0, (_shi - float(_snow)) / max(_shi - _slo, 1e-6)))
                gate = gate * gate.new_tensor(_w)
        jq = torch.cat([tq, iq], dim=1)
        jk = torch.cat([tk, ik, Kg], dim=1)
        jv = torch.cat([tv, iv, gate * Vg], dim=1)
        # run58 QUERY MASK: only garment-region C-slot queries may attend to the appended garment K/V.
        # bg/skin/other-slot/text queries are blocked from the garment columns -> no leakage at the source.
        attn_mask = None
        if int(os.environ.get("GARMENT_KV_QUERY_MASK", "0")):
            p_g = gh.get("p_g_tok"); N_C = gh.get("N_C")
            if p_g is not None and N_C is not None:
                B_, N_q, H_, D_ = jq.shape
                N_g = Kg.shape[1]
                allow = torch.zeros(B_, N_q, device=jq.device, dtype=torch.bool)   # who may see garment
                cg = (p_g[:, :, 0] > 0.5)                                          # (B, N_C) garment C-tokens
                allow[:, seq_txt: seq_txt + N_C] = cg[:, :N_C]
                neg = torch.finfo(jq.dtype).min
                gcol = torch.where(allow[:, None, :, None], jq.new_zeros(1), jq.new_full((1,), neg))  # (B,1,N_q,1)
                attn_mask = torch.zeros(B_, 1, N_q, N_q + N_g, device=jq.device, dtype=jq.dtype)
                attn_mask[:, :, :, N_q:] = gcol
                if int(os.environ.get("GKV_MASK_DEBUG", "0")) and not getattr(GarmentKVAttnProcessor, "_mdbg", False):
                    _nall = int(allow.sum()); _tot = allow.numel()
                    print(f"[GKV-MASK] applied: N_q={N_q} N_C={N_C} N_g={N_g} | queries allowed to see garment: {_nall}/{_tot} "
                          f"({100*_nall/_tot:.1f}%) -> bg/skin/text/other-slot BLOCKED from garment K/V", flush=True)
                    GarmentKVAttnProcessor._mdbg = True
        out = dispatch_attention_fn(jq, jk, jv, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                                    backend=self.base._attention_backend,
                                    parallel_config=self.base._parallel_config)
        out = out.flatten(2, 3)
        txt_out, img_out = out[:, :seq_txt], out[:, seq_txt:]
        img_out = attn.to_out[0](img_out); img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out
