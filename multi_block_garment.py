"""Multi-block garment injection (instructions13).

For 15 selected transformer blocks, inject:
  - xAttn delta (no gate, no zero-init) at block OUTPUT
  - AdaLN-β delta at same block OUTPUT

Spatial mask routing (from v6 route head):
  - garment route: full strength
  - boundary band: medium (xattn 0.4, adaln 0.6)
  - skin/bg: zero/low (xattn 0, adaln 0.2)

Contribution logging:
  Each block records ||delta|| / ||hidden|| at each forward.

Public API:
  build_multi_block_injection(hidden_dim, num_heads, head_dim, n_blocks) → ModuleList
  install_multi_block_hooks(transformer, modules, target_blocks, holder) → list of hook handles
  build_spatial_mask_from_route(route_logits, garment_w, boundary_w, skin_w, bg_w) → (B, N_C, 1)
  per_block_norms_table(holder) → dict for logging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GarmentCrossAttnNoGate(nn.Module):
    """Cross-attention F → G with NO scalar gate and Xavier-init out_proj.

    Args:
      dim: hidden_dim of main transformer
      num_heads, head_dim: attention shape
    """
    def __init__(self, dim=3072, num_heads=24, head_dim=128, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = head_dim
        self.Wq = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.Wk = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.Wv = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=True)
        # qk-RMSNorm matching Qwen's qk_norm="rms_norm" for stability when
        # Wq/Wk are copied from pretrained (OOTD init).
        try:
            from diffusers.models.normalization import RMSNorm
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)
            self._use_qk_norm = True
        except Exception:
            self.norm_q = None; self.norm_k = None; self._use_qk_norm = False
        # Init std controllable via env var (5_31): default raised from 0.0001
        # to 0.01 so out_proj actually contributes a meaningful residual at start.
        # With qk-RMSNorm + grad clip + clip on multi-block params, should not NaN.
        import os as _os
        _op_std = float(_os.environ.get("XATTN_OUT_PROJ_STD", "0.01"))
        nn.init.normal_(self.out_proj.weight, std=_op_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, F_h, G):
        B, N_F, _ = F_h.shape
        N_G = G.shape[1]
        Q = self.Wq(F_h).view(B, N_F, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1, 2)
        if self._use_qk_norm:
            Q = self.norm_q(Q); K = self.norm_k(K)
        A = F.scaled_dot_product_attention(Q, K, V)
        A = A.transpose(1, 2).reshape(B, N_F, self.num_heads * self.head_dim)
        return self.out_proj(A)


class AdaLNBetaBlock(nn.Module):
    """Compute per-token β residual from garment encoder tokens G.

    β shape (B, N_C, dim) — added to hidden state at the C-slot.
    No gamma (β-only), small Xavier init.
    """
    def __init__(self, enc_dim=3072, dim=3072):
        super().__init__()
        self.to_beta = nn.Linear(enc_dim, dim, bias=True)
        # Init std controllable via env var (5_31): default raised from 0.0001
        # to 0.01 so β residual actually contributes a meaningful change at start.
        import os as _os
        _b_std = float(_os.environ.get("ADALN_BETA_STD", "0.01"))
        nn.init.normal_(self.to_beta.weight, std=_b_std)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, G):
        return self.to_beta(G)


class MultiBlockGarmentInjection(nn.Module):
    """Holds per-block xattn + adaln-β modules.

    n_blocks: number of injection points (e.g. 15)
    """
    def __init__(self, n_blocks, dim=3072, num_heads=24, head_dim=128, enc_dim=3072):
        super().__init__()
        self.xattn = nn.ModuleList([
            GarmentCrossAttnNoGate(dim=dim, num_heads=num_heads, head_dim=head_dim)
            for _ in range(n_blocks)
        ])
        self.adaln = nn.ModuleList([
            AdaLNBetaBlock(enc_dim=enc_dim, dim=dim) for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def init_from_qwen_blocks(self, transformer, target_blocks):
        """OOTD-style init: copy Wq/Wk/Wv/out_proj from main Qwen blocks
        at the matching target indices, so the xattn starts as the same
        attention computation the main block performs.

        Args:
          transformer: the inner Qwen transformer (has .transformer_blocks)
          target_blocks: list of block indices matching self.xattn order
        """
        assert len(target_blocks) == self.n_blocks
        blocks_list = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks
        copied = 0
        for inj_idx, blk_idx in enumerate(target_blocks):
            src = blocks_list[blk_idx]
            if not hasattr(src, "attn"):
                continue
            attn = src.attn
            try:
                xa = self.xattn[inj_idx]
                # to_q -> Wq
                with torch.no_grad():
                    xa.Wq.weight.data.copy_(attn.to_q.weight.data.to(xa.Wq.weight.dtype))
                    if attn.to_q.bias is not None and xa.Wq.bias is not None:
                        xa.Wq.bias.data.copy_(attn.to_q.bias.data.to(xa.Wq.bias.dtype))
                    xa.Wk.weight.data.copy_(attn.to_k.weight.data.to(xa.Wk.weight.dtype))
                    if attn.to_k.bias is not None and xa.Wk.bias is not None:
                        xa.Wk.bias.data.copy_(attn.to_k.bias.data.to(xa.Wk.bias.dtype))
                    xa.Wv.weight.data.copy_(attn.to_v.weight.data.to(xa.Wv.weight.dtype))
                    if attn.to_v.bias is not None and xa.Wv.bias is not None:
                        xa.Wv.bias.data.copy_(attn.to_v.bias.data.to(xa.Wv.bias.dtype))
                    # to_out[0] is the projection back to dim
                    to_out = attn.to_out[0] if isinstance(attn.to_out, (nn.Sequential, list)) or hasattr(attn.to_out, "__getitem__") else attn.to_out
                    # Note: do NOT copy out_proj from Qwen — leave it at the
                    # tiny near-zero init from __init__. This keeps the residual
                    # delta near-zero at start (critical for 15-block additive
                    # injection NaN safety), while Wq/Wk/Wv learn to use pretrained
                    # attention computations.
                    # Copy qk-RMSNorm weights too if both sides have them
                    if (getattr(attn, "norm_q", None) is not None
                            and getattr(xa, "norm_q", None) is not None):
                        try:
                            xa.norm_q.weight.data.copy_(attn.norm_q.weight.data.to(xa.norm_q.weight.dtype))
                            xa.norm_k.weight.data.copy_(attn.norm_k.weight.data.to(xa.norm_k.weight.dtype))
                        except Exception: pass
                copied += 1
            except Exception as e:
                print(f"[multi_block_inj] OOTD-init copy failed at inj_idx={inj_idx} blk_idx={blk_idx}: {e}")
        print(f"[multi_block_inj] OOTD-style init: copied Qwen attention weights into {copied}/{self.n_blocks} blocks")


def build_spatial_mask_from_route(route_probs, garment_w=1.0, boundary_w=0.4,
                                    skin_w=0.0, bg_w=0.0):
    """Build per-token spatial mask from v6 route probabilities.

    Args:
      route_probs: (B, N_C, K) softmax probabilities (K=4: garment, skin, bg, keep)
      *_w: weights for each route class

    Returns:
      mask: (B, N_C, 1) ∈ [0, 1]
    """
    # route_probs assumed (B, N_C, K)
    p_g = route_probs[..., 0:1]
    p_s = route_probs[..., 1:2] if route_probs.shape[-1] > 1 else torch.zeros_like(p_g)
    p_b = route_probs[..., 2:3] if route_probs.shape[-1] > 2 else torch.zeros_like(p_g)
    mask = garment_w * p_g + skin_w * p_s + bg_w * p_b
    return mask.clamp(0, 1)


def build_spatial_mask_from_warped(warped_mask_packed, M_full_packed,
                                    garment_w=1.0, boundary_w=0.4,
                                    skin_w=0.0, bg_w=0.0):
    """Alternative: build spatial mask from warped_mask (garment) and M_full (edit region).

    Args:
      warped_mask_packed: (B, N_C, 1) ∈ [0, 1] — garment silhouette in latent token space
      M_full_packed:      (B, N_C, 1) ∈ [0, 1] — edit/repair region

    Returns:
      mask: (B, N_C, 1)
    """
    g = warped_mask_packed
    bg = (1 - M_full_packed)  # outside edit area
    # boundary = M_full but not garment core
    # we use erosion-like via a soft op: boundary_token = (M_full * (1 - g)) ~~ ring zone
    boundary = (M_full_packed * (1 - g)).clamp(0, 1)
    # rest is approximated as "skin" inside M_full
    skin = boundary  # boundary acts as repair_skin proxy at token level (no per-token parse)
    mask = garment_w * g + boundary_w * boundary
    # subtract bg weighting if requested
    if bg_w != 0: mask = mask + bg_w * bg
    return mask.clamp(0, 1)


def install_multi_block_hooks(transformer, injection: MultiBlockGarmentInjection,
                              target_blocks, holder, gar_holder_key="multi_G"):
    """Register hooks on each target block's output to inject xattn + adaln-β.

    Args:
      transformer: the inner transformer (has .transformer_blocks ModuleList)
      injection: MultiBlockGarmentInjection
      target_blocks: list of block indices, e.g. [4,8,12,...,59]
      holder: dict; reads holder[gar_holder_key]["G"] (encoder output) and
              holder[gar_holder_key]["spatial_mask"] (B, N_C, 1) at each step
              holder[gar_holder_key]["N_C"] = #image tokens in C slot
              writes holder[gar_holder_key]["block_norms"] = {block_idx: {"xattn_delta_norm": float, ...}}

    Returns:
      list of hook handles
    """
    assert len(target_blocks) == injection.n_blocks, \
        f"target_blocks length {len(target_blocks)} != n_blocks {injection.n_blocks}"

    handles = []
    blocks_list = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks

    for inj_idx, blk_idx in enumerate(target_blocks):
        blk = blocks_list[blk_idx]
        xattn = injection.xattn[inj_idx]
        adaln = injection.adaln[inj_idx]

        def make_hook(_inj_idx, _blk_idx, _xattn, _adaln):
            def hook(module, inputs, output):
                gh = holder if gar_holder_key is None else holder.get(gar_holder_key, {})
                if "G" not in gh: return output
                # Accept either "spatial_mask" (training) or "p_g_tok" (inference legacy key)
                mask = gh.get("spatial_mask", gh.get("p_g_tok"))
                if mask is None: return output
                G = gh["G"]
                N_C = gh.get("N_C")
                if N_C is None: return output

                # output of block — qwen blocks return (text_out, img_out) typically.
                # We need to extract image-stream output.
                if isinstance(output, tuple) and len(output) == 2:
                    text_out, img_out = output
                else:
                    # Single tensor — assume image stream
                    text_out, img_out = None, output

                # img_out: (B, N_img_total, dim). C-slot = first N_C tokens.
                if img_out.dim() != 3:
                    return output
                if img_out.shape[1] < N_C:
                    return output
                F_C = img_out[:, :N_C, :]
                # xattn delta
                A_g = _xattn(F_C, G)                              # (B, N_C, dim)
                # adaln-β delta
                B_g = _adaln(G[:, :N_C, :] if G.shape[1] >= N_C else G)  # (B, N_C, dim)
                # Apply spatial mask
                m = mask.to(dtype=A_g.dtype)
                if m.shape[1] != N_C:
                    # broadcast / pad
                    m = m[:, :N_C]
                xattn_mask_w = gh.get("xattn_spatial_w", 1.0)
                adaln_mask_w = gh.get("adaln_spatial_w", 1.0)
                delta_x = xattn_mask_w * m * A_g
                delta_a = adaln_mask_w * m * B_g

                # Log norms (detached, float)
                with torch.no_grad():
                    hn = float(F_C.detach().float().norm().item())
                    xn = float(delta_x.detach().float().norm().item())
                    an = float(delta_a.detach().float().norm().item())
                bn = gh.setdefault("block_norms", {})
                bn[_blk_idx] = {"hidden_norm": hn, "xattn_delta_norm": xn,
                                "adaln_delta_norm": an,
                                "xattn_ratio": xn / max(hn, 1e-9),
                                "adaln_ratio": an / max(hn, 1e-9)}

                # NaN guard: if the injection produced NaN/Inf, skip this block
                # (defensive — fresh-init blocks can spike under amp+bf16).
                if not (torch.isfinite(delta_x).all() and torch.isfinite(delta_a).all()):
                    return output
                # Modify img_out
                F_C_new = F_C + delta_x + delta_a
                img_out_new = torch.cat([F_C_new, img_out[:, N_C:, :]], dim=1)

                if text_out is not None:
                    return (text_out, img_out_new)
                return img_out_new
            return hook

        h = blk.register_forward_hook(make_hook(inj_idx, blk_idx, xattn, adaln))
        handles.append(h)
    return handles


def per_block_norms_table(holder, target_blocks):
    """Aggregate block_norms into a flat dict for logging."""
    bn = holder.get("block_norms", {})
    return {f"blk{bi}/xattn_ratio": bn.get(bi, {}).get("xattn_ratio", 0.0)
            for bi in target_blocks}
