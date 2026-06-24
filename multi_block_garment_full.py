"""5_31 v6 redesign — parallel chain of full Qwen blocks for garment processing.

Approach (chosen for simplicity + OOTD-faithfulness):
  - 8 full QwenImageTransformerBlock copies form a parallel garment chain.
  - Chain initial input: packed garment_latent.
  - Pre-hook on main block 0 captures temb / image_rotary_emb / encoder_hidden
    from the main transformer's call, then runs the chain producing 8
    intermediate hidden_i states.
  - At each main block target site, a post-hook adds hidden_i × scalar_gate
    to main's F_C (C-slot tokens).

The chain effectively transforms garment_latent through pretrained Qwen
attention machinery (full block: attn + MLP + norms) before contributing.
This is a much stronger feature pathway than fresh-init Linear modules.

Memory: 8 × ~470M params + Adam states. With grad-ckpt + bf16 ≈ ~40GB.
"""
import torch
import torch.nn as nn
from copy import deepcopy


class GarmentChain(nn.Module):
    """Parallel chain of N full QwenImageTransformerBlock copies."""
    def __init__(self, transformer, target_blocks, dim=3072):
        super().__init__()
        blocks_list = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks
        self.target_blocks = list(target_blocks)
        self.n = len(target_blocks)
        new_blocks = []
        for blk_idx in target_blocks:
            new = deepcopy(blocks_list[blk_idx])
            new_blocks.append(new)
        self.blocks = nn.ModuleList(new_blocks)
        # Per-site gate scalar so residual injections start small.
        self.gates = nn.Parameter(torch.full((self.n,), 0.01))
        self.dim = dim
        # Patch projector (16 latent channels × 4 packed → dim) — fresh
        self.patch_proj = nn.Linear(16 * 4, dim, bias=True)
        # Pos embed for ~3072 tokens (64×48 packed)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3072, dim))
        nn.init.normal_(self.pos_embed, std=0.01)

    def encode_garment_latent(self, garment_latent):
        """Pack garment_latent (B, 16, 128, 96) → (B, 3072 tokens, dim)."""
        B, C, H, W = garment_latent.shape
        # Pixel-shuffle pack: 16ch × 2×2 = 64 → projected to dim
        x = garment_latent.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)
        x = self.patch_proj(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        return x

    def forward(self, garment_latent, encoder_hidden_states, encoder_hidden_states_mask,
                temb, image_rotary_emb=None):
        """Run the chain, return list of per-block image hidden states (B, N, dim)."""
        x = self.encode_garment_latent(garment_latent)
        text = encoder_hidden_states
        mask = encoder_hidden_states_mask
        states = []
        for blk in self.blocks:
            out = blk(
                hidden_states=x,
                encoder_hidden_states=text,
                encoder_hidden_states_mask=mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            if isinstance(out, tuple) and len(out) == 2:
                text_out, x = out
                text = text_out
            else:
                x = out
            states.append(x)
        return states


def install_garment_chain_hooks(transformer, chain, holder, gar_holder_key=None):
    """Install:
      1. Pre-hook on block 0 that captures temb/rope/enc and runs the chain,
         storing per-site outputs in holder['chain_states'].
      2. Per-site post-hooks at target blocks that add chain_states[i] × gate
         to F_C.

    Returns list of hook handles.
    """
    handles = []
    blocks_list = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks

    # 1) Pre-hook on block 0 to compute the chain
    def block0_pre_hook(module, args, kwargs):
        gh = holder if gar_holder_key is None else holder.get(gar_holder_key, {})
        if "garment_latent" not in gh: return
        garment_latent = gh["garment_latent"]
        temb = kwargs.get("temb")
        rope = kwargs.get("image_rotary_emb")
        enc = kwargs.get("encoder_hidden_states")
        mask = kwargs.get("encoder_hidden_states_mask")
        if temb is None or enc is None: return
        try:
            states = chain(garment_latent, enc, mask, temb, rope)
            gh["chain_states"] = states
        except Exception as _e:
            gh.pop("chain_states", None)

    h = blocks_list[chain.target_blocks[0]].register_forward_pre_hook(block0_pre_hook, with_kwargs=True)
    handles.append(h)

    # Wait — pre-hook above is on the FIRST TARGET block, but we want pre-hook BEFORE block 0
    # so chain can be computed once. Re-register on block 0 instead:
    h.remove()
    handles.pop()
    h = blocks_list[0].register_forward_pre_hook(block0_pre_hook, with_kwargs=True)
    handles.append(h)

    # 2) Post-hooks at each target block: add chain_states[i] × gate to F_C
    for site_idx, blk_idx in enumerate(chain.target_blocks):
        blk = blocks_list[blk_idx]

        def make_post_hook(_site_idx, _blk_idx):
            def post_hook(module, inputs, output):
                gh = holder if gar_holder_key is None else holder.get(gar_holder_key, {})
                states = gh.get("chain_states")
                if states is None or _site_idx >= len(states): return output
                N_C = gh.get("N_C")
                if N_C is None: return output
                state = states[_site_idx]
                if isinstance(output, tuple) and len(output) == 2:
                    text_out, img_out = output
                else:
                    text_out, img_out = None, output
                if img_out.dim() != 3 or img_out.shape[1] < N_C: return output
                F_C = img_out[:, :N_C, :]
                # state shape (B, N_C, dim) — add to F_C
                delta = chain.gates[_site_idx].to(state.dtype) * state
                if not torch.isfinite(delta).all(): return output
                F_C_new = F_C + delta
                img_out_new = torch.cat([F_C_new, img_out[:, N_C:, :]], dim=1)
                if text_out is not None: return (text_out, img_out_new)
                return img_out_new
            return post_hook

        h = blk.register_forward_hook(make_post_hook(site_idx, blk_idx))
        handles.append(h)

    return handles


# ─────────── Distributed v6 head sets ───────────

class MultiV6HeadSet(nn.Module):
    """8 sets of (to_route, to_s, to_b) heads distributed at 8 block sites."""
    def __init__(self, hidden_dim=3072, packed_dim=64, n_classes=4, patch=2, n_sites=8):
        super().__init__()
        self.n_sites = n_sites
        self.to_s_set = nn.ModuleList([nn.Linear(hidden_dim, packed_dim, bias=True) for _ in range(n_sites)])
        self.to_b_set = nn.ModuleList([nn.Linear(hidden_dim, packed_dim, bias=True) for _ in range(n_sites)])
        self.to_route_set = nn.ModuleList([nn.Linear(hidden_dim, n_classes * patch * patch, bias=True) for _ in range(n_sites)])
        # Zero-init s/b heads so the residuals start small
        for m in list(self.to_s_set) + list(self.to_b_set):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        for m in self.to_route_set:
            nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)

    def aggregate(self, hiddens_dict):
        """hiddens_dict: {site_idx: tensor(B, N_C, hidden_dim)} captured by hooks."""
        if not hiddens_dict: return None
        s_acc = 0.0; b_acc = 0.0; r_acc = 0.0; n = 0
        for site, h in hiddens_dict.items():
            if site >= self.n_sites: continue
            s_acc = s_acc + self.to_s_set[site](h)
            b_acc = b_acc + self.to_b_set[site](h)
            r_acc = r_acc + self.to_route_set[site](h)
            n += 1
        if n == 0: return None
        return {
            "delta_s_packed": s_acc / n,
            "delta_b_packed": b_acc / n,
            "route_logits": r_acc / n,
        }


def install_multi_v6_hooks(transformer, v6_sites, holder, gar_holder_key=None):
    """Post-hooks on each v6 site to capture F_C into holder['v6_hiddens'][i]."""
    handles = []
    blocks_list = transformer.transformer_blocks if hasattr(transformer, "transformer_blocks") else transformer.blocks
    for site_idx, blk_idx in enumerate(v6_sites):
        def make_hook(_site, _blk):
            def h(module, inputs, output):
                gh = holder if gar_holder_key is None else holder.get(gar_holder_key, {})
                N_C = gh.get("N_C")
                if N_C is None: return
                if isinstance(output, tuple) and len(output) == 2:
                    _, img_out = output
                else: img_out = output
                if not torch.is_tensor(img_out) or img_out.dim() != 3: return
                gh.setdefault("v6_hiddens", {})[_site] = img_out[:, :N_C, :]
            return h
        handle = blocks_list[blk_idx].register_forward_hook(make_hook(site_idx, blk_idx))
        handles.append(handle)
    return handles
