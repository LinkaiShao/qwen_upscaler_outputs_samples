"""Model-subsystem factories, hooks, processors (use trainlib.state)."""
import os, sys
import torch
import torch.nn as nn
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
try:
    from multi_block_garment import (MultiBlockGarmentInjection, install_multi_block_hooks,
                                      build_spatial_mask_from_warped, per_block_norms_table)
except Exception:
    pass
from trainlib import state

from trainlib.models import (GarmentCrossAttn, GarmentLatentEnhancer, GarmentNet, GarmentNetAdaLN, GarmentNetCrossAttn, GarmentNetOutput, GarmentRepairGate, GarmentSlotEncoder, OOTDInjector, PatchDiscriminator, QwenAuxSlotEncoder, QwenControlNet, QwenGarmentEncoder, QwenGarmentNetAdaLN, QwenGarmentNetBlockCopyAdaLN, QwenGarmentNetCrossAttn, QwenLatentRefiner, QwenSlotEnricher, V6Heads, _make_qwen_block, perceptual_loss)
from trainlib.data import (pack_latents, unpack_latents, vae_decode_to_pil, precompute_rough_pils, precompute_prompt_embeds, load_pose_latents, VTONDataset, collate_fn)
from trainlib.constants import *


def get_vgg_features(device, dtype):
    if state._VGG_MODEL is None:
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        vgg.requires_grad_(False)
        # Use layers up to relu3_3 (index 16) for perceptual features
        state._VGG_MODEL = torch.nn.Sequential(*list(vgg.children())[:16]).to(device, dtype=dtype)
    return state._VGG_MODEL


def _get_v6_heads(device, dtype, hidden_dim=3072):
    if state._V6_HEADS is None:
        state._V6_HEADS = V6Heads(hidden_dim=hidden_dim, packed_dim=64).to(device, dtype=dtype)
        state._V6_HEADS.requires_grad_(True)
    return state._V6_HEADS


def _v6_hidden_hook(module, input, output):
    """Forward hook on transformer.norm_out: captures (B, N_total, 3072) features."""
    state._HIDDEN_HOLDER["hidden"] = output


def _get_invalid_token(device, dtype, dim=64):
    if state._INVALID_TOKEN is None:
        _t = torch.zeros(1, 1, dim, device=device, dtype=dtype)
        torch.nn.init.normal_(_t, std=0.02)
        state._INVALID_TOKEN = torch.nn.Parameter(_t)
        state._INVALID_TOKEN.requires_grad_(True)
    return state._INVALID_TOKEN


class CrossAttnGarmentProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Drop-in replacement for QwenDoubleStreamAttnProcessor2_0 that adds an
    extra cross-attention contribution to the IMAGE stream output:

        img_attn_total = img_attn_joint + gate · sdpa(img_Q, K_g, V_g)

    K_g, V_g come from a shared global `state._CROSS_ATTN_HOLDER` populated once per
    forward by the trainer. `gate` is a per-image-token mask (1 inside warped,
    0 outside) — image queries outside the garment silhouette get NO garment
    contribution → no bleeding.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn  # noqa: F401
        if encoder_hidden_states is None:
            raise ValueError("CrossAttnGarmentProcessor requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key   = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key   = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key   = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key   = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None: img_query = attn.norm_q(img_query)
        if attn.norm_k is not None: img_key   = attn.norm_k(img_key)
        if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None: txt_key   = attn.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key   = apply_rotary_emb_qwen(img_key,   img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend, parallel_config=self._parallel_config)
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # ── Garment cross-attention (C-slot tokens only — first n_c image tokens) ──
        K_g  = state._CROSS_ATTN_HOLDER.get("K_g")
        V_g  = state._CROSS_ATTN_HOLDER.get("V_g")
        gate = state._CROSS_ATTN_HOLDER.get("gate")          # (B, n_c, 1), n_c = 3072 (C-slot)
        if K_g is not None and V_g is not None:
            n_c = gate.shape[1] if gate is not None else 3072
            gk = K_g.unflatten(-1, (attn.heads, -1)).to(dtype=img_query.dtype)   # (B, N_g, H, D)
            gv = V_g.unflatten(-1, (attn.heads, -1)).to(dtype=img_query.dtype)
            if attn.norm_k is not None:
                gk = attn.norm_k(gk)
            img_q_c = img_query[:, :n_c]                                          # (B, n_c, H, D)
            cross_out_c = dispatch_attention_fn(
                img_q_c, gk, gv,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            cross_out_c = cross_out_c.flatten(2, 3).to(img_query.dtype)           # (B, n_c, inner_dim)
            if gate is not None:
                cross_out_c = cross_out_c * gate.to(dtype=cross_out_c.dtype)
            img_attn_output = torch.cat([
                img_attn_output[:, :n_c] + cross_out_c,
                img_attn_output[:, n_c:],
            ], dim=1)

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output


class QwenGarmentBranch(nn.Module):
    """v2 (2026-04-29): parallel garment branch with ZERO-INIT pos/temb/dummy_text
    (random init was destroying the copied blocks' input distribution). Returns
    per-block POST-block outputs (not pre-block inputs) so the branch's processed
    representations are what flows out — input is just patch_proj which is random
    until trained."""
    def __init__(self, in_lat_ch=16, n_layers=None):
        if n_layers is None:
            n_layers = int(os.environ.get("GARMENT_OOTD_LAYERS", "4"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128  # match main
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        # ZERO-INIT pos/temb/dummy_text (was random — caused distribution mismatch
        # for the main-block copies that expect specific conditioning shape).
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks)
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def precompute(self, garment_latent):
        """Returns per-block POST-block outputs (the processed representation after
        each branch block). Last element is the deepest representation — typically
        what gets used at the main injection blocks."""
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        per_block_outputs = []
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
            per_block_outputs.append(x)                     # POST-block
        return per_block_outputs                            # list of N tensors (B, 3072, 3072)


class OOTDQwenAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    """v2: OOTDiffusion-style K,V injection with DEDICATED to_k_g/to_v_g per
    injection block (not shared with main attn) and a learned scalar gate.
    block_idx is the MAIN block index (e.g., 48 for late-stage injection).
    Reads `state._GARMENT_BRANCH_HOLDER["final_h"]` (single tensor, the garment branch's
    deepest output) and the per-block injector module from `state._OOTD_INJECTORS[block_idx]`.
    """
    def __init__(self, block_idx=0):
        super().__init__()
        self.block_idx = block_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn  # noqa: F401
        if encoder_hidden_states is None:
            raise ValueError("OOTDQwenAttnProcessor requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states)
        img_key   = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key   = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key   = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key   = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q       is not None: img_query = attn.norm_q(img_query)
        if attn.norm_k       is not None: img_key   = attn.norm_k(img_key)
        if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None: txt_key   = attn.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key   = apply_rotary_emb_qwen(img_key,   img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # ── Main attention (baseline path, unchanged) ──
        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend, parallel_config=self._parallel_config)

        # ── OOTD v5 ──
        # Selectivity stack:
        #   (1) IMG tokens only — txt untouched
        #   (2) Per-token spatial mask p_g_tok (mixed source: target/warped/jittered
        #       at train; warped at infer) so garment info only modifies garment region
        #   (3) Per-block learned scalar gate sigmoid(injector.gate_logit), init logit=-4
        #       for low initial influence; per-block amplitude control
        #   (4) Depth-specific garment hidden: each injection block reads its own
        #       branch-depth output (h_{block_idx}), not a shared "final" representation
        #   (5) Garment dropout: occasionally skip the contribution entirely (set in
        #       train_step via state._GARMENT_BRANCH_HOLDER["dropout"])
        # to_v_g is zero-init so gar_attn ≡ 0 at start (baseline preserved).
        if state._GARMENT_BRANCH_HOLDER.get("dropout", False):
            gar_h = None
        else:
            gar_h = state._GARMENT_BRANCH_HOLDER.get(f"h_{self.block_idx}",
                                                state._GARMENT_BRANCH_HOLDER.get("final_h"))
        injector = state._OOTD_INJECTORS.get(self.block_idx)
        if gar_h is not None and injector is not None:
            gar_h_ = gar_h.to(dtype=img_query.dtype)
            gar_key   = injector.to_k_g(gar_h_).unflatten(-1, (attn.heads, -1))     # (B, N_g, h, d)
            gar_value = injector.to_v_g(gar_h_).unflatten(-1, (attn.heads, -1))
            if attn.norm_k is not None:
                gar_key = attn.norm_k(gar_key)
            gar_attn_img = dispatch_attention_fn(
                img_query, gar_key, gar_value,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            # Per-token spatial mask for image tokens (only first N_C tokens are C-slot)
            p_g_tok = state._GARMENT_BRANCH_HOLDER.get("p_g_tok")
            if p_g_tok is not None:
                B_, N_img_total = img_query.shape[0], img_query.shape[1]
                N_C = p_g_tok.shape[1]
                if N_img_total > N_C:
                    p_g_full = torch.cat([p_g_tok, torch.zeros(B_, N_img_total - N_C, 1,
                                                              device=p_g_tok.device, dtype=p_g_tok.dtype)], dim=1)
                else:
                    p_g_full = p_g_tok
                gar_attn_img = gar_attn_img * p_g_full.unsqueeze(-1).to(gar_attn_img.dtype)
            # Per-block learned scalar gate
            gate = torch.sigmoid(injector.gate_logit).to(dtype=gar_attn_img.dtype)
            gar_attn_img = gate * gar_attn_img
            # σ-conditional gating: f(σ) ∈ [0,1], 1 at high noise / 0 at clean.
            # Apply garment contribution at early shape decisions only; at low σ
            # let the baseline finish detail without garment perturbation.
            sigma_w = state._GARMENT_BRANCH_HOLDER.get("sigma_w")
            if sigma_w is not None:
                # sigma_w is (B, 1, 1, 1) — broadcast to (B, N, h, d)
                gar_attn_img = gar_attn_img * sigma_w.view(-1, 1, 1, 1).to(dtype=gar_attn_img.dtype)
            # Add ONLY to image stream, leave txt unchanged
            img_attn_part = joint_hidden_states[:, seq_txt:, :, :] + gar_attn_img
            joint_hidden_states = torch.cat([joint_hidden_states[:, :seq_txt, :, :], img_attn_part], dim=1)

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        # Output length matches Q (txt + img); garment contributed only via K,V
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output


def _get_garment_branch(device, dtype, n_layers=None):
    if state._GARMENT_BRANCH is None:
        state._GARMENT_BRANCH = QwenGarmentBranch(n_layers=n_layers).to(device, dtype=dtype)
        state._GARMENT_BRANCH.requires_grad_(True)
    return state._GARMENT_BRANCH


def _get_qwen_slot_enricher(device, dtype):
    if state._QWEN_SLOT_ENRICHER is None:
        state._QWEN_SLOT_ENRICHER = QwenSlotEnricher().to(device, dtype=dtype)
        state._QWEN_SLOT_ENRICHER.requires_grad_(True)
    return state._QWEN_SLOT_ENRICHER


def _get_qwen_aux_slot(device, dtype):
    if state._QWEN_AUX_SLOT is None:
        state._QWEN_AUX_SLOT = QwenAuxSlotEncoder().to(device, dtype=dtype)
        state._QWEN_AUX_SLOT.requires_grad_(True)
    return state._QWEN_AUX_SLOT


def _get_qwen_refiner(device, dtype):
    if state._QWEN_REFINER is None:
        state._QWEN_REFINER = QwenLatentRefiner().to(device, dtype=dtype)
        state._QWEN_REFINER.requires_grad_(True)
    return state._QWEN_REFINER


def _get_garment_encoder(device, dtype, n_layers=None):
    if state._GARMENT_ENCODER is None:
        state._GARMENT_ENCODER = QwenGarmentEncoder(n_layers=n_layers).to(device, dtype=dtype)
        state._GARMENT_ENCODER.requires_grad_(True)
    return state._GARMENT_ENCODER


def _get_garment_xattn(device, dtype, gate_init_logit=None):
    if state._GARMENT_XATTN is None:
        if gate_init_logit is None:
            gate_init_logit = float(os.environ.get("GARMENT_XATTN_GATE_INIT_LOGIT", "-3.0"))
        state._GARMENT_XATTN = GarmentCrossAttn(dim=3072, num_heads=24, head_dim=128,
                                          gate_init_logit=gate_init_logit).to(device, dtype=dtype)
        # Force gate_logit back to fp32 (the .to(bf16) cast downcasts; updates would quantize)
        state._GARMENT_XATTN.gate_logit.data = state._GARMENT_XATTN.gate_logit.data.to(torch.float32)
        state._GARMENT_XATTN.requires_grad_(True)
    return state._GARMENT_XATTN


def _get_controlnet(device, dtype, n_layers=None):
    if state._CONTROLNET is None:
        state._CONTROLNET = QwenControlNet(in_lat_ch=16, hidden_dim=3072, n_layers=n_layers).to(device, dtype=dtype)
        state._CONTROLNET.requires_grad_(True)
    return state._CONTROLNET


def _make_controlnet_block_hook(block_idx):
    """Returns a forward_hook that adds the ControlNet residual to this block's
    hidden_states (image stream, C-slot only). The block forward returns a tuple
    (encoder_hidden_states, hidden_states); we modify hidden_states in place by
    adding state._CONTROLNET_HOLDER[block_idx] to its first N_C tokens (C-slot)."""
    def hook(module, inputs, outputs):
        residual = state._CONTROLNET_HOLDER.get(block_idx)
        if residual is None:
            return outputs
        # outputs = (encoder_hidden_states, hidden_states)
        ehs, h = outputs[0], outputs[1]
        N_C = state._CONTROLNET_HOLDER.get("N_C", residual.size(1))
        # Add residual to C-slot tokens [:, :N_C, :]
        h_C = h[:, :N_C, :] + residual.to(h.dtype)
        h_new = torch.cat([h_C, h[:, N_C:, :]], dim=1)
        return (ehs, h_new)
    return hook


def _proj_out_pre_hook(module, args):
    """Fires before transformer.proj_out. The input has shape (B, N_total_img, 3072)
    where N_total_img = NUM_SLOTS × N_C (e.g., 4 × 3072 = 12288). Only the FIRST
    N_C tokens are the C-slot whose proj_out output is read as v_pred — the other
    slots are conditioning context that doesn't affect downstream prediction. So we
    enhance the C-slot only, leave other slots untouched.

    Cross-attention: Q from C-slot tokens F_C, K/V from garment encoder features G.
    Gated per-token by garment mask. v6 skin/bg/route heads see un-enhanced F via
    the norm_out hook, which fires BEFORE this proj_out pre-hook."""
    holder = state._GARMENT_XATTN_HOLDER
    if not holder or "G" not in holder or "p_g_tok" not in holder:
        return None
    F_full = args[0]                             # (B, NUM_SLOTS*N_C, 3072)
    N_C = holder.get("N_C", 3072)
    if not getattr(_proj_out_pre_hook, "_logged_shape", False):
        print(f"[xattn-hook] F_full shape={tuple(F_full.shape)} N_C={N_C} G shape={tuple(holder['G'].shape)} p_g_tok shape={tuple(holder['p_g_tok'].shape)}")
        _proj_out_pre_hook._logged_shape = True
    F_C = F_full[:, :N_C, :]
    A_g = state._GARMENT_XATTN(F_C, holder["G"])       # (B, N_C, 3072)
    gamma_inj = holder.get("gamma", 1.0)
    F_C_g = F_C + gamma_inj * holder["p_g_tok"] * A_g
    F_modified = torch.cat([F_C_g, F_full[:, N_C:, :]], dim=1)
    return (F_modified,) + args[1:]


def _get_garment_net(device, dtype, hidden_dim=3072):
    if state._GARMENT_NET is None:
        mode = os.environ.get("GARMENT_NET_MODE", "norm_residual")
        encoder = os.environ.get("GARMENT_NET_ENCODER", "conv")  # "conv" | "qwen"
        if mode == "output_space":
            state._GARMENT_NET = GarmentNetOutput().to(device, dtype=dtype)
        elif mode == "garment_latent_residual":
            state._GARMENT_NET = GarmentLatentEnhancer().to(device, dtype=dtype)
        elif mode == "extra_slot":
            state._GARMENT_NET = GarmentSlotEncoder().to(device, dtype=dtype)
        elif mode == "adaln":
            if encoder == "qwen_blockcopy":
                state._GARMENT_NET = QwenGarmentNetBlockCopyAdaLN(hidden_dim=hidden_dim).to(device, dtype=dtype)
            elif encoder == "qwen":
                state._GARMENT_NET = QwenGarmentNetAdaLN(hidden_dim=hidden_dim).to(device, dtype=dtype)
            else:
                state._GARMENT_NET = GarmentNetAdaLN(hidden_dim=hidden_dim).to(device, dtype=dtype)
        elif mode == "cross_attn":
            if encoder == "qwen":
                state._GARMENT_NET = QwenGarmentNetCrossAttn(inner_dim=hidden_dim).to(device, dtype=dtype)
            else:
                state._GARMENT_NET = GarmentNetCrossAttn(inner_dim=hidden_dim).to(device, dtype=dtype)
        else:
            state._GARMENT_NET = GarmentNet(hidden_dim=hidden_dim).to(device, dtype=dtype)
        state._GARMENT_NET.requires_grad_(True)
    return state._GARMENT_NET


def _garment_inject_hook(module, inputs, output):
    """Forward hook on transformer.norm_out. Two modes:
      - residual (additive): out = out + residual * gate
      - adaln    (FiLM):     out = out * (1 + γ·gate) + β·gate
    Spatial gate from packed warped_mask ensures zero modulation outside the
    garment silhouette → no bleeding into repair/keep regions.
    Constructs output via torch.cat to avoid in-place autograd errors.
    """
    res = state._GARMENT_RESIDUAL_HOLDER.get("residual")
    gamma = state._GARMENT_RESIDUAL_HOLDER.get("gamma")
    beta  = state._GARMENT_RESIDUAL_HOLDER.get("beta")
    gate  = state._GARMENT_RESIDUAL_HOLDER.get("gate")
    if res is None and (gamma is None or beta is None):
        return output

    n_c = (res.shape[1] if res is not None else gamma.shape[1])
    if output.shape[1] < n_c:
        return output

    head = output[:, :n_c, :]
    tail = output[:, n_c:, :]

    if res is not None:
        add = res.to(dtype=head.dtype)
        if gate is not None:
            add = add * gate.to(dtype=head.dtype)
        head = head + add
    if gamma is not None and beta is not None:
        g = gamma.to(dtype=head.dtype)
        b = beta.to(dtype=head.dtype)
        if int(os.environ.get("GARMENT_NET_BETA_ONLY", "0")):
            g = torch.zeros_like(g)         # zero out scaling, keep only additive
        if int(os.environ.get("GARMENT_NET_GAMMA_ONLY", "0")):
            b = torch.zeros_like(b)         # zero out additive, keep only scaling
        if gate is not None:
            gt = gate.to(dtype=head.dtype)
            g = g * gt
            b = b * gt
        head = head * (1.0 + g) + b

    return torch.cat([head, tail], dim=1)


def _get_discriminator(device, dtype):
    if state._DISC is None:
        state._DISC = PatchDiscriminator().to(device, dtype=dtype)
        state._DISC_OPT = torch.optim.Adam(state._DISC.parameters(), lr=1e-4, betas=(0.5, 0.999))
    return state._DISC, state._DISC_OPT


def _get_critic(device, dtype):
    if state._CRITIC is None:
        from conditioned_critic import ConditionedPatchCritic
        state._CRITIC = ConditionedPatchCritic().to(device, dtype=dtype)
        state._CRITIC_OPT = torch.optim.Adam(state._CRITIC.parameters(),
                                       lr=float(os.environ.get("CRITIC_LR", "1e-4")),
                                       betas=(0.5, 0.999))
    return state._CRITIC, state._CRITIC_OPT


class HoleyAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Same as parent but reads attention_mask from global holder when not provided."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        if attention_mask is None:
            attention_mask = ATTN_MASK_HOLDER.get("mask")
        return super().__call__(attn, hidden_states, encoder_hidden_states,
                                encoder_hidden_states_mask, attention_mask, image_rotary_emb)


class AgnosticCtrlProcessor(QwenDoubleStreamAttnProcessor2_0):
    """v19/v20/v21: trust-map-weighted suppression of the agnostic slot.
      - v19 key bias : per-token additive pre-softmax bias on agnostic key columns
      - v20 V scale  : per-token multiplicative scale on agnostic value vectors
      - v21          : both
    Per-token bias/scale come from state._AGN_CTRL (populated by the trainer). The trust
    map is high (≈1) for useful agnostic regions (face/hair/keep) and low (≈0.3)
    for the grey/edit core, so useful agnostic context is NOT suppressed."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        if encoder_hidden_states is None:
            raise ValueError("AgnosticCtrlProcessor requires encoder_hidden_states")
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key   = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key   = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key   = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key   = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None: img_query = attn.norm_q(img_query)
        if attn.norm_k is not None: img_key   = attn.norm_k(img_key)
        if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None: txt_key   = attn.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key   = apply_rotary_emb_qwen(img_key,   img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

        agn_start = state._AGN_CTRL.get("agn_img_start")
        n_agn     = state._AGN_CTRL.get("n_agn")

        # ── v20: trust-weighted V-only scaling on agnostic value vectors ──
        v_scale = state._AGN_CTRL.get("v_scale_tok")
        if v_scale is not None and agn_start is not None:
            s = v_scale.to(img_value.dtype)[:, :, None, None]   # (B, n_agn, 1, 1)
            img_value = img_value.clone()
            img_value[:, agn_start:agn_start + n_agn] = \
                img_value[:, agn_start:agn_start + n_agn] * s

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # ── v19: trust-weighted key bias on agnostic key columns ──
        key_bias = state._AGN_CTRL.get("key_bias_tok")
        if key_bias is not None and agn_start is not None:
            total_seq = joint_key.shape[1]
            bias = torch.zeros(joint_key.shape[0], 1, 1, total_seq,
                               device=joint_key.device, dtype=joint_query.dtype)
            kstart = seq_txt + agn_start
            bias[:, 0, 0, kstart:kstart + n_agn] = key_bias.to(joint_query.dtype)
            attention_mask = bias if attention_mask is None else attention_mask + bias

        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend, parallel_config=self._parallel_config)
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output


def build_repair_attn_mask(M_full_lat, warped_mask_lat, txt_seq_len, num_slots,
                            B, device, dtype, agnostic_slot_idx_in_seq=1):
    """
    Build attention bias to block C-slot repair-band queries from attending to
    agnostic-slot out-of-mask keys.

    Args:
      M_full_lat: (B, 1, H, W) binary edit-region mask at latent res
      warped_mask_lat: (B, 1, H, W) warped garment mask (soft) at latent res
      txt_seq_len: scalar — number of text tokens prepended to image tokens
      num_slots: NUM_SLOTS (C + spatial slots)
      agnostic_slot_idx_in_seq: 1-based position of agnostic in spatial slots
                                 (0 = C, 1 = first spatial slot which is agnostic)

    Returns: (B, 1, total_seq, total_seq) bias tensor with -inf at blocked Q,K pairs.
    """
    H, W = M_full_lat.shape[-2:]
    img_tok_per_slot = (H // 2) * (W // 2)             # 3072 with 128x96 / 2x2 packing
    total_img = num_slots * img_tok_per_slot
    total_seq = txt_seq_len + total_img

    # Repair band = M_full minus warped (binary)
    wm_bin = (warped_mask_lat > 0.5).to(dtype=M_full_lat.dtype)
    repair_band = (M_full_lat - wm_bin).clamp(0, 1)
    keep_mask = (1.0 - M_full_lat).clamp(0, 1)

    # Pack to token level (mean over channels for the broadcast)
    repair_tok = pack_latents(repair_band.expand(B, 16, H, W), B, 16, H, W).mean(dim=-1)  # (B, 3072)
    keep_tok   = pack_latents(keep_mask.expand(B, 16, H, W),   B, 16, H, W).mean(dim=-1)   # (B, 3072)

    repair_pos = (repair_tok > 0.5)   # (B, 3072) bool
    keep_pos   = (keep_tok   > 0.5)   # (B, 3072) bool

    mask = torch.zeros(B, 1, total_seq, total_seq, device=device, dtype=dtype)
    for b in range(B):
        q_idx = txt_seq_len + torch.where(repair_pos[b])[0]                          # C-slot repair positions
        k_idx = txt_seq_len + agnostic_slot_idx_in_seq * img_tok_per_slot + torch.where(keep_pos[b])[0]
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        mask[b, 0, q_idx.unsqueeze(1), k_idx.unsqueeze(0)] = -1e4
    return mask


def install_garment_gates(transformer, n_c_tokens, garment_slot_idx, device, dtype,
                          n_heads=8, head_dim=64, every_n=3):
    """
    Install GarmentRepairGate on every `every_n`-th transformer block via forward hooks.
    Returns list of gate modules (for optimizer) and the hooks (for cleanup).

    n_c_tokens: number of C tokens (3072 for 128×96 latent with 2×2 packing)
    garment_slot_idx: which slot (0-indexed after C) is garment. E.g. if SLOT_ORDER
                      = [ag, pose, gar] and garment is slot 2 → garment_slot_idx = 2.
    """
    gates = []
    hooks = []
    # Find all transformer blocks
    blocks = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ == "QwenImageTransformerBlock":
            blocks.append((name, module))

    dim = 3072  # Qwen: num_attention_heads(24) × attention_head_dim(128)
    for i, (name, block) in enumerate(blocks):
        if i % every_n != 0:
            continue

        gate = GarmentRepairGate(dim, n_heads, head_dim).to(device, dtype=dtype)
        gates.append(gate)

        def make_hook(gate_module, nc, gs_idx):
            def hook_fn(module, args, output):
                # output = (txt_hidden, img_hidden) from the block
                txt_h, img_h = output
                # img_h: (B, total_img_tokens, D)
                # Extract C chunk and garment chunk
                h_c = img_h[:, :nc, :]                                              # C tokens
                gar_start = nc + gs_idx * nc
                h_gar = img_h[:, gar_start:gar_start + nc, :]                      # garment tokens
                # The block's normal output for C is already in h_c (this IS u_rep,
                # since it's the residual-connected output of self-attention + FFN)
                # We compute u_gar from garment cross-attention and gate-mix into h_c
                u_gar_mixed, alpha = gate_module(h_c, h_gar, h_c)
                # Replace C chunk with the gated output
                img_h_new = img_h.clone()
                img_h_new[:, :nc, :] = u_gar_mixed
                return (txt_h, img_h_new)
            return hook_fn

        h = block.register_forward_hook(make_hook(gate, n_c_tokens, garment_slot_idx))
        hooks.append(h)

    return gates, hooks
