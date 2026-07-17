"""Inference adapter — exp401 (5-pos: exp395 + rough spatial slot, pose-mode aware)."""

import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel, QwenDoubleStreamAttnProcessor2_0,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

# exp415: custom attention processor that reads attention_mask from module global
_INFER_MASK_HOLDER = {"mask": None}

class CoreRingAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        if attention_mask is None:
            attention_mask = _INFER_MASK_HOLDER.get("mask")
        return super().__call__(attn, hidden_states, encoder_hidden_states,
                                encoder_hidden_states_mask, attention_mask, image_rotary_emb)


class CrossAttnGarmentProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Inference-side mirror of train.py's CrossAttnGarmentProcessor."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn  # noqa: F401
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
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        K_g  = _INF_CROSS_HOLDER.get("K_g")
        V_g  = _INF_CROSS_HOLDER.get("V_g")
        gate = _INF_CROSS_HOLDER.get("gate")
        if K_g is not None and V_g is not None:
            n_c = gate.shape[1] if gate is not None else 3072
            gk = K_g.unflatten(-1, (attn.heads, -1)).to(dtype=img_query.dtype)
            gv = V_g.unflatten(-1, (attn.heads, -1)).to(dtype=img_query.dtype)
            if attn.norm_k is not None:
                gk = attn.norm_k(gk)
            img_q_c = img_query[:, :n_c]
            cross_out_c = dispatch_attention_fn(
                img_q_c, gk, gv, attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            cross_out_c = cross_out_c.flatten(2, 3).to(img_query.dtype)
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

# Set by load_model based on config. 0=none, 1=block C-core→ring, 2=block both directions
ATTN_MASK_MODE = 0
PIN_RING_ROUGH = 0
USE_SIGMA_SCHED = 1
SIGMA_SCHED_LO = 0.6
SIGMA_SCHED_HI = 1.4
USE_PURE_NOISE = 1
USE_ROUGH_BLUR_FIXED = 0
ROUGH_BLUR_FIXED_SIG = 4.0
USE_ROUGH_MASKED = 0
USE_AGNOSTIC_MEAN_FILL = 0
USE_AGNOSTIC_ROUGH_FILL = 0
USE_AGNOSTIC_INPAINT = 1
AGNOSTIC_INPAINT_SOFT_SIG = 0.0
AGNOSTIC_ZERO_REPAIR = 0
SILHOUETTE_SCALE = 1.0
SILHOUETTE_SOFT = 0
USE_VAE_SILHOUETTE = 0
SILHOUETTE_SOFT_SIG = 2.0
ROUGH_MASK_SOFT = 0
ROUGH_MASK_SOFT_SIG = 3.0
USE_REPAIR_ATTN_MASK = 0
USE_BG_HINT = 0
BG_HINT_SCALE = 1.0
V6_ZERO_G_CORE = 0
V6_R_IN = 2
V6_R_OUT = 7
USE_AGN_CTRL = 0  # Set by train.py at save time
AGN_KEY_BIAS = 0  # Set by train.py at save time
AGN_V_SCALE = 0  # Set by train.py at save time
AGN_KEY_BIAS_ALPHA = 0.5  # Set by train.py at save time
AGN_TRUST_CORE = 0.3  # Set by train.py at save time
AGN_TRUST_BND = 0.85  # Set by train.py at save time
AGN_TRUST_KEEP = 1.0  # Set by train.py at save time
AGN_TRUST_EPS = 0.001  # Set by train.py at save time
AGN_VSCALE_CORE = 0.5  # Set by train.py at save time
AGN_VSCALE_BND = 0.9  # Set by train.py at save time
AGN_VSCALE_KEEP = 1.0  # Set by train.py at save time
AGN_TRUST_K = 3  # Set by train.py at save time
AGN_ERODE = 2  # Set by train.py at save time
AGN_DILATE = 2  # Set by train.py at save time
USE_INVALID_TOKEN = 0  # Set by train.py at save time
INVALID_TOKEN_K = 3  # Set by train.py at save time
INVALID_TOKEN_ERODE = 2  # Set by train.py at save time
INVALID_TOKEN_DILATE = 2  # Set by train.py at save time
INVALID_TOKEN_BND_VALID = 0.7  # Set by train.py at save time
USE_ZERO_AGNOSTIC_SLOT = 0  # Set by train.py at save time

# v22: learned [INVALID_AGNOSTIC] token, loaded from invalid_token.pt.
_INVALID_TOKEN_HOLDER = {}
_SIGMA_DUMP = []  # D5 diagnostic: per-step C_lat snapshots when SIGMA_DUMP_DIR is set

# v19/v20/v21: trust-map agnostic-slot suppression at inference (mirrors train.py).
_AGN_CTRL = {}


class AgnosticCtrlProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Inference-side twin of train.py's AgnosticCtrlProcessor. Applies the
    trust-map key bias / V scaling read from _AGN_CTRL."""
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

        agn_start = _AGN_CTRL.get("agn_img_start")
        n_agn     = _AGN_CTRL.get("n_agn")

        v_scale = _AGN_CTRL.get("v_scale_tok")
        if v_scale is not None and agn_start is not None:
            s = v_scale.to(img_value.dtype)[:, :, None, None]
            img_value = img_value.clone()
            img_value[:, agn_start:agn_start + n_agn] = \
                img_value[:, agn_start:agn_start + n_agn] * s

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        key_bias = _AGN_CTRL.get("key_bias_tok")
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


WRONG_POSE_MAP = {
    "00006_00": "00008_00",
    "00008_00": "00013_00",
    "00013_00": "00017_00",
    "00017_00": "00034_00",
    "00034_00": "00006_00",
}

# Training slot order for this run (indices into [agnostic, pose, rough, garment]).
# Patched by train.py at save time to match what the LoRA was trained with.
DEFAULT_SLOT_ORDER = [0, 3, 4]  # ['agnostic', 'garment', 'silhouette']


def _pack(lat, B, C, H, W):
    return lat.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

def _unpack(lat, B, C, H, W):
    return lat.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


# ── run07: V6Heads (matches trainlib/models.py) so deploy can apply trained bg/skin/route heads ──
import torch.nn as _v6nn
class V6Heads(_v6nn.Module):
    def __init__(self, hidden_dim=3072, packed_dim=64, n_classes=4, patch=2):
        super().__init__()
        self.to_s = _v6nn.Linear(hidden_dim, packed_dim, bias=True)
        self.to_b = _v6nn.Linear(hidden_dim, packed_dim, bias=True)
        self.to_route = _v6nn.Linear(hidden_dim, n_classes*patch*patch, bias=True)
    def forward(self, hidden):
        return {"delta_s_packed": self.to_s(hidden),
                "delta_b_packed": self.to_b(hidden),
                "route_logits":   self.to_route(hidden)}


_GN_VAE = None
_INF_GAR_HOLDER = {}
_INF_CROSS_HOLDER = {}
def _maybe_apply_output_space_garment_net(final_latents, batch, garment_net, wd):
    """If output-space garment net is loaded, decode final latents, apply
    the image-space correction gated by warped_pixel_mask, and return pred_image.
    Otherwise return {"pred_latents": final_latents}."""
    if garment_net is None or _GN_VAE is None or "garment_pixel" not in batch:
        return {"pred_latents": final_latents}
    # Detect output_space (decoder-style) vs norm_residual (Linear proj)
    if not hasattr(garment_net, "dec"):
        return {"pred_latents": final_latents}

    vae = _GN_VAE
    dev = next(vae.parameters()).device
    lat = final_latents.to(dev, dtype=wd)
    if lat.dim() == 4:
        lat = lat.unsqueeze(2)
    m = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(dev, wd)
    s = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to(dev, wd)
    with torch.no_grad():
        decoded = vae.decode(lat * s + m, return_dict=False)[0][:, :, 0]
    pred_img = decoded.clamp(-1, 1)                                          # (B, 3, Hi, Wi) in [-1, 1]

    # Garment-only correction
    gp = batch["garment_pixel"].to(dev, dtype=wd)
    if gp.dim() == 3: gp = gp.unsqueeze(0)
    with torch.no_grad():
        correction = garment_net(gp)                                          # (B, 3, 1024, 768)

    # Pixel-space soft gate by warped_fullres_mask
    wm_pix = batch.get("warped_fullres_mask")
    if wm_pix is None and "warped_mask" in batch:
        wm_pix = F.interpolate(batch["warped_mask"].to(dev, dtype=wd),
                                size=pred_img.shape[-2:], mode="nearest")
    if wm_pix is not None:
        wm_pix = wm_pix.to(dev, dtype=wd)
        if wm_pix.dim() == 3: wm_pix = wm_pix.unsqueeze(1)
        if wm_pix.shape[-2:] != pred_img.shape[-2:]:
            wm_pix = F.interpolate(wm_pix, size=pred_img.shape[-2:], mode="nearest")
        wm_b = (wm_pix > 0.5).to(wd)
        from torchvision.transforms.functional import gaussian_blur as _gbgn
        wm_soft = _gbgn(wm_b.float(), kernel_size=[7, 7], sigma=2.0).to(wd)
        pred_img = (pred_img + correction * wm_soft).clamp(-1, 1)
    else:
        pred_img = (pred_img + correction).clamp(-1, 1)

    img01 = (pred_img + 1.0) / 2.0
    return {"pred_image": img01[0].permute(1, 2, 0).float().cpu().numpy()}


# ─────────────────────────── load_model ───────────────────────────

def load_model(run_dir, device, config):
    td = torch.device(device); wd = torch.bfloat16

    t = QwenImageTransformer2DModel.from_pretrained(
        config.pretrained_model, subfolder="transformer", torch_dtype=wd)
    t = get_peft_model(
        t, LoraConfig(r=config.rank, lora_alpha=config.alpha,
                      init_lora_weights=config.init_lora_weights,
                      target_modules=config.lora_targets, lora_dropout=0.0),
        adapter_name="tryon")

    tryon_path = os.path.join(run_dir, "tryon_lora.safetensors")
    if os.path.exists(tryon_path):
        set_peft_model_state_dict(t, load_file(tryon_path), adapter_name="tryon")

    # Load trained proj_out / norm_out if present (UNFREEZE_PROJ_OUT runs)
    proj_out_path = os.path.join(run_dir, "proj_out_norm_out.pt")
    if os.path.exists(proj_out_path):
        sd = torch.load(proj_out_path, weights_only=True)
        _xfmr_inner = t.base_model.model if hasattr(t, "base_model") else t
        no_sd = {k[len("norm_out."):]: v for k, v in sd.items() if k.startswith("norm_out.")}
        po_sd = {k[len("proj_out."):]: v for k, v in sd.items() if k.startswith("proj_out.")}
        _xfmr_inner.norm_out.load_state_dict(no_sd)
        _xfmr_inner.proj_out.load_state_dict(po_sd)
        print(f"loaded proj_out_norm_out.pt ({len(no_sd)} norm_out keys, {len(po_sd)} proj_out keys)")

    # exp415: install custom attention processor if this run trained with mask mode
    if ATTN_MASK_MODE or USE_REPAIR_ATTN_MASK:
        n = 0
        for mod in t.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = CoreRingAttnProcessor()
                n += 1
        print(f"CoreRingAttnProcessor installed on {n} blocks (mode={ATTN_MASK_MODE}, repair_mask={USE_REPAIR_ATTN_MASK})")

    if USE_AGN_CTRL:
        n = 0
        for mod in t.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = AgnosticCtrlProcessor()
                n += 1
        print(f"AgnosticCtrlProcessor installed on {n} blocks "
              f"(key_bias={AGN_KEY_BIAS}, v_scale={AGN_V_SCALE})")

    t.to(td).eval()

    prompt_cache_raw = torch.load(os.path.join(run_dir, "prompt_cache.pt"), weights_only=True)
    pose_cache_raw   = torch.load(os.path.join(run_dir, "pose_latent_cache.pt"), weights_only=True)
    prompt_cache = {k: (v[0].to(td, dtype=wd), v[1].to(td, dtype=torch.long))
                    for k, v in prompt_cache_raw.items()}
    pose_cache   = {k: v.to(td, dtype=wd) for k, v in pose_cache_raw.items()}

    # Load v6 specialized heads if present
    repair_head = None
    routing_head = None
    rh_path = os.path.join(run_dir, "repair_head.pt")
    if os.path.exists(rh_path):
        import torch.nn as nn
        repair_head = nn.Conv2d(16, 16, 1, bias=True).to(td, dtype=wd)
        sd = torch.load(rh_path, weights_only=True)
        # state_dict saved from module with key "conv.weight"/"conv.bias"
        repair_head.load_state_dict({"weight": sd["conv.weight"], "bias": sd["conv.bias"]})
        repair_head.eval()
        print(f"loaded repair_head from {rh_path}")
    rt_path = os.path.join(run_dir, "routing_head.pt")
    if os.path.exists(rt_path):
        import torch.nn as nn
        routing_head = nn.Conv2d(16, 4, 1, bias=True).to(td, dtype=wd)
        sd = torch.load(rt_path, weights_only=True)
        routing_head.load_state_dict({"weight": sd["conv.weight"], "bias": sd["conv.bias"]})
        routing_head.eval()
        print(f"loaded routing_head from {rt_path}")

    # run07: load the trained V6Heads (to_s/to_b/to_route) for authoritative deploy composition
    v6_heads = None
    v6_path = os.path.join(run_dir, "v6_heads.pt")
    if int(os.environ.get("USE_V6_HEADS_DEPLOY", "0")) and os.path.exists(v6_path):
        v6_heads = V6Heads(hidden_dim=3072).to(td, dtype=wd)
        v6_heads.load_state_dict(torch.load(v6_path, weights_only=True))
        v6_heads.eval()
        print(f"[run07] loaded v6_heads from {v6_path} (USE_V6_HEADS_DEPLOY=1)")

    # v22: load the learned [INVALID_AGNOSTIC] token if present
    _it_path = os.path.join(run_dir, "invalid_token.pt")
    if USE_INVALID_TOKEN and os.path.exists(_it_path):
        _INVALID_TOKEN_HOLDER["token"] = torch.load(_it_path, weights_only=True).to(td, dtype=wd)
        print(f"loaded invalid_token from {_it_path}")

    # Load garment_net helper if present (autodetect mode by state-dict shape)
    garment_net = None
    gn_path = os.path.join(run_dir, "garment_net.pt")
    if os.path.exists(gn_path):
        import torch.nn as nn
        sd = torch.load(gn_path, weights_only=True)
        is_output_space = any(k.startswith("dec.") for k in sd.keys())
        is_adaln = any(k.startswith("to_gamma.") for k in sd.keys()) or any(k.startswith("to_beta.") for k in sd.keys())
        is_cross_attn = any(k.startswith("to_k_g.") for k in sd.keys()) or any(k.startswith("to_v_g.") for k in sd.keys())
        is_latent_residual = (any(k.startswith("out.") for k in sd.keys())
                              and not is_output_space and not is_adaln and not is_cross_attn)
        # Qwen-style transformer encoder uses patch_proj + blocks.* keys
        is_qwen_encoder = any(k.startswith("patch_proj.") for k in sd.keys()) or any(k.startswith("blocks.") for k in sd.keys())

        class _GarmentNet(nn.Module):
            """norm_residual mode (Linear proj to hidden_dim)."""
            def __init__(self, in_ch=3, hidden_dim=3072, ch_mult=(32, 64, 128, 256)):
                super().__init__()
                c1, c2, c3, c4 = ch_mult
                self.enc = nn.Sequential(
                    nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c4,    c4, 3, 1, 1), nn.GELU(),
                )
                self.proj = nn.Linear(c4, hidden_dim, bias=True)
            def forward(self, garment_pixel):
                x = garment_pixel * 2.0 - 1.0
                h = self.enc(x)
                B, Ch, H, W = h.shape
                h = h.flatten(2).transpose(1, 2).contiguous()
                return self.proj(h)

        class _GarmentNetOutput(nn.Module):
            """output_space mode (image-space ConvTranspose decoder)."""
            def __init__(self, in_ch=3, ch_mult=(32, 64, 128, 256)):
                super().__init__()
                c1, c2, c3, c4 = ch_mult
                self.enc = nn.Sequential(
                    nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c4,    c4, 3, 1, 1), nn.GELU(),
                )
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(c4, c3, 4, 2, 1), nn.GELU(),
                    nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.GELU(),
                    nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.GELU(),
                    nn.ConvTranspose2d(c1, in_ch, 4, 2, 1),
                )
            def forward(self, garment_pixel):
                x = garment_pixel * 2.0 - 1.0
                return self.dec(self.enc(x))

        class _GarmentLatentEnhancer(nn.Module):
            """Adds residual to garment_latent (16ch latent shape)."""
            def __init__(self, in_ch=3, ch_mult=(32, 64, 128, 256), out_ch=16):
                super().__init__()
                c1, c2, c3, c4 = ch_mult
                self.enc = nn.Sequential(
                    nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c3,    c4, 3, 1, 1), nn.GELU(),
                )
                self.out = nn.Conv2d(c4, out_ch, 1)
            def forward(self, garment_pixel):
                x = garment_pixel * 2.0 - 1.0
                return self.out(self.enc(x))

        class _GarmentNetAdaLN(nn.Module):
            """AdaLN modulation: produces (γ, β) per-token."""
            def __init__(self, in_ch=3, hidden_dim=3072, ch_mult=(32, 64, 128, 256)):
                super().__init__()
                c1, c2, c3, c4 = ch_mult
                self.enc = nn.Sequential(
                    nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c4,    c4, 3, 1, 1), nn.GELU(),
                )
                self.to_gamma = nn.Linear(c4, hidden_dim, bias=True)
                self.to_beta  = nn.Linear(c4, hidden_dim, bias=True)
            def forward(self, garment_pixel):
                x = garment_pixel * 2.0 - 1.0
                h = self.enc(x)
                B, Ch, H, W = h.shape
                h = h.flatten(2).transpose(1, 2).contiguous()
                return self.to_gamma(h), self.to_beta(h)

        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as _QwenBlock

        class _QwenGarmentNetAdaLN(nn.Module):
            def __init__(self, in_lat_ch=16, hidden_dim=3072, enc_dim=512, n_layers=2, num_heads=None, head_dim=None):
                super().__init__()
                if num_heads is None:
                    # 3072 → main-model geometry (24×128); 1024 → 16×64; <1024 → 8×head_dim/8
                    num_heads = 24 if enc_dim == 3072 else (16 if enc_dim >= 1024 else 8)
                if head_dim is None:
                    head_dim = enc_dim // num_heads
                self.enc_dim = enc_dim
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.out_norm = nn.RMSNorm(enc_dim)
                self.to_gamma = nn.Linear(enc_dim, hidden_dim, bias=True)
                self.to_beta  = nn.Linear(enc_dim, hidden_dim, bias=True)
            def forward(self, garment_latent):
                B, C, H, W = garment_latent.shape
                x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
                x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B, -1, -1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B, -1).contiguous()
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                x = self.out_norm(x)
                return self.to_gamma(x), self.to_beta(x)

        class _QwenGarmentNetCrossAttn(nn.Module):
            def __init__(self, in_lat_ch=16, inner_dim=3072, enc_dim=512, n_layers=2, num_heads=None, head_dim=None, n_tokens=192):
                super().__init__()
                if num_heads is None:
                    num_heads = 16 if enc_dim >= 1024 else 8
                if head_dim is None:
                    head_dim = enc_dim // num_heads
                self.enc_dim = enc_dim; self.n_tokens = n_tokens
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.out_norm = nn.RMSNorm(enc_dim)
                self.pool = nn.Linear(64 * 48, n_tokens, bias=True)
                self.to_k_g = nn.Linear(enc_dim, inner_dim, bias=True)
                self.to_v_g = nn.Linear(enc_dim, inner_dim, bias=True)
            def forward(self, garment_latent):
                B, C, H, W = garment_latent.shape
                x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
                x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B, -1, -1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B, -1).contiguous()
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                x = self.out_norm(x)
                x = x.transpose(1, 2)
                x = self.pool(x)
                x = x.transpose(1, 2)
                return self.to_k_g(x), self.to_v_g(x)

        class _GarmentNetCrossAttn(nn.Module):
            """Cross-attn IP-Adapter-style: encoder + shared to_k_g, to_v_g."""
            def __init__(self, in_ch=3, inner_dim=3072, ch_mult=(32, 64, 128, 256)):
                super().__init__()
                c1, c2, c3, c4 = ch_mult
                self.enc = nn.Sequential(
                    nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c4,    c4, 4, 2, 1), nn.GELU(),
                    nn.Conv2d(c4,    c4, 4, 2, 1), nn.GELU(),
                )
                self.tok_proj = nn.Linear(c4, inner_dim)
                self.to_k_g = nn.Linear(inner_dim, inner_dim)
                self.to_v_g = nn.Linear(inner_dim, inner_dim)
            def forward(self, garment_pixel):
                x = garment_pixel * 2.0 - 1.0
                h = self.enc(x).flatten(2).transpose(1, 2).contiguous()
                tokens = self.tok_proj(h)
                return self.to_k_g(tokens), self.to_v_g(tokens)

        if is_cross_attn:
            if is_qwen_encoder:
                enc_dim = sd["patch_proj.weight"].shape[0]
                n_layers = max([int(k.split(".")[1]) for k in sd.keys() if k.startswith("blocks.")] + [-1]) + 1
                garment_net = _QwenGarmentNetCrossAttn(inner_dim=t.inner_dim, enc_dim=enc_dim, n_layers=n_layers).to(td, dtype=wd).eval()
                garment_net.load_state_dict(sd)
                print(f"loaded qwen garment_net (cross_attn) enc_dim={enc_dim} layers={n_layers}")
            else:
                c1 = sd["enc.0.weight"].shape[0]
                c2 = sd["enc.2.weight"].shape[0]
                c3 = sd["enc.4.weight"].shape[0]
                c4 = sd["enc.6.weight"].shape[0]
                garment_net = _GarmentNetCrossAttn(inner_dim=t.inner_dim, ch_mult=(c1, c2, c3, c4)).to(td, dtype=wd).eval()
                garment_net.load_state_dict(sd)
            # Install CrossAttnGarmentProcessor on every attention block
            n_proc = 0
            for mod in t.modules():
                if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                    mod.processor = CrossAttnGarmentProcessor()
                    n_proc += 1
            print(f"loaded garment_net (cross_attn) from {gn_path}; installed processor on {n_proc} blocks")
        elif is_adaln:
            if is_qwen_encoder:
                enc_dim = sd["patch_proj.weight"].shape[0]
                n_layers = max([int(k.split(".")[1]) for k in sd.keys() if k.startswith("blocks.")] + [-1]) + 1
                garment_net = _QwenGarmentNetAdaLN(hidden_dim=t.inner_dim, enc_dim=enc_dim, n_layers=n_layers).to(td, dtype=wd).eval()
                garment_net.load_state_dict(sd)
                print(f"loaded qwen garment_net (adaln) enc_dim={enc_dim} layers={n_layers}")
            else:
                c1 = sd["enc.0.weight"].shape[0]
                c2 = sd["enc.2.weight"].shape[0]
                c3 = sd["enc.4.weight"].shape[0]
                c4 = sd["enc.6.weight"].shape[0]
                garment_net = _GarmentNetAdaLN(hidden_dim=t.inner_dim, ch_mult=(c1, c2, c3, c4)).to(td, dtype=wd).eval()
                garment_net.load_state_dict(sd)
            _INF_GAR_HOLDER.clear()
            def _gar_hook_adaln(module, inputs, output):
                gamma = _INF_GAR_HOLDER.get("gamma")
                beta  = _INF_GAR_HOLDER.get("beta")
                gate  = _INF_GAR_HOLDER.get("gate")
                if gamma is None or beta is None: return output
                n_c = gamma.shape[1]
                if output.shape[1] < n_c: return output
                g = gamma.to(dtype=output.dtype)
                b = beta.to(dtype=output.dtype)
                if int(os.environ.get("GARMENT_NET_BETA_ONLY", "0")):
                    g = torch.zeros_like(g)
                if int(os.environ.get("GARMENT_NET_GAMMA_ONLY", "0")):
                    b = torch.zeros_like(b)
                if gate is not None:
                    gt = gate.to(dtype=output.dtype)
                    g = g * gt
                    b = b * gt
                out = output.clone()
                out[:, :n_c, :] = out[:, :n_c, :] * (1.0 + g) + b
                return out
            t.norm_out.register_forward_hook(_gar_hook_adaln)
            print(f"loaded garment_net (adaln) from {gn_path}")
        elif is_latent_residual:
            garment_net = _GarmentLatentEnhancer().to(td, dtype=wd).eval()
            garment_net.load_state_dict(sd)
            print(f"loaded garment_net (garment_latent_residual) from {gn_path}")
        elif is_output_space:
            garment_net = _GarmentNetOutput().to("cuda:1", dtype=wd).eval()
            garment_net.load_state_dict(sd)
            # Output-space mode also needs VAE to decode pred_latents internally
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
            _vae_for_gn = AutoencoderKLQwenImage.from_pretrained(
                config.pretrained_model, subfolder="vae", torch_dtype=wd).to("cuda:1").eval()
            for p in _vae_for_gn.parameters(): p.requires_grad_(False)
            global _GN_VAE
            _GN_VAE = _vae_for_gn
            print(f"loaded garment_net (output_space) from {gn_path}; VAE loaded for in-process decode")
        else:
            # Detect channel multipliers from state dict (supports BIG variant)
            c1 = sd["enc.0.weight"].shape[0]
            c2 = sd["enc.2.weight"].shape[0]
            c3 = sd["enc.4.weight"].shape[0]
            c4 = sd["enc.6.weight"].shape[0]
            garment_net = _GarmentNet(hidden_dim=t.inner_dim, ch_mult=(c1, c2, c3, c4)).to(td, dtype=wd).eval()
            garment_net.load_state_dict(sd)
            _INF_GAR_HOLDER.clear()
            def _gar_hook(module, inputs, output):
                res = _INF_GAR_HOLDER.get("residual")
                if res is None: return output
                n_c = res.shape[1]
                if output.shape[1] < n_c: return output
                gate = _INF_GAR_HOLDER.get("gate")
                out = output.clone()
                add = res.to(dtype=out.dtype)
                if gate is not None:
                    add = add * gate.to(dtype=out.dtype)
                out[:, :n_c, :] = out[:, :n_c, :] + add
                return out
            t.norm_out.register_forward_hook(_gar_hook)
            print(f"loaded garment_net (norm_residual) from {gn_path}")

    # ── Garment cross-attn at proj_out (USE_GARMENT_XATTN path) ──
    # Mirrors train.py: encoder produces G (features), CrossAttn enhances F before
    # proj_out, gated by per-token garment mask. v6 heads still see un-enhanced F.
    garment_encoder = None
    garment_xattn_mod = None
    gx_enc_path = os.path.join(run_dir, "garment_encoder.pt")
    gx_xa_path  = os.path.join(run_dir, "garment_xattn.pt")
    if os.path.exists(gx_enc_path) and os.path.exists(gx_xa_path):
        import torch.nn as nn
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as _QwenBlock_xa

        class _QwenGarmentEncoder(nn.Module):
            def __init__(self, in_lat_ch=16, n_layers=2):
                super().__init__()
                enc_dim, num_heads, head_dim = 3072, 24, 128
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64*48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock_xa(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.out_norm = nn.RMSNorm(enc_dim)
            def forward(self, gl):
                B, C, H, W = gl.shape
                x = gl.unfold(2,2,2).unfold(3,2,2)
                x = x.permute(0,2,3,1,4,5).reshape(B, 64*48, C*4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B,-1,-1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B,-1).contiguous()
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                return self.out_norm(x)

        class _GarmentCrossAttn(nn.Module):
            def __init__(self, dim=3072, num_heads=24, head_dim=128, has_gate=False):
                super().__init__()
                self.num_heads = num_heads; self.head_dim = head_dim
                self.Wq = nn.Linear(dim, num_heads*head_dim, bias=True)
                self.Wk = nn.Linear(dim, num_heads*head_dim, bias=True)
                self.Wv = nn.Linear(dim, num_heads*head_dim, bias=True)
                self.out_proj = nn.Linear(num_heads*head_dim, dim, bias=True)
                if has_gate:
                    self.gate_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            def forward(self, F_h, G):
                B, N_F, _ = F_h.shape; N_G = G.shape[1]
                Q = self.Wq(F_h).view(B, N_F, self.num_heads, self.head_dim).transpose(1,2)
                K = self.Wk(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1,2)
                V = self.Wv(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1,2)
                A = F.scaled_dot_product_attention(Q, K, V)
                A = A.transpose(1,2).reshape(B, N_F, self.num_heads*self.head_dim)
                out = self.out_proj(A)
                if hasattr(self, "gate_logit"):
                    out = torch.sigmoid(self.gate_logit).to(out.dtype) * out
                return out

        sd_enc = torch.load(gx_enc_path, weights_only=True)
        n_layers_xa = max([int(k.split(".")[1]) for k in sd_enc.keys() if k.startswith("blocks.")] + [-1]) + 1
        garment_encoder = _QwenGarmentEncoder(n_layers=n_layers_xa).to(td, dtype=wd).eval()
        garment_encoder.load_state_dict(sd_enc)
        _gx_sd = torch.load(gx_xa_path, weights_only=True)
        _gx_has_gate = "gate_logit" in _gx_sd
        garment_xattn_mod = _GarmentCrossAttn(has_gate=_gx_has_gate).to(td, dtype=wd).eval()
        garment_xattn_mod.load_state_dict(_gx_sd)
        if _gx_has_gate:
            garment_xattn_mod.gate_logit.data = garment_xattn_mod.gate_logit.data.to(torch.float32)
        print(f"loaded garment_encoder ({n_layers_xa} layers) and garment_xattn from {run_dir}")

        # Pre-hook on proj_out: read holder and enhance F before proj_out runs.
        # proj_out input is (B, NUM_SLOTS*N_C, 3072); enhance C-slot only, leave the
        # conditioning slots untouched (downstream takes pred_C = out[:, :N_C, :]).
        def _proj_out_pre_hook_inf(module, args):
            holder = _INF_GAR_HOLDER
            if "G" not in holder or "p_g_tok" not in holder:
                return None
            F_full = args[0]
            N_C = holder.get("N_C", 3072)
            F_C = F_full[:, :N_C, :]
            A_g = garment_xattn_mod(F_C, holder["G"])
            gamma_inj = holder.get("gamma", 1.0)
            F_C_g = F_C + gamma_inj * holder["p_g_tok"] * A_g
            F_modified = torch.cat([F_C_g, F_full[:, N_C:, :]], dim=1)
            return (F_modified,) + args[1:]
        t.proj_out.register_forward_pre_hook(_proj_out_pre_hook_inf)
        print("registered forward_pre_hook on transformer.proj_out (garment xattn)")

    # ── ControlNet branch (USE_CONTROLNET) ──
    # Loads controlnet.pt + controlnet_meta.json. Per-block residuals from agnostic
    # are added to main blocks' hidden_states (C-slot tokens) via forward_hooks.
    # ── IntegratedRepairNet (USE_INTEGRATED_REPAIRNET) ──
    # Latent-space repair generator. Composition at inference:
    #   final = M_g * C_lat + ring * R_lat + M_k * al
    # where R_lat = (C_lat + RepairNet(agnostic, C_lat, M_g, ring, densepose)) * ring + 0.
    integrated_repairnet = None
    irn_path = os.path.join(run_dir, "integrated_repairnet.pt")
    if os.path.exists(irn_path):
        import torch.nn as nn
        class _IntegratedRepairNet(nn.Module):
            def __init__(self, in_ch=37, out_ch=16, base=64):
                super().__init__()
                self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(),
                                          nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
                self.down1 = nn.Conv2d(base, base*2, 4, stride=2, padding=1)
                self.enc2 = nn.Sequential(nn.SiLU(), nn.Conv2d(base*2, base*2, 3, padding=1), nn.SiLU(),
                                          nn.Conv2d(base*2, base*2, 3, padding=1), nn.SiLU())
                self.down2 = nn.Conv2d(base*2, base*4, 4, stride=2, padding=1)
                self.mid = nn.Sequential(nn.SiLU(), nn.Conv2d(base*4, base*4, 3, padding=1), nn.SiLU(),
                                         nn.Conv2d(base*4, base*4, 3, padding=1), nn.SiLU())
                self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1)
                self.dec2 = nn.Sequential(nn.SiLU(), nn.Conv2d(base*4, base*2, 3, padding=1), nn.SiLU(),
                                          nn.Conv2d(base*2, base*2, 3, padding=1), nn.SiLU())
                self.up1 = nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1)
                self.dec1 = nn.Sequential(nn.SiLU(), nn.Conv2d(base*2, base, 3, padding=1), nn.SiLU(),
                                          nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
                self.out = nn.Conv2d(base, out_ch, 3, padding=1)
            def forward(self, x):
                e1 = self.enc1(x); d1 = self.down1(e1)
                e2 = self.enc2(d1); d2 = self.down2(e2)
                m = self.mid(d2)
                u2 = self.up2(m); u2 = self.dec2(torch.cat([u2, e2], dim=1))
                u1 = self.up1(u2); u1 = self.dec1(torch.cat([u1, e1], dim=1))
                return self.out(u1)
        sd = torch.load(irn_path, weights_only=True)
        # Detect base channel count from first conv weight
        base = int(sd["enc1.0.weight"].shape[0])
        integrated_repairnet = _IntegratedRepairNet(in_ch=37, out_ch=16, base=base).to(td, dtype=wd).eval()
        integrated_repairnet.load_state_dict(sd)
        print(f"loaded IntegratedRepairNet ({sum(p.numel() for p in integrated_repairnet.parameters()):,} params)")

    controlnet_mod = None
    cn_path = os.path.join(run_dir, "controlnet.pt")
    cn_meta_path = os.path.join(run_dir, "controlnet_meta.json")
    if os.path.exists(cn_path) and os.path.exists(cn_meta_path):
        import torch.nn as nn
        import json as _json_cn
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as _QwenBlock_cn
        with open(cn_meta_path) as _f:
            _cn_meta = _json_cn.load(_f)
        cn_n_layers = int(_cn_meta.get("n_layers", 12))
        cn_inject_blocks = list(_cn_meta.get("inject_blocks", list(range(cn_n_layers))))

        class _QwenControlNet(nn.Module):
            def __init__(self, in_lat_ch=16, hidden_dim=3072, n_layers=12):
                super().__init__()
                enc_dim, num_heads, head_dim = hidden_dim, 24, 128
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed = nn.Parameter(torch.zeros(1, 64*48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock_cn(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.zero_projs = nn.ModuleList([
                    nn.Linear(enc_dim, hidden_dim, bias=True) for _ in range(n_layers)
                ])
            def forward(self, agn):
                B, C, H, W = agn.shape
                x = agn.unfold(2,2,2).unfold(3,2,2)
                x = x.permute(0,2,3,1,4,5).reshape(B, 64*48, C*4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B,-1,-1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B,-1).contiguous()
                residuals = []
                for blk, zproj in zip(self.blocks, self.zero_projs):
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                    residuals.append(zproj(x))
                return residuals

        controlnet_mod = _QwenControlNet(n_layers=cn_n_layers).to(td, dtype=wd).eval()
        controlnet_mod.load_state_dict(torch.load(cn_path, weights_only=True))
        print(f"loaded controlnet ({cn_n_layers} layers, inject {cn_inject_blocks}) from {cn_path}")

        # Holder for per-block residuals + N_C, populated each predict_sample call.
        _INF_CN_HOLDER = {}

        def _make_cn_hook_inf(block_idx):
            def hook(module, inputs, outputs):
                residual = _INF_CN_HOLDER.get(block_idx)
                if residual is None:
                    return outputs
                ehs, h = outputs[0], outputs[1]
                N_C = _INF_CN_HOLDER.get("N_C", residual.size(1))
                h_C = h[:, :N_C, :] + residual.to(h.dtype)
                h_new = torch.cat([h_C, h[:, N_C:, :]], dim=1)
                return (ehs, h_new)
            return hook

        for blk_i in cn_inject_blocks:
            t.transformer_blocks[blk_i].register_forward_hook(_make_cn_hook_inf(blk_i))
        # Expose holder + module via the model dict (return at end of load_model)
        # We store them on `t` itself (transformer) for retrieval in predict_sample.
        t._cn_holder = _INF_CN_HOLDER
        t._cn_mod = controlnet_mod
        t._cn_inject_blocks = cn_inject_blocks
        print(f"registered forward_hook on main blocks {cn_inject_blocks} (controlnet)")

    # ── v11 Qwen Latent Refiner ──
    qwen_refiner = None
    qr_path = os.path.join(run_dir, "qwen_refiner.pt")
    if os.path.exists(qr_path):
        import torch.nn as nn
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock as _QwenBlock_qr,
        )
        class _QwenLatentRefiner(nn.Module):
            def __init__(self, n_layers=4):
                super().__init__()
                enc_dim, num_heads, head_dim = 3072, 24, 128
                self.patch_proj = nn.Linear(33 * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64*48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock_qr(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.out_norm = nn.RMSNorm(enc_dim)
                self.to_residual = nn.Linear(enc_dim, 64, bias=True)
            def forward(self, pred_lat, gar_lat, wm):
                B, _, H, W = pred_lat.shape
                if wm.dim() == 3: wm = wm.unsqueeze(1)
                x_in = torch.cat([pred_lat, gar_lat, wm], dim=1)
                x = x_in.unfold(2,2,2).unfold(3,2,2)
                x = x.permute(0,2,3,1,4,5).reshape(B, 64*48, 33*4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B,-1,-1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B,-1).contiguous()
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                x = self.out_norm(x)
                return self.to_residual(x)
        sd_qr = torch.load(qr_path, weights_only=True)
        n_qr_layers = max([int(k.split(".")[1]) for k in sd_qr.keys() if k.startswith("blocks.")] + [-1]) + 1
        qwen_refiner = _QwenLatentRefiner(n_layers=n_qr_layers).to(td, dtype=wd).eval()
        qwen_refiner.load_state_dict(sd_qr)
        print(f"loaded qwen_refiner ({n_qr_layers} layers) from {qr_path}")

    # ── v8 Qwen Slot Enricher (USE_QWEN_SLOT_ENRICH path) ──
    qwen_slot_enricher = None
    qse_path = os.path.join(run_dir, "qwen_slot_enricher.pt")
    if os.path.exists(qse_path):
        import torch.nn as nn
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock as _QwenBlock_qse,
        )
        class _QwenSlotEnricher(nn.Module):
            def __init__(self, in_lat_ch=16, n_layers=4, slot_dim=64):
                super().__init__()
                enc_dim, num_heads, head_dim = 3072, 24, 128
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64*48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock_qse(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
                self.out_norm = nn.RMSNorm(enc_dim)
                self.to_slot_residual = nn.Linear(enc_dim, slot_dim, bias=True)
            def forward(self, gl):
                B, C, H, W = gl.shape
                x = gl.unfold(2,2,2).unfold(3,2,2)
                x = x.permute(0,2,3,1,4,5).reshape(B, 64*48, C*4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B,-1,-1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B,-1).contiguous()
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                return self.to_slot_residual(self.out_norm(x))
        sd_qse = torch.load(qse_path, weights_only=True)
        n_qse_layers = max([int(k.split(".")[1]) for k in sd_qse.keys() if k.startswith("blocks.")] + [-1]) + 1
        qwen_slot_enricher = _QwenSlotEnricher(n_layers=n_qse_layers).to(td, dtype=wd).eval()
        qwen_slot_enricher.load_state_dict(sd_qse)
        print(f"loaded qwen_slot_enricher ({n_qse_layers} layers) from {qse_path}")

    # ── OOTD-style garment branch (USE_GARMENT_OOTD path) ──
    garment_branch = None
    ootd_injectors = {}
    gb_path = os.path.join(run_dir, "garment_branch.pt")
    if os.path.exists(gb_path):
        import torch.nn as nn
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock as _QwenBlock_ootd,
            QwenDoubleStreamAttnProcessor2_0 as _QwenDSP_ootd,
            apply_rotary_emb_qwen as _apply_rope_ootd,
        )
        from diffusers.models.attention_dispatch import dispatch_attention_fn as _dispatch_ootd

        class _QwenGarmentBranch(nn.Module):
            """v2: zero-init pos/temb/dummy_text; returns POST-block outputs."""
            def __init__(self, in_lat_ch=16, n_layers=4):
                super().__init__()
                enc_dim, num_heads, head_dim = 3072, 24, 128
                self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
                self.pos_embed  = nn.Parameter(torch.zeros(1, 64*48, enc_dim))
                self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
                self.temb = nn.Parameter(torch.zeros(1, enc_dim))
                self.blocks = nn.ModuleList([
                    _QwenBlock_ootd(dim=enc_dim, num_attention_heads=num_heads, attention_head_dim=head_dim) for _ in range(n_layers)
                ])
            def precompute(self, gl):
                B, C, H, W = gl.shape
                x = gl.unfold(2,2,2).unfold(3,2,2)
                x = x.permute(0,2,3,1,4,5).reshape(B, 64*48, C*4)
                x = self.patch_proj(x) + self.pos_embed
                text = self.dummy_text.expand(B,-1,-1).contiguous()
                tmask = torch.ones(B, 1, dtype=torch.long, device=x.device)
                temb = self.temb.expand(B,-1).contiguous()
                outs = []
                for blk in self.blocks:
                    text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
                    outs.append(x)            # POST-block output
                return outs

        class _OOTDInjector(nn.Module):
            def __init__(self, dim=3072, num_heads=24, head_dim=128, has_gate=True):
                super().__init__()
                self.to_k_g = nn.Linear(dim, num_heads*head_dim, bias=True)
                self.to_v_g = nn.Linear(dim, num_heads*head_dim, bias=True)
                if has_gate:
                    self.gate_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        sd_gb = torch.load(gb_path, weights_only=True)
        n_layers_gb = max([int(k.split(".")[1]) for k in sd_gb.keys() if k.startswith("blocks.")] + [-1]) + 1
        garment_branch = _QwenGarmentBranch(n_layers=n_layers_gb).to(td, dtype=wd).eval()
        garment_branch.load_state_dict(sd_gb)
        print(f"loaded garment_branch ({n_layers_gb} layers) from {gb_path}")

        # Load injectors (variable already declared at outer scope so it survives
        # the if-block and is visible in predict_sample via the returned model dict)
        inj_path = os.path.join(run_dir, "ootd_injectors.pt")
        inject_indices = []
        if os.path.exists(inj_path):
            inj_save = torch.load(inj_path, weights_only=False)
            inject_indices = inj_save["inject_block_indices"]
            for blk_i in inject_indices:
                _sd = inj_save["injectors"][str(blk_i)]
                _has_gate = "gate_logit" in _sd
                inj = _OOTDInjector(has_gate=_has_gate).to(td, dtype=wd).eval()
                # Keep gate_logit in fp32 even after .to(bf16)
                if _has_gate:
                    inj.gate_logit.data = inj.gate_logit.data.to(torch.float32)
                inj.load_state_dict(_sd)
                if _has_gate:
                    inj.gate_logit.data = inj.gate_logit.data.to(torch.float32)
                ootd_injectors[blk_i] = inj
            print(f"loaded ootd_injectors for blocks {inject_indices} (has_gate={_has_gate if inject_indices else False})")

        # Install processor on injection blocks; processor reads injector + holder
        class _OOTDQwenAttnProcessorInf(_QwenDSP_ootd):
            def __init__(self, block_idx=0):
                super().__init__()
                self.block_idx = block_idx
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         encoder_hidden_states_mask=None, attention_mask=None,
                         image_rotary_emb=None):
                seq_txt = encoder_hidden_states.shape[1]
                img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
                img_key   = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
                img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
                txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
                txt_key   = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
                txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
                if attn.norm_q       is not None: img_query = attn.norm_q(img_query)
                if attn.norm_k       is not None: img_key   = attn.norm_k(img_key)
                if attn.norm_added_q is not None: txt_query = attn.norm_added_q(txt_query)
                if attn.norm_added_k is not None: txt_key   = attn.norm_added_k(txt_key)
                if image_rotary_emb is not None:
                    img_freqs, txt_freqs = image_rotary_emb
                    img_query = _apply_rope_ootd(img_query, img_freqs, use_real=False)
                    img_key   = _apply_rope_ootd(img_key,   img_freqs, use_real=False)
                    txt_query = _apply_rope_ootd(txt_query, txt_freqs, use_real=False)
                    txt_key   = _apply_rope_ootd(txt_key,   txt_freqs, use_real=False)
                joint_query = torch.cat([txt_query, img_query], dim=1)
                joint_key   = torch.cat([txt_key,   img_key],   dim=1)
                joint_value = torch.cat([txt_value, img_value], dim=1)
                jhs = _dispatch_ootd(joint_query, joint_key, joint_value,
                                     attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                     backend=self._attention_backend, parallel_config=self._parallel_config)
                # OOTD v5/v7: img-only + spatial mask + per-block gate + (optional) σ gate
                gar_h = _INF_GAR_HOLDER.get(f"ootd_h_{self.block_idx}",
                                            _INF_GAR_HOLDER.get("ootd_final_h"))
                p_g_tok = _INF_GAR_HOLDER.get("ootd_p_g_tok")
                injector = ootd_injectors.get(self.block_idx)
                if gar_h is not None and injector is not None:
                    gh = gar_h.to(dtype=img_query.dtype)
                    gar_key   = injector.to_k_g(gh).unflatten(-1, (attn.heads, -1))
                    gar_value = injector.to_v_g(gh).unflatten(-1, (attn.heads, -1))
                    if attn.norm_k is not None: gar_key = attn.norm_k(gar_key)
                    gar_attn_img = _dispatch_ootd(img_query, gar_key, gar_value,
                                                  attn_mask=None, dropout_p=0.0, is_causal=False,
                                                  backend=self._attention_backend, parallel_config=self._parallel_config)
                    if p_g_tok is not None:
                        B_, N_img_total = img_query.shape[0], img_query.shape[1]
                        N_C = p_g_tok.shape[1]
                        if N_img_total > N_C:
                            p_g_full = torch.cat([p_g_tok,
                                torch.zeros(B_, N_img_total - N_C, 1, device=p_g_tok.device, dtype=p_g_tok.dtype)], dim=1)
                        else:
                            p_g_full = p_g_tok
                        gar_attn_img = gar_attn_img * p_g_full.unsqueeze(-1).to(gar_attn_img.dtype)
                    if hasattr(injector, "gate_logit"):
                        gate_v = torch.sigmoid(injector.gate_logit).to(dtype=gar_attn_img.dtype)
                        gar_attn_img = gate_v * gar_attn_img
                    sigma_w_inf = _INF_GAR_HOLDER.get("ootd_sigma_w")
                    if sigma_w_inf is not None:
                        gar_attn_img = gar_attn_img * sigma_w_inf.view(-1, 1, 1, 1).to(dtype=gar_attn_img.dtype)
                    img_part = jhs[:, seq_txt:, :, :] + gar_attn_img
                    jhs = torch.cat([jhs[:, :seq_txt, :, :], img_part], dim=1)
                jhs = jhs.flatten(2, 3).to(joint_query.dtype)
                txt_out = jhs[:, :seq_txt, :]
                img_out = jhs[:, seq_txt:, :]
                img_out = attn.to_out[0](img_out)
                if len(attn.to_out) > 1: img_out = attn.to_out[1](img_out)
                txt_out = attn.to_add_out(txt_out)
                return img_out, txt_out

        # Install on configured injection blocks (NOT first N)
        _xfmr_inner = t.base_model.model if hasattr(t, "base_model") else t
        n_proc = 0
        for blk_i in inject_indices:
            mblk = _xfmr_inner.transformer_blocks[blk_i]
            for sub_mod in mblk.modules():
                if hasattr(sub_mod, "processor") and isinstance(sub_mod.processor, _QwenDSP_ootd):
                    sub_mod.processor = _OOTDQwenAttnProcessorInf(block_idx=blk_i)
                    n_proc += 1
        print(f"installed OOTDQwenAttnProcessorInf on {n_proc} attn (main blocks {inject_indices})")

    return {
        "transformer":     t,
        "prompt_cache":    prompt_cache,
        "pose_cache":      pose_cache,
        "repair_head":     repair_head,
        "routing_head":    routing_head,
        "v6_heads":        v6_heads,
        "garment_net":     garment_net,
        "garment_encoder": garment_encoder,
        "garment_xattn":   garment_xattn_mod,
        "garment_branch":  garment_branch,
        "ootd_injectors":  ootd_injectors,
        "qwen_slot_enricher": qwen_slot_enricher,
        "qwen_refiner":    qwen_refiner,
        "integrated_repairnet": integrated_repairnet,
        "config":          config,
        "device":          td,
        "weight_dtype":    wd,
    }


# ─────────────────────────── predict_sample ───────────────────────────

def predict_sample(model, batch, device, seed, settings):
    _SIGMA_DUMP.clear()
    t             = model["transformer"]
    prompt_cache  = model["prompt_cache"]
    pose_cache    = model["pose_cache"]
    repair_head   = model.get("repair_head")
    routing_head  = model.get("routing_head")
    garment_net   = model.get("garment_net")
    td            = model["device"]
    wd            = model["weight_dtype"]
    config        = model["config"]

    # Reset per-sample garment-net state
    _INF_GAR_HOLDER.clear()

    al = batch["agnostic_latent"].to(td, dtype=wd)
    # run07: register norm_out hook to capture the COND-pass hidden for v6 head deploy composition
    _v6_heads_deploy = model.get("v6_heads")
    _v6_hh = {}; _v6_hook_handle = None; _v6_last_cond_hidden = None
    if _v6_heads_deploy is not None:
        def _v6_capture(_m, _i, _o): _v6_hh["hidden"] = _o
        _v6_hook_handle = model["transformer"].norm_out.register_forward_hook(_v6_capture)
    gl = batch["garment_latent"].to(td, dtype=wd)
    rl = batch["rough_latent"].to(td, dtype=wd)
    am = batch["agnostic_mask_latent"].to(td, dtype=torch.float32)
    image_ids = batch.get("image_id", batch.get("image_ids"))
    if isinstance(image_ids, str): image_ids = [image_ids]

    B, C, H, W = al.shape

    # ── OOTD-style garment branch (USE_GARMENT_OOTD path, v5) ──
    # Per-block depth-specific hidden states (matches train.py v5 logic).
    garment_branch = model.get("garment_branch")
    _ootd_injectors = model.get("ootd_injectors", {})
    for _k in list(_INF_GAR_HOLDER.keys()):
        if _k.startswith("ootd_"):
            _INF_GAR_HOLDER.pop(_k, None)
    if garment_branch is not None:
        with torch.no_grad():
            _gb_lat = batch["garment_latent"].to(td, dtype=wd)
            if _gb_lat.dim() == 3: _gb_lat = _gb_lat.unsqueeze(0)
            _gb_per = garment_branch.precompute(_gb_lat)
            _INF_GAR_HOLDER["ootd_final_h"] = _gb_per[-1]
            # Depth-specific: zip sorted inject_indices with branch outputs
            _ootd_keys = sorted(_ootd_injectors.keys()) if _ootd_injectors else []
            for i, blk_i in enumerate(_ootd_keys):
                depth_idx = min(i, len(_gb_per) - 1)
                _INF_GAR_HOLDER[f"ootd_h_{blk_i}"] = _gb_per[depth_idx]
            # warped_mask packed to per-token for spatial selectivity at inference
            wm_o = batch.get("warped_mask")
            if wm_o is not None:
                wm_o = wm_o.to(td, dtype=wd)
                if wm_o.dim() == 3: wm_o = wm_o.unsqueeze(1)
                wm_o_b = (wm_o > 0.5).to(wd).expand(B, C, H, W)
                p_g_tok_o = _pack(wm_o_b, B, C, H, W).mean(dim=-1, keepdim=True)
                _INF_GAR_HOLDER["ootd_p_g_tok"] = p_g_tok_o

    # ── Garment cross-attn at proj_out (USE_GARMENT_XATTN path) ──
    # Compute G + p_g_tok per sample; the proj_out pre-hook (registered in load_model)
    # consumes them. Inference uses warped_mask as the gate (no GT silhouette available).
    garment_encoder = model.get("garment_encoder")
    garment_xattn_mod = model.get("garment_xattn")
    _INF_GAR_HOLDER.pop("G", None)
    _INF_GAR_HOLDER.pop("p_g_tok", None)
    _INF_GAR_HOLDER.pop("gamma", None)
    if garment_encoder is not None and garment_xattn_mod is not None:
        with torch.no_grad():
            _gx_lat = batch["garment_latent"].to(td, dtype=wd)
            if _gx_lat.dim() == 3: _gx_lat = _gx_lat.unsqueeze(0)
            _G_inf = garment_encoder(_gx_lat)                                          # (B, 3072, 3072)
            wm_gx = batch.get("warped_mask")
            if wm_gx is not None:
                wm_gx = wm_gx.to(td, dtype=wd)
                if wm_gx.dim() == 3: wm_gx = wm_gx.unsqueeze(1)
                _gx_mask = (wm_gx > 0.5).to(wd).expand(B, C, H, W)
                _p_g_tok_inf = _pack(_gx_mask, B, C, H, W).mean(dim=-1, keepdim=True)
            else:
                _p_g_tok_inf = torch.ones(B, 64*48, 1, device=td, dtype=wd)
            _INF_GAR_HOLDER["G"]       = _G_inf
            _INF_GAR_HOLDER["p_g_tok"] = _p_g_tok_inf
            _INF_GAR_HOLDER["gamma"]   = float(os.environ.get("GARMENT_XATTN_GAMMA", "1.0"))
            _INF_GAR_HOLDER["N_C"]     = 64 * 48

    # ── Run ControlNet branch on agnostic, populate per-block residuals into holder ──
    if hasattr(t, "_cn_mod") and t._cn_mod is not None:
        cn_holder = t._cn_holder
        cn_holder.clear()
        agn_lat_full = batch["agnostic_latent"].to(td, dtype=wd)
        if agn_lat_full.dim() == 3: agn_lat_full = agn_lat_full.unsqueeze(0)
        with torch.no_grad():
            cn_residuals = t._cn_mod(agn_lat_full)
        for blk_i, res in zip(t._cn_inject_blocks, cn_residuals):
            cn_holder[blk_i] = res
        cn_holder["N_C"] = 64 * 48

    # Compute garment_net outputs for hook-based modes (norm_residual / adaln / cross_attn).
    _gnet_does_residual = garment_net is not None and "garment_pixel" in batch and hasattr(garment_net, "proj") and not hasattr(garment_net, "to_k_g")
    _gnet_does_adaln    = garment_net is not None and "garment_pixel" in batch and hasattr(garment_net, "to_gamma")
    _gnet_does_cross    = garment_net is not None and "garment_pixel" in batch and hasattr(garment_net, "to_k_g")
    _INF_CROSS_HOLDER.clear()
    if _gnet_does_residual or _gnet_does_adaln or _gnet_does_cross:
        with torch.no_grad():
            # Qwen-style encoder consumes garment_latent (16,128,96); ConvNet consumes garment_pixel (3,1024,768).
            _is_qwen_enc = hasattr(garment_net, "patch_proj") and hasattr(garment_net, "blocks")
            if _is_qwen_enc:
                gp_imgs = batch["garment_latent"].to(td, dtype=wd)
                if gp_imgs.dim() == 3: gp_imgs = gp_imgs.unsqueeze(0)
            else:
                gp_imgs = batch["garment_pixel"].to(td, dtype=wd)
                if gp_imgs.dim() == 3: gp_imgs = gp_imgs.unsqueeze(0)
            if _gnet_does_residual:
                _INF_GAR_HOLDER["residual"] = garment_net(gp_imgs)
            elif _gnet_does_adaln:
                gamma, beta = garment_net(gp_imgs)
                _INF_GAR_HOLDER["gamma"] = gamma
                _INF_GAR_HOLDER["beta"]  = beta
            else:  # cross_attn
                K_g, V_g = garment_net(gp_imgs)
                _INF_CROSS_HOLDER["K_g"] = K_g
                _INF_CROSS_HOLDER["V_g"] = V_g
            # Spatial gate by warped_mask
            wm_g = batch.get("warped_mask")
            if wm_g is not None:
                wm_g = wm_g.to(td, dtype=wd)
                if wm_g.dim() == 3: wm_g = wm_g.unsqueeze(1)
                if int(os.environ.get("GARMENT_NET_GATE_SOFT", "0")):
                    wm_g_b = wm_g.clamp(0.0, 1.0).to(wd)
                else:
                    wm_g_b = (wm_g > 0.5).to(wd)
                _gate_dil = int(os.environ.get("GARMENT_NET_GATE_DILATE", "0"))
                if _gate_dil > 0:
                    wm_g_b = F.max_pool2d(wm_g_b, 2*_gate_dil+1, 1, _gate_dil).clamp(0, 1)
                wm_g_b = wm_g_b.expand(B, C, H, W)
                _gate_packed = _pack(wm_g_b, B, C, H, W).mean(dim=-1, keepdim=True)
                if _gnet_does_cross:
                    _INF_CROSS_HOLDER["gate"] = _gate_packed
                else:
                    _INF_GAR_HOLDER["gate"] = _gate_packed
            else:
                _INF_GAR_HOLDER.pop("gate", None)
    M = (am > 0.5).to(dtype=wd)

    # Mean-fill agnostic: mirror training-time substitution of grey mask with
    # the per-sample mean color of unmasked agnostic region.
    if USE_AGNOSTIC_MEAN_FILL:
        _M = M
        _unmasked_sum = (al * (1 - _M)).sum(dim=(-2, -1), keepdim=True)
        _unmasked_area = (1 - _M).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        _agn_mean = _unmasked_sum / _unmasked_area
        al = al * (1 - _M) + _agn_mean * _M

    # Rough-fill repair zone: agnostic's repair zone (M_full minus garment silhouette)
    # gets filled with rough's body estimate so the model sees a coherent input.
    if USE_AGNOSTIC_ROUGH_FILL:
        wm_rf = batch.get("warped_mask")
        if wm_rf is not None:
            wm_rf = wm_rf.to(td, dtype=wd)
            if wm_rf.dim() == 3: wm_rf = wm_rf.unsqueeze(1)
            _gar_bin_rf = (wm_rf > 0.5).to(wd)
            _repair_rf = (M - _gar_bin_rf).clamp(0, 1)
            al = al * (1 - _repair_rf) + rl * _repair_rf

    if USE_AGNOSTIC_INPAINT:
        from torchvision.transforms.functional import gaussian_blur as _gb
        _M_paste = M
        if AGNOSTIC_INPAINT_SOFT_SIG > 0:
            _k_sp = int(2 * round(2 * AGNOSTIC_INPAINT_SOFT_SIG) + 1)
            _M_paste = _gb(M.float(), kernel_size=[_k_sp, _k_sp],
                            sigma=AGNOSTIC_INPAINT_SOFT_SIG).to(M.dtype).clamp(0, 1)
        _keep = (1 - _M_paste)
        _result = al.clone()
        for _ in range(20):
            _blurred = _gb(_result.float(), kernel_size=[7, 7], sigma=2.0).to(_result.dtype)
            _result = _result * _keep + _blurred * _M_paste
        al = _result

    # v6: zero agnostic inside the confident garment core (M_core = erode(warped, r_in)).
    # Mirrors train-time V6_ZERO_G_CORE without needing target parse (uses eroded warped
    # as a safe proxy for M_g; parse could refine but not available at inference for
    # arbitrary samples). Removes torso template from input → model must synthesize.
    if V6_ZERO_G_CORE:
        wm_v6 = batch.get("warped_mask")
        if wm_v6 is not None:
            wm_v6 = wm_v6.to(td, dtype=wd)
            if wm_v6.dim() == 3: wm_v6 = wm_v6.unsqueeze(1)
            wm_bin_v6 = (wm_v6 > 0.5).to(wd)
            _r_in_v6 = V6_R_IN
            _core_v6 = -F.max_pool2d(-wm_bin_v6, 2*_r_in_v6+1, 1, _r_in_v6)
            _core_v6 = (_core_v6 > 0.5).to(wd)
            al = al * (1.0 - _core_v6)

    # exp552+: zero the agnostic in the repair band (M_full - warped_mask). Mirror
    # of train-time AGNOSTIC_ZERO_REPAIR. Uses warped_mask (inference-available).
    if AGNOSTIC_ZERO_REPAIR:
        wm_zr = batch.get("warped_mask")
        if wm_zr is not None:
            wm_zr = wm_zr.to(td, dtype=wd)
            if wm_zr.dim() == 3: wm_zr = wm_zr.unsqueeze(1)
            _wm_zr_bin = (wm_zr > 0.5).to(wd)
            _M_ag_bin = M.to(wd)
            _repair_proxy = (_M_ag_bin - _wm_zr_bin).clamp(0, 1)
            al = al * (1 - _repair_proxy)

    # exp419: no neutralization — feed raw agnostic and rough as conditioning.
    # The model learns routing from the soft-mask loss structure.

    pose_mode = settings.get("pose_mode", "normal")
    if not pose_cache:
        # Empty pose_cache (full-train mode): use zero placeholder. SLOT_ORDER
        # in baseline doesn't include "pose" so it's unused downstream.
        pose = torch.zeros(B, C, H, W, device=td, dtype=wd)
    else:
        pose_list = []
        for iid in image_ids:
            if pose_mode == "no_pose":
                pose_list.append(torch.zeros_like(pose_cache[iid]))
            elif pose_mode == "wrong_pose":
                wrong_id = WRONG_POSE_MAP.get(iid, iid)
                pose_list.append(pose_cache.get(wrong_id, pose_cache[iid]))
            else:
                pose_list.append(pose_cache[iid])
        pose = torch.stack(pose_list).to(td, dtype=wd)

    # ── Input ablation: zero out a named slot ──
    zero_slot = settings.get("zero_slot", None)
    if zero_slot == "agnostic": al   = torch.zeros_like(al)
    if zero_slot == "pose":     pose = torch.zeros_like(pose)
    if zero_slot == "rough":    rl   = torch.zeros_like(rl)
    if zero_slot == "garment":  gl   = torch.zeros_like(gl)

    # ── Garment-identity override: replace garment latent with arbitrary tensor ──
    if "garment_latent_override" in settings:
        gl = settings["garment_latent_override"].to(td, dtype=wd)

    # ── Slot order override: reorder the spatial-slot concatenation ──
    # settings["slot_order"] is a list of ints indexing into [agnostic, pose, rough, garment]
    # at positions 1..4 after C (position 0). Default = [0, 1, 2, 3].

    # Mirror training-time fixed rough blur, if configured
    if USE_ROUGH_BLUR_FIXED:
        from torchvision.transforms.functional import gaussian_blur
        bk = int(2 * round(2 * ROUGH_BLUR_FIXED_SIG) + 1)
        rl = gaussian_blur(rl.float(), kernel_size=[bk, bk], sigma=ROUGH_BLUR_FIXED_SIG).to(rl.dtype)

    if USE_ROUGH_MASKED:
        wm_r = batch.get("warped_mask")
        if wm_r is not None:
            wm_r = wm_r.to(td, dtype=wd)
            if wm_r.dim() == 3: wm_r = wm_r.unsqueeze(1)
            if ROUGH_MASK_SOFT:
                from torchvision.transforms.functional import gaussian_blur as _gbrm
                _k_r = int(2 * round(2 * ROUGH_MASK_SOFT_SIG) + 1)
                _wm_bin_r = (wm_r > 0.5).to(wd)
                _wm_soft_r = _gbrm(_wm_bin_r.float(), kernel_size=[_k_r, _k_r], sigma=ROUGH_MASK_SOFT_SIG).to(wd)
                rl = rl * _wm_soft_r
            else:
                rl = rl * (wm_r > 0.5).to(wd)

    agn_p   = _pack(al,   B, C, H, W)
    pose_p  = _pack(pose, B, C, H, W)
    rough_p = _pack(rl,   B, C, H, W)
    gar_p   = _pack(gl,   B, C, H, W)

    # v22: replace agnostic edit-core tokens with the learned [INVALID] token.
    if USE_INVALID_TOKEN and "token" in _INVALID_TOKEN_HOLDER:
        _Med = am.to(td, dtype=torch.float32).clamp(0, 1)
        if _Med.dim() == 3: _Med = _Med.unsqueeze(1)
        _Med = _Med[:, :1]
        _pad_it = INVALID_TOKEN_K // 2
        def _dil_it(x, it):
            for _ in range(it):
                x = F.max_pool2d(x, kernel_size=INVALID_TOKEN_K, stride=1, padding=_pad_it)
            return x.clamp(0, 1)
        _core_it = (1.0 - _dil_it(1.0 - _Med, INVALID_TOKEN_ERODE)).clamp(0, 1)
        _dilm_it = _dil_it(_Med, INVALID_TOKEN_DILATE).clamp(0, 1)
        _bnd_it  = (_dilm_it - _core_it).clamp(0, 1)
        _keep_it = (1.0 - _dilm_it).clamp(0, 1)
        _Aval = (0.0 * _core_it + INVALID_TOKEN_BND_VALID * _bnd_it + 1.0 * _keep_it).clamp(0, 1)
        _Aval_tok = _pack(_Aval.expand(B, 16, H, W), B, 16, H, W).mean(dim=-1, keepdim=True).to(agn_p.dtype)
        _e_inv = _INVALID_TOKEN_HOLDER["token"].to(agn_p.dtype)
        agn_p = _Aval_tok * agn_p + (1.0 - _Aval_tok) * _e_inv

    if USE_ZERO_AGNOSTIC_SLOT:
        agn_p = torch.zeros_like(agn_p)

    # v8 Qwen Slot Enricher: add Qwen-encoded residual to gar_p
    qwen_slot_enricher = model.get("qwen_slot_enricher")
    if qwen_slot_enricher is not None:
        with torch.no_grad():
            _qse_lat = batch["garment_latent"].to(td, dtype=wd)
            if _qse_lat.dim() == 3: _qse_lat = _qse_lat.unsqueeze(0)
            _slot_resid = qwen_slot_enricher(_qse_lat)   # (B, 3072, 64)
            gar_p = gar_p + _slot_resid
    # Silhouette slot: use warped_mask as explicit garment location signal
    sil_p = None
    if 4 in settings.get("slot_order", DEFAULT_SLOT_ORDER):
        if USE_VAE_SILHOUETTE:
            # Pre-computed VAE-encoded silhouette latent (in-distribution).
            vl = batch.get("warped_silhouette_latent")
            if vl is not None:
                vl = vl.to(td, dtype=wd)
                sil_p = _pack(vl, B, C, H, W)
        if sil_p is None:
            wm = batch.get("warped_mask")
            if wm is not None:
                wm = wm.to(td, dtype=wd)
                if wm.dim() == 3: wm = wm.unsqueeze(1)
                if SILHOUETTE_SOFT:
                    from torchvision.transforms.functional import gaussian_blur as _gb_sil
                    _k_s = int(2 * round(2 * SILHOUETTE_SOFT_SIG) + 1)
                    wm_bin = (wm > 0.5).to(wd)
                    wm_b = _gb_sil(wm_bin.float(), kernel_size=[_k_s, _k_s], sigma=SILHOUETTE_SOFT_SIG).to(wd)
                else:
                    wm_b = (wm > 0.5).to(wd)
                if USE_BG_HINT:
                    # ch 0 = garment silhouette, ch 1 = bg-hint (agnostic ∩ ¬body ∩ ¬warped)
                    _sil = torch.zeros((B, C, H, W), device=td, dtype=wd)
                    _sil[:, 0:1] = wm_b * SILHOUETTE_SCALE
                    dp = batch.get("densepose")
                    if dp is not None:
                        dp = dp.to(td, dtype=wd)
                        body_img = (dp.sum(dim=1, keepdim=True) > 0.02).to(wd)
                        body_lat = F.interpolate(body_img, size=(H, W), mode="area")
                        body_lat = (body_lat > 0.5).to(wd)
                        wm_bin_lat = (wm > 0.5).to(wd)
                        M_ag_bin = (M > 0.5).to(wd)
                        _bg = (M_ag_bin * (1.0 - body_lat) * (1.0 - wm_bin_lat)).clamp(0, 1)
                        _sil[:, 1:2] = _bg * BG_HINT_SCALE
                    sil_p = _pack(_sil, B, C, H, W)
                else:
                    wm_b = (wm_b * SILHOUETTE_SCALE).expand(B, C, H, W)
                    sil_p = _pack(wm_b, B, C, H, W)
    # Body-rough slot: rough * (1 - warped_mask) for explicit body/background context
    br_p = None
    if 5 in settings.get("slot_order", DEFAULT_SLOT_ORDER):
        wm = batch.get("warped_mask")
        if wm is not None:
            wm = wm.to(td, dtype=wd)
            if wm.dim() == 3: wm = wm.unsqueeze(1)
            wm_bin = (wm > 0.5).to(wd)
            br_p = _pack(rl * (1 - wm_bin), B, C, H, W)
    # img_shapes_base is filled after slot_order is resolved (variable length)

    pe_list = [prompt_cache[iid][0] for iid in image_ids]
    pm_list = [prompt_cache[iid][1] for iid in image_ids]
    max_txt = max(p.shape[1] for p in pe_list)
    pe_pad, pm_pad = [], []
    for pe, pm in zip(pe_list, pm_list):
        cur = pe.shape[1]
        if cur < max_txt:
            pe = torch.cat([pe, torch.zeros(1, max_txt - cur, pe.shape[-1],
                                            device=pe.device, dtype=pe.dtype)], dim=1)
            pm = torch.cat([pm, torch.zeros(1, max_txt - cur,
                                            device=pm.device, dtype=pm.dtype)], dim=1)
        pe_pad.append(pe); pm_pad.append(pm)
    pe = torch.cat(pe_pad, dim=0).to(td, dtype=wd)
    pm = torch.cat(pm_pad, dim=0).to(td, dtype=torch.long)
    txt_seq_lens = pm.sum(dim=1).tolist()

    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")
    isl = (H//2) * (W//2)
    sl  = (sch.config.max_shift - sch.config.base_shift) / (sch.config.max_image_seq_len - sch.config.base_image_seq_len)
    mu  = isl * sl + (sch.config.base_shift - sl * sch.config.base_image_seq_len)
    sch.set_timesteps(settings.get("num_inference_steps", 50), mu=mu)

    g = torch.Generator(device=td).manual_seed(seed)
    noise = torch.randn(al.shape, device=td, dtype=wd, generator=g)

    C_lat = noise if USE_PURE_NOISE else (1 - M) * al + M * noise

    slot_order   = settings.get("slot_order", DEFAULT_SLOT_ORDER)

    # Compute garment_net output once and route by output shape:
    #   shape (B, 3072, 64) → extra_slot mode (append a new slot)
    #   shape (B, 16, 128, 96) → garment_latent_residual mode (delta on gl)
    if garment_net is not None and "garment_pixel" in batch and hasattr(garment_net, "out"):
        with torch.no_grad():
            _gp_test = batch["garment_pixel"].to(td, dtype=wd)
            if _gp_test.dim() == 3: _gp_test = _gp_test.unsqueeze(0)
            _out = garment_net(_gp_test)
            if _out.dim() == 3 and _out.shape[-1] == C * 4:
                _INF_GAR_HOLDER["extra_slot"] = _out
            elif _out.dim() == 4 and _out.shape[1] == C:
                gl = gl + _out
    n_extra_slots = 1 if "extra_slot" in _INF_GAR_HOLDER else 0
    img_shapes_base = [(1, H//2, W//2)] * (1 + len(slot_order) + n_extra_slots)

    cfg_scale = settings.get("cfg_scale", 1.0)

    # Build attention mask once (depends on M and warped_mask, both static across steps)
    if USE_REPAIR_ATTN_MASK:
        # M is agnostic_mask_latent at latent res, shape (B, 1, H, W)
        # wm is warped_mask_128 at latent res, also (B, 1, H, W)
        wm_for_mask = batch.get("warped_mask")
        if wm_for_mask is not None:
            wm_for_mask = wm_for_mask.to(td, dtype=wd)
            if wm_for_mask.dim() == 3: wm_for_mask = wm_for_mask.unsqueeze(1)
            # agnostic slot index in seq: position 1 (right after C) for the standard SLOT_ORDER
            agn_slot_pos = 1 if 0 in slot_order else 1
            wm_bin = (wm_for_mask > 0.5).to(M.dtype)
            repair_band = (M - wm_bin).clamp(0, 1)
            keep_mask = (1.0 - M).clamp(0, 1)
            repair_tok = _pack(repair_band.expand(B, C, H, W), B, C, H, W).mean(dim=-1)
            keep_tok   = _pack(keep_mask.expand(B, C, H, W),   B, C, H, W).mean(dim=-1)
            repair_pos = (repair_tok > 0.5)
            keep_pos   = (keep_tok > 0.5)
            txt_len = pe.shape[1]
            img_tok = (H // 2) * (W // 2)
            num_slots = 1 + len(slot_order)
            total_seq = txt_len + num_slots * img_tok
            inf_mask = torch.zeros(B, 1, total_seq, total_seq, device=td, dtype=wd)
            for b in range(B):
                q_idx = txt_len + torch.where(repair_pos[b])[0]
                k_idx = txt_len + agn_slot_pos * img_tok + torch.where(keep_pos[b])[0]
                if q_idx.numel() > 0 and k_idx.numel() > 0:
                    inf_mask[b, 0, q_idx.unsqueeze(1), k_idx.unsqueeze(0)] = -1e4
            _INFER_MASK_HOLDER["mask"] = inf_mask

    # v19/v20/v21: populate _AGN_CTRL trust map (static across denoise steps)
    _AGN_CTRL.clear()
    if USE_AGN_CTRL:
        _M_edit_ac = am.to(td, dtype=torch.float32).clamp(0, 1)
        if _M_edit_ac.dim() == 3: _M_edit_ac = _M_edit_ac.unsqueeze(1)
        _M_edit_ac = _M_edit_ac[:, :1]
        _pad_ac = AGN_TRUST_K // 2
        def _dil_ac(x, it):
            for _ in range(it):
                x = F.max_pool2d(x, kernel_size=AGN_TRUST_K, stride=1, padding=_pad_ac)
            return x
        _core_ac = (1.0 - _dil_ac(1.0 - _M_edit_ac, AGN_ERODE)).clamp(0, 1)
        _dilm_ac = _dil_ac(_M_edit_ac, AGN_DILATE).clamp(0, 1)
        _bnd_ac  = (_dilm_ac - _core_ac).clamp(0, 1)
        _keep_ac = (1.0 - _dilm_ac).clamp(0, 1)
        def _tok_ac(m):
            return _pack(m.expand(B, 16, H, W), B, 16, H, W).mean(dim=-1)
        _core_t = _tok_ac(_core_ac); _bnd_t = _tok_ac(_bnd_ac); _keep_t = _tok_ac(_keep_ac)
        _tok_per_slot = _core_t.shape[1]
        # agnostic is the first conditioning slot after C (position 1)
        _AGN_CTRL["agn_img_start"] = 1 * _tok_per_slot
        _AGN_CTRL["n_agn"] = _tok_per_slot
        if AGN_KEY_BIAS:
            _A_tok = (AGN_TRUST_CORE * _core_t + AGN_TRUST_BND * _bnd_t
                      + AGN_TRUST_KEEP * _keep_t).clamp(0, 1)
            _AGN_CTRL["key_bias_tok"] = AGN_KEY_BIAS_ALPHA * torch.log(_A_tok + AGN_TRUST_EPS)
        if AGN_V_SCALE:
            _AGN_CTRL["v_scale_tok"] = (AGN_VSCALE_CORE * _core_t + AGN_VSCALE_BND * _bnd_t
                                        + AGN_VSCALE_KEEP * _keep_t).clamp(0, 1)

    # ONE denoising path. Default grad_start_step=None -> every step under no_grad
    # (byte-identical to before). The GAN rollout passes grad_start_step=12 so the
    # prefix (0..11) stays no_grad/detached and the tail (12..19) is differentiable.
    _grad_start = settings.get("grad_start_step", None)
    for _i, ts in enumerate(sch.timesteps):
        with torch.set_grad_enabled(bool(_grad_start is not None and _i >= _grad_start)), torch.amp.autocast("cuda", dtype=wd):
            C_p    = _pack(C_lat, B, C, H, W)
            sig    = (ts / 1000).to(device=td, dtype=wd).expand(B)

            # σ-conditional gating for OOTD (matches train.py GARMENT_OOTD_SIGMA_GATE)
            if int(os.environ.get("GARMENT_OOTD_SIGMA_GATE", "0")):
                _thr = float(os.environ.get("GARMENT_OOTD_SIGMA_THR", "0.3"))
                _sigma_w = ((sig.float() - _thr) / max(1.0 - _thr, 1e-6)).clamp(0.0, 1.0)
                _INF_GAR_HOLDER["ootd_sigma_w"] = _sigma_w.to(wd)
            else:
                _INF_GAR_HOLDER.pop("ootd_sigma_w", None)

            # Sigma-scheduled conditioning scales (must match training)
            if USE_SIGMA_SCHED:
                s_val = (ts / 1000).to(device=td, dtype=wd)
                _span = SIGMA_SCHED_HI - SIGMA_SCHED_LO
                struct_scale = (SIGMA_SCHED_LO + _span * s_val).to(agn_p.dtype)
                detail_scale = (SIGMA_SCHED_HI - _span * s_val).to(gar_p.dtype)
                agn_p_s   = agn_p   * struct_scale
                pose_p_s  = pose_p  * struct_scale
                rough_p_s = rough_p * detail_scale
                gar_p_s   = gar_p   * detail_scale
            else:
                agn_p_s, pose_p_s, rough_p_s, gar_p_s = agn_p, pose_p, rough_p, gar_p

            slot_tensors = [agn_p_s, pose_p_s, rough_p_s, gar_p_s, sil_p, br_p]
            cond_seq = [slot_tensors[i] for i in slot_order]
            # Append extra-slot from garment_net (paradigm #4) if present.
            extra_slot_tok = _INF_GAR_HOLDER.get("extra_slot")
            if extra_slot_tok is not None:
                cond_seq = cond_seq + [extra_slot_tok]
            if cfg_scale != 1.0:
                uncond_tensors = [agn_p_s, pose_p_s, rough_p_s, torch.zeros_like(gar_p_s), sil_p, br_p]
                uncond_seq = [uncond_tensors[i] for i in slot_order]
                if extra_slot_tok is not None:
                    uncond_seq = uncond_seq + [torch.zeros_like(extra_slot_tok)]

            n_extra = 1 if extra_slot_tok is not None else 0
            hidden = torch.cat([C_p] + cond_seq, dim=1)
            out = t(
                hidden_states              = hidden,
                timestep                   = sig,
                encoder_hidden_states      = pe,
                encoder_hidden_states_mask = pm,
                img_shapes                 = [img_shapes_base] * B,
                txt_seq_lens               = txt_seq_lens,
                return_dict                = False,
            )[0]
            v_cond = out[:, :C_p.size(1), :]
            if _v6_heads_deploy is not None:            # run07: stash COND-pass hidden (last step wins)
                _v6_last_cond_hidden = _v6_hh.get("hidden")

            # v11 Qwen Latent Refiner — apply post-pred residual on garment region
            qwen_refiner = model.get("qwen_refiner")
            if qwen_refiner is not None:
                _pred_lat = _unpack(v_cond, B, C, H, W)
                _gar_lat = batch["garment_latent"].to(td, dtype=wd)
                if _gar_lat.dim() == 3: _gar_lat = _gar_lat.unsqueeze(0)
                _wm = batch.get("warped_mask")
                if _wm is not None:
                    _wm = _wm.to(td, dtype=wd)
                    if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
                    _wm_b = (_wm > 0.5).to(wd)
                    _resid = qwen_refiner(_pred_lat, _gar_lat, _wm_b)
                    _wm_e = _wm_b.expand(B, C, H, W)
                    _wm_packed = _pack(_wm_e, B, C, H, W).mean(dim=-1, keepdim=True)
                    v_cond = v_cond + _resid * _wm_packed.to(v_cond.dtype)

            if cfg_scale != 1.0:
                hidden_u = torch.cat([C_p] + uncond_seq, dim=1)
                out_u = t(
                    hidden_states              = hidden_u,
                    timestep                   = sig,
                    encoder_hidden_states      = pe,
                    encoder_hidden_states_mask = pm,
                    img_shapes                 = [img_shapes_base] * B,
                    txt_seq_lens               = txt_seq_lens,
                    return_dict                = False,
                )[0]
                v_uncond = out_u[:, :C_p.size(1), :]
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            C_p    = sch.step(v_pred, ts, C_p, return_dict=False)[0]
            C_lat  = _unpack(C_p, B, C, H, W)
            # Keep last step's unpacked v_pred for v6 repair head
            v_pred_lat_last = _unpack(v_pred, B, C, H, W)
            if os.environ.get("SIGMA_DUMP_DIR"):
                _SIGMA_DUMP.append((float((ts/1000).item()) if hasattr(ts, 'item') else float(ts),
                                    C_lat.detach().clone()))

    # Pixel-space paste-back mode (RAW_FULL_PRED=1): return the full-frame prediction;
    # caller decodes + pastes the agnostic hole over real pixels (no latent-composite bands).
    if int(os.environ.get("RAW_FULL_PRED", "0")):
        # run07: authoritative v6 composition by region before decode.
        #   final = route[gar]*C_lat + route[skin]*(al+δ_s) + route[bg]*(al+δ_b) + route[keep]*al
        if _v6_heads_deploy is not None and _v6_last_cond_hidden is not None:
            _hc = _v6_last_cond_hidden[:, :C_p.size(1), :].to(wd)
            _v6o = _v6_heads_deploy(_hc)
            _ds = _unpack(_v6o["delta_s_packed"], B, C, H, W)
            _db = _unpack(_v6o["delta_b_packed"], B, C, H, W)
            _rp = _v6o["route_logits"]; _H2, _W2 = H // 2, W // 2
            _route = _rp.view(B, _H2, _W2, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 4, H, W)
            _rs = F.softmax(_route.float(), dim=1).to(wd)          # [gar, skin, bg, keep]
            # ── GATED composition (edit-zone only) ──
            _wm = batch.get("warped_mask")
            if _wm is None: raise RuntimeError("v6 gated compose requires warped_mask")
            _wm = _wm.to(td, dtype=wd)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            if tuple(_wm.shape[-2:]) != (H, W): _wm = F.interpolate(_wm, size=(H, W), mode="nearest")
            _M_g   = (_wm > 0.5).to(wd)                                          # garment
            _M_edit = F.max_pool2d(_M_g, 2*V6_R_OUT+1, 1, V6_R_OUT).clamp(0, 1)  # dilated edit zone
            _M_repair = (_M_edit - _M_g).clamp(0, 1)                             # non-garment repair inside edit
            # route-blend skin vs bg inside M_repair (renormalized over skin+bg)
            _skin_w, _bg_w = _rs[:, 1:2], _rs[:, 2:3]
            _den = (_skin_w + _bg_w).clamp(min=1e-6)
            _v6_blend = (_skin_w * (al + _ds) + _bg_w * (al + _db)) / _den
            # final = garment→C_lat, repair→v6 blend, outside edit→al (preserved)
            _final_v6 = C_lat * _M_g + _v6_blend * _M_repair + al * (1.0 - _M_edit)
            if _v6_hook_handle is not None: _v6_hook_handle.remove()
            if int(os.environ.get("V6_DEPLOY_DEBUG", "0")):
                # route % INSIDE edit only + outside-edit bg-route leakage check
                _me = _M_edit; _men = _me.sum().clamp(min=1.0)
                _in = [round(float(((_rs[:,k:k+1]*_me).sum()/_men)), 3) for k in range(4)]
                _out_bg = float((_rs[:,2:3] * (1.0 - _me)).sum() / ((1.0 - _me).sum().clamp(min=1.0)))
                print(f"[run07 v6 GATED] route%_in_edit gar/skin/bg/keep={_in} "
                      f"M_edit_frac={float(_me.mean()):.3f} M_repair_frac={float(_M_repair.mean()):.3f} "
                      f"outside_edit_bg_applied={_out_bg:.4f}(should~0) "
                      f"|final-C|={float((_final_v6-C_lat).abs().mean()):.4f}", flush=True)
            return {"pred_latents": _final_v6, "raw_full_pred": True}
        if _v6_hook_handle is not None: _v6_hook_handle.remove()
        return {"pred_latents": C_lat, "raw_full_pred": True}

    # ── IntegratedRepairNet composition (USE_INTEGRATED_REPAIRNET path) ──
    # If integrated_repairnet is loaded, compose:
    #   final = M_g * C_lat + ring * (C_lat + RepairNet(...)) + M_k * al
    # RepairNet output is hard-masked to ring at training; mirror that here.
    integrated_repairnet = model.get("integrated_repairnet")
    if integrated_repairnet is not None:
        wm_irn = batch.get("warped_mask")
        if wm_irn is None:
            raise RuntimeError("integrated_repairnet requires warped_mask in batch")
        wm_irn = wm_irn.to(td, dtype=wd)
        if wm_irn.dim() == 3: wm_irn = wm_irn.unsqueeze(1)
        wm_irn_b = (wm_irn > 0.5).to(wd)
        # Use the WIDE agnostic_mask as the inpaint zone (not dilate(warped, V6_R_OUT)).
        # This way the keep region uses the actual source image pixels, not grey-fill.
        am_irn = batch.get("agnostic_mask_latent")
        if am_irn is None:
            raise RuntimeError("integrated_repairnet requires agnostic_mask_latent in batch")
        am_irn = am_irn.to(td, dtype=wd)
        if am_irn.dim() == 3: am_irn = am_irn.unsqueeze(1)
        if am_irn.dim() == 4 and am_irn.shape[1] != 1:
            am_irn = am_irn.unsqueeze(1)
        M_edit_irn = (am_irn > 0.5).to(wd)
        M_g_irn = wm_irn_b
        ring_irn = (M_edit_irn - M_g_irn).clamp(0, 1)
        M_k_irn = (1.0 - M_edit_irn).clamp(0, 1)
        # Densepose at latent res
        H_l, W_l = C_lat.shape[-2], C_lat.shape[-1]
        if "densepose" in batch:
            _dp = batch["densepose"].to(td, dtype=wd)
            if _dp.dim() == 3: _dp = _dp.unsqueeze(0)
            if _dp.shape[-2:] != (H_l, W_l):
                _dp = F.interpolate(_dp, size=(H_l, W_l), mode="bilinear", align_corners=False)
        else:
            _dp = torch.zeros(C_lat.size(0), 3, H_l, W_l, device=td, dtype=wd)
        rnet_in = torch.cat([al, C_lat, M_g_irn, ring_irn, _dp], dim=1)
        with torch.no_grad():
            repair_residual = integrated_repairnet(rnet_in) * ring_irn
        R_lat = C_lat + repair_residual
        final = M_g_irn * C_lat + ring_irn * R_lat + M_k_irn * al
        return {"pred_latents": final}

    # ── v6 composition path: use specialized heads to compose final latent ──
    # USE_V6_COMPOSE=0 falls through to the main-pred-only legacy soft composite,
    # using main transformer's pred for the entire mask region (no v6 deltas).
    if repair_head is not None and int(os.environ.get("USE_V6_COMPOSE", "1")):
        wm_v6c = batch.get("warped_mask")
        if wm_v6c is None:
            raise RuntimeError("v6 inference requires warped_mask in batch")
        wm_v6c = wm_v6c.to(td, dtype=wd)
        if wm_v6c.dim() == 3: wm_v6c = wm_v6c.unsqueeze(1)
        wm_bin_c = (wm_v6c > 0.5).to(wd)
        # M_edit (dilate), M_g = warped, ring = M_edit - M_g, M_k = 1 - M_edit
        M_edit_c = F.max_pool2d(wm_bin_c, 2*V6_R_OUT+1, 1, V6_R_OUT).clamp(0, 1)
        M_g_c    = wm_bin_c
        ring_c   = (M_edit_c - M_g_c).clamp(0, 1)
        M_k_c    = (1.0 - M_edit_c).clamp(0, 1)
        # δ from repair head on last v_pred
        with torch.no_grad():
            delta_c = repair_head(v_pred_lat_last.to(wd))
        x_repair_c = al + delta_c                                              # source + residual
        # Compose: garment from denoised latent, ring from source+δ, keep from source
        final = M_g_c * C_lat + ring_c * x_repair_c + M_k_c * al
        return _maybe_apply_output_space_garment_net(final, batch, garment_net, wd)

    # Legacy soft composite path (non-v6)
    from torchvision.transforms.functional import gaussian_blur as _gb_final
    _comp_sig = float(os.environ.get("COMPOSITE_SIGMA", "2.0"))
    _comp_k = int(2 * round(2 * _comp_sig) + 1) if _comp_sig >= 2.0 else 7
    _M_blur = _gb_final(M.float(), kernel_size=[_comp_k, _comp_k], sigma=_comp_sig).to(M.dtype)
    _M_soft = _M_blur
    final = (1 - _M_soft) * al + _M_soft * C_lat
    return {"pred_latents": final}
