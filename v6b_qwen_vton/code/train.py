#!/usr/bin/env python3
"""
VTON Autoresearch — exp401: exp395 + rough_latent as a fifth spatial slot.

exp395 (current best, main 35.55) used 4 spatial slots [C, agnostic, pose, garment]
with zero rough_latent involvement. exp401 mimics exp395 exactly and adds the degraded
rough try-on latent as an additional spatial condition at position 3 of the image
sequence. Clean ablation: stock Qwen attention, no custom processors, no garment net,
no CNN, only the tryon LoRA is trainable.

Sequence layout:
    position 0  C                    (B, 16, 128, 96)  ← noisy denoising target (masked flow)
    position 1  agnostic_latent      (B, 16, 128, 96)
    position 2  pose_latent          (B, 16, 128, 96)  ← VAE-encoded densepose RGB
    position 3  degraded_rough_latent (B, 16, 128, 96) ← NEW vs exp395, from _degraded_rough_latent.pt
    position 4  garment_latent       (B, 16, 128, 96)

Token counts: 3072 × 5 = 15,360 image tokens.

Trainable: only the tryon LoRA.
Loss: exp395's 3-region weighted latent MSE + image L1 (0.1) + masked flow.
VL prompt: [agnostic_pil, pose_pil, rough_pil, garment_pil] + fixed prompt.

Mask partition (per-sample):
    M_core    = target_mask_latent   (tight garment silhouette — what the evaluator scores)
    M_full    = agnostic_mask_latent (broad inpaint region, core + repair ring)
    M_repair  = (M_full - M_core).clamp(0, 1)
    M_outside = 1 - M_full

Region weights:
    core    : 1.0    (highest — the garment itself)
    repair  : 0.3    (medium — arm/neck/shoulder boundary repair)
    outside : 0.05   (keep loss — penalize any drift in unchanged regions)
Plus explicit core-boundary emphasis: +1.0 on the 7×7 dilation ring around core.

Forward process:
    C_t      = person + s * M_full * (noise - person)   (full-agnostic generation)
    v_target = M_full * (noise - person)

Latent loss:
    weight_map = 0.05 + 0.95*M_core + 0.25*M_repair + 1.0*M_core_boundary
    L_latent   = mean( (pred_C - pack(v_target))^2 * pack(weight_map).mean(-1) )

Image L1 loss (same region weighting at image resolution):
    weight_map_img = interp to (H_img, W_img)
    L_image = mean( |VAE_decode(x0_pred) - cached_person_image| * weight_map_img )

Total: L = L_latent + 0.1 * L_image

Semantic conditioning (Qwen2.5-VL via QwenImageEditPlusPipeline.encode_prompt):
    Per-sample, cached at startup, using 4 standard-aspect source pictures
    [agnostic_pil, pose_pil, rough_pil, garment_pil] + fixed prompt.

Flow matching:
    sigma ~ Beta(1, 1)
    noise = randn_like(person_latent)
    M     = (agnostic_mask_latent > 0.5)
    C_t      = person + s * M * (noise - person)
    v_target = M * (noise - person)
    pred_C   = out[:, :3072, :]
    loss     = mean((pred_C - pack(v_target)) ** 2)

grad_accum = 1, TIME_BUDGET = 5400 s (1.5 hours).

Inference pose_mode handling:
    normal     → real pose latent
    no_pose    → zero latent
    wrong_pose → another sample's pose latent
    → pose deltas now respond to the eval's tests.

Final paste at inference: (1 - M) * agnostic + M * C.
"""

import json, logging, os, shutil, sys, time, traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers import QwenImageEditPlusPipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file


BASE = "/home/link/Desktop/Code/fashion gen testing"
LOCAL_CACHE = f"{BASE}/vtonautoresearch/local_cache"

# VGG perceptual loss (loaded lazily if USE_PERCEPTUAL=1)
_VGG_MODEL = None
def get_vgg_features(device, dtype):
    global _VGG_MODEL
    if _VGG_MODEL is None:
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        vgg.requires_grad_(False)
        # Use layers up to relu3_3 (index 16) for perceptual features
        _VGG_MODEL = torch.nn.Sequential(*list(vgg.children())[:16]).to(device, dtype=dtype)
    return _VGG_MODEL

def perceptual_loss(pred_img, target_img, weight_map, vgg_model):
    """VGG perceptual loss between pred and target, weighted by spatial map."""
    # Normalize from [-1,1] to ImageNet range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred_img.device, pred_img.dtype)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred_img.device, pred_img.dtype)
    pred_norm = ((pred_img + 1) / 2 - mean) / std
    targ_norm = ((target_img + 1) / 2 - mean) / std
    # Downsample to 256px for VGG (much faster, still captures structure)
    pred_ds = F.interpolate(pred_norm, size=(256, 192), mode="bilinear", align_corners=False)
    targ_ds = F.interpolate(targ_norm, size=(256, 192), mode="bilinear", align_corners=False)
    wm_ds = F.interpolate(weight_map, size=(256, 192), mode="bilinear", align_corners=False)
    with torch.no_grad():
        targ_feat = vgg_model(targ_ds)
    pred_feat = vgg_model(pred_ds)
    wm_feat = F.interpolate(wm_ds, size=pred_feat.shape[2:], mode="bilinear", align_corners=False)
    return ((pred_feat - targ_feat).pow(2) * wm_feat.expand_as(pred_feat)).mean()
# ─────────── PatchGAN discriminator (for adversarial loss on repair zone) ───────────
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN-ish. Input: (B, 3, H, W) image in [-1,1]. Output: (B, 1, h, w) logits."""
    def __init__(self):
        super().__init__()
        def _c(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm: layers.append(nn.InstanceNorm2d(out_c, affine=False))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        self.net = nn.Sequential(
            *_c(3, 64, norm=False),
            *_c(64, 128),
            *_c(128, 256),
            *_c(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )
    def forward(self, x):
        return self.net(x)

class V6Heads(nn.Module):
    """v6 specialized heads operating on Qwen's pre-proj 3072-dim per-token features
    (post norm_out, pre proj_out). Each is a tokenwise Linear:
      to_s     (3072 → 64)  packed skin residual δ_s
      to_b     (3072 → 64)  packed bg   residual δ_b
      to_route (3072 → 16)  4-class routing logits per 2×2 patch
    Zero-init residual heads so δ ≈ 0 initially (no early disruption to garment).
    Each head's gradient flows ONLY through its own params + shared transformer
    features, so cross-region bleed is bounded by feature sharing, not by
    direct mask-weight averaging on a single head."""
    def __init__(self, hidden_dim=3072, packed_dim=64, n_classes=4, patch=2):
        super().__init__()
        self.to_s = nn.Linear(hidden_dim, packed_dim, bias=True)
        self.to_b = nn.Linear(hidden_dim, packed_dim, bias=True)
        self.to_route = nn.Linear(hidden_dim, n_classes * patch * patch, bias=True)
        nn.init.zeros_(self.to_s.weight); nn.init.zeros_(self.to_s.bias)
        nn.init.zeros_(self.to_b.weight); nn.init.zeros_(self.to_b.bias)
        nn.init.normal_(self.to_route.weight, std=0.01); nn.init.zeros_(self.to_route.bias)
    def forward(self, hidden):
        return {
            "delta_s_packed": self.to_s(hidden),       # (B, N, 64)
            "delta_b_packed": self.to_b(hidden),       # (B, N, 64)
            "route_logits":   self.to_route(hidden),   # (B, N, 16)
        }


_V6_HEADS = None
_HIDDEN_HOLDER = {}
def _get_v6_heads(device, dtype, hidden_dim=3072):
    global _V6_HEADS
    if _V6_HEADS is None:
        _V6_HEADS = V6Heads(hidden_dim=hidden_dim, packed_dim=64).to(device, dtype=dtype)
        _V6_HEADS.requires_grad_(True)
    return _V6_HEADS

def _v6_hidden_hook(module, input, output):
    """Forward hook on transformer.norm_out: captures (B, N_total, 3072) features."""
    _HIDDEN_HOLDER["hidden"] = output

_DISC = None
_DISC_OPT = None
def _get_discriminator(device, dtype):
    global _DISC, _DISC_OPT
    if _DISC is None:
        _DISC = PatchDiscriminator().to(device, dtype=dtype)
        _DISC_OPT = torch.optim.Adam(_DISC.parameters(), lr=1e-4, betas=(0.5, 0.999))
    return _DISC, _DISC_OPT

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "3600"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",   "400"))
TRAIN_IDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
FIXED_PROMPT = "A realistic photo of the same person wearing the target garment."

# Slot-order config — parameterized via env var SLOT_ORDER
# e.g. SLOT_ORDER="garment,agnostic,pose,rough" → [C, gar, ag, pose, rough]
# SLOT_ORDER can be a SUBSET of ALL_SLOT_NAMES (exp414: drop rough entirely).
ALL_SLOT_NAMES = ["agnostic", "pose", "rough", "garment", "silhouette", "body_rough"]
SLOT_ORDER = os.environ.get("SLOT_ORDER", "agnostic,pose,rough,garment").split(",")
assert set(SLOT_ORDER).issubset(set(ALL_SLOT_NAMES)), f"SLOT_ORDER must be a subset of {ALL_SLOT_NAMES}, got {SLOT_ORDER}"
SLOT_INDEX = {n: i for i, n in enumerate(ALL_SLOT_NAMES)}
SLOT_ORDER_IDX = [SLOT_INDEX[n] for n in SLOT_ORDER]
NUM_SLOTS = len(SLOT_ORDER) + 1   # +1 for C position 0

ATTN_MASK_HOLDER = {"mask": None}
GARMENT_GATES = []   # filled by main() when USE_GARMENT_GATE=1

# ── Custom attention processor that reads attention_mask from ATTN_MASK_HOLDER ──
# Used when USE_REPAIR_ATTN_MASK=1 to block repair-band → out-of-mask attention.
from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0

class HoleyAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    """Same as parent but reads attention_mask from global holder when not provided."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        if attention_mask is None:
            attention_mask = ATTN_MASK_HOLDER.get("mask")
        return super().__call__(attn, hidden_states, encoder_hidden_states,
                                encoder_hidden_states_mask, attention_mask, image_rotary_emb)


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


# ── exp420: Garment-Repair Gated Dual-Branch Module ──
# Two asymmetric branches per block:
#   garment branch: cross-attention Q=C_tokens, K=V=garment_tokens (garment identity)
#   repair branch: the block's normal self-attention output (body/boundary cleanup)
# A per-token gate α ∈ [0,1] decides how much of each to use.
# u = α * u_gar + (1-α) * u_rep

class GarmentRepairGate(nn.Module):
    def __init__(self, dim, n_heads=4, head_dim=32):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Garment cross-attention (4 heads × 32 dim = 128 inner)
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        # Gate MLP: dim → 128 → 1
        self.gate = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # Init: gate starts low (α ≈ 0.12) so repair-dominant early in training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, h_c, h_gar, u_rep):
        """
        h_c:    (B, N_c, D)   — C chunk hidden states (current denoising state)
        h_gar:  (B, N_g, D)   — garment chunk hidden states
        u_rep:  (B, N_c, D)   — repair update (from block's normal self-attention)
        Returns: u_mixed (B, N_c, D), alpha (B, N_c, 1)
        Stores self.last_alpha for loss computation.
        """
        B, N_c, D = h_c.shape
        # Garment cross-attention
        q = self.q_proj(h_c).view(B, N_c, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_gar).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_gar).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_c, -1)
        u_gar = self.out_proj(attn_out)
        # Gate
        alpha = torch.sigmoid(self.gate(h_c))                                     # (B, N_c, 1)
        self.last_alpha = alpha
        # Mix
        u_mixed = alpha * u_gar + (1 - alpha) * u_rep
        return u_mixed, alpha


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


@dataclass
class Args:
    pretrained_model: str = f"{BASE}/Qwen-Image-Edit-2511"
    vitonhd_dir: str = f"{BASE}/VITON-HD-dataset"
    latent_cache_dir: str = f"{BASE}/my_vton_cache/latents"
    text_cache_dir: str = f"{BASE}/my_vton_cache/text"
    output_dir: str = f"{BASE}/vtonautoresearch/runs"
    rank: int = int(os.environ.get("LORA_RANK", "32"))
    alpha: int = int(os.environ.get("LORA_ALPHA", "64"))
    init_lora_weights: str = "gaussian"
    lora_targets: list = field(default_factory=lambda: os.environ.get(
        "LORA_TARGETS", "to_k,to_q,to_v,to_out.0").split(","))
    inject_blocks: list = field(default_factory=lambda: list(range(60)))
    sigma_beta_alpha: float = 1.0
    sigma_beta_beta: float = 1.0
    lr: float = 3e-5
    projector_lr: float = 2e-3
    batch_size: int = 1
    grad_accum: int = 1   # every microstep = optimizer step; more updates at same walltime
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    gradient_checkpointing: bool = True
    device_transformer: str = "cuda:0"
    logging_steps: int = 10
    seed: int = 42
    train_split: str = "test"


# ─────────────────────────── Garment cross-attention ───────────────────────────

# exp401 has no custom modules — stock Qwen attention + tryon LoRA only.


# ─────────────────────────── Pack / unpack ───────────────────────────

def pack_latents(lat, B, C, H, W):
    return lat.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

def unpack_latents(lat, B, C, H, W):
    return lat.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


# ─────────────────────────── Precompute at startup ───────────────────────────

def vae_decode_to_pil(vae, latent_norm, vae_device, dtype):
    """(16, H, W) normalized latent → PIL RGB image."""
    m = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    s = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    x = latent_norm.unsqueeze(0).unsqueeze(2).to(vae_device, dtype=dtype)  # (1,16,1,H,W)
    x = x * s + m
    with torch.no_grad():
        img = vae.decode(x, return_dict=False)[0][:, :, 0]                  # (1,3,H_img,W_img)
    img = img.clamp(-1, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(img)


def precompute_rough_pils(latent_cache_dir, vae, vae_device, dtype):
    cache = {}
    for sid in TRAIN_IDS:
        rough_file = os.environ.get("ROUGH_FILE", "_degraded_rough_latent.pt")
        lat = torch.load(os.path.join(latent_cache_dir, f"{sid}{rough_file}"), weights_only=True)
        pil = vae_decode_to_pil(vae, lat, vae_device, dtype)
        cache[sid] = pil
    return cache


def precompute_prompt_embeds(pipe, latent_cache_dir, local_cache, rough_pils, device, dtype):
    """Per-sample Qwen2.5-VL encode of [agnostic, pose, rough, garment] + fixed prompt.
    exp396: 4 source images fed to the text encoder so it has semantic access to all
    four signals, including the garment (which is no longer in the transformer's
    spatial image stream).
    """
    cache = {}
    for sid in TRAIN_IDS:
        agn_px = torch.load(os.path.join(latent_cache_dir, f"{sid}_agnostic_pixel.pt"), weights_only=True)
        agn_pil = Image.fromarray((agn_px.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
        pose_pil = Image.open(os.path.join(local_cache, f"pose_pixel_{sid}.png")).convert("RGB")
        rough_pil = rough_pils[sid]
        garment_px = torch.load(os.path.join(latent_cache_dir, f"{sid}_garment_pixel.pt"), weights_only=True)
        garment_pil = Image.fromarray((garment_px.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

        # exp486+: drop rough_pil from VL (was double-injecting artifacts via spatial
        # and semantic paths; rough is now only in the spatial slot). Keep pose_pil
        # and garment_pil and agn_pil. Pose goes ONLY via VL (the spatial pose slot
        # is off-manifold densepose and is dropped via SLOT_ORDER), so removing pose
        # from VL would eliminate pose info entirely.
        with torch.no_grad():
            if int(os.environ.get("ROUGH_AS_CONTROL", "0")):
                vl_images = [agn_pil, pose_pil, rough_pil, garment_pil]
            else:
                vl_images = [agn_pil, pose_pil, garment_pil]
            pe, pm = pipe.encode_prompt(
                image=vl_images,
                prompt=FIXED_PROMPT,
                device=device,
                num_images_per_prompt=1,
            )
        cache[sid] = (pe.detach().clone().to(device, dtype=dtype),
                      pm.detach().clone().to(device, dtype=torch.long))
    return cache


def load_pose_latents(local_cache, device, dtype):
    cache = {}
    for sid in TRAIN_IDS:
        lat = torch.load(os.path.join(local_cache, f"pose_latent_{sid}.pt"), weights_only=True)
        cache[sid] = lat.to(device, dtype=dtype)                              # (16, 128, 96)
    return cache


# ─────────────────────────── Dataset ───────────────────────────

class VTONDataset(Dataset):
    def __init__(self, args, split="train"):
        self.latent_dir = args.latent_cache_dir
        # Filter by TRAIN_IDS + latent cache availability. Ignore split — caches
        # for all IDs live in the same cache dir regardless of which split the
        # original VITON-HD image came from.
        required = ["_person_latent.pt", "_garment_latent.pt", "_degraded_rough_latent.pt",
                    "_agnostic_latent.pt", "_agnostic_mask_latent.pt", "_target_mask.pt",
                    "_warped_mask_128.pt"]
        self.image_ids = [i for i in TRAIN_IDS if all(
            os.path.exists(os.path.join(self.latent_dir, f"{i}{s}")) for s in required)]
        print(f"VTONDataset [{split}]: {len(self.image_ids)} samples")

    def __len__(self): return len(self.image_ids)

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        L = lambda s: torch.load(os.path.join(self.latent_dir, f"{i}{s}"), weights_only=True)
        item = {
            "image_id":      i,
            "person_latent": L("_person_latent.pt"),
            "rough_latent":  L(os.environ.get("ROUGH_FILE", "_degraded_rough_latent.pt")),
            "garment_latent":L("_garment_latent.pt"),
            "agnostic_latent":       L(os.environ.get("AGNOSTIC_FILE", "_agnostic_latent.pt")),
            "warped_mask":           L("_warped_mask_128.pt"),
            "agnostic_mask_latent":  L(os.environ.get("AGNOSTIC_MASK_FILE", "_agnostic_mask_latent.pt")),
            "target_mask":           L("_target_mask.pt"),
        }
        if int(os.environ.get("USE_DP_SPLIT", "0")) or int(os.environ.get("USE_BG_HINT", "0")):
            item["densepose"] = L("_densepose.pt")                               # (3, 1024, 768)
        if int(os.environ.get("USE_V6", "0")):
            item["parse_garment"] = L("_parse_garment_latent.pt")                # (1, 128, 96)
            item["parse_skin"]    = L("_parse_skin_latent.pt")                   # (1, 128, 96)
            item["parse_bg"]      = L("_parse_bg_latent.pt")                     # (1, 128, 96)
        return item


def collate_fn(batch):
    keys = ["person_latent", "rough_latent", "garment_latent",
            "agnostic_latent", "agnostic_mask_latent", "target_mask",
            "warped_mask"]
    out = {
        **{k: torch.stack([b[k] for b in batch]) for k in keys},
        "image_id": [b["image_id"] for b in batch],
    }
    if "densepose" in batch[0]:
        out["densepose"] = torch.stack([b["densepose"] for b in batch])
    if "parse_garment" in batch[0]:
        out["parse_garment"] = torch.stack([b["parse_garment"] for b in batch])
        out["parse_skin"]    = torch.stack([b["parse_skin"]    for b in batch])
        out["parse_bg"]      = torch.stack([b["parse_bg"]      for b in batch])
    return out


# ─────────────────────────── Train step ───────────────────────────

def train_step(transformer,
               pose_cache, prompt_cache,
               vae, person_image_cache, vae_device, img_loss_weight,
               loss_weights,
               batch, device, weight_dtype,
               sigma_beta_alpha=1.0, sigma_beta_beta=1.0,
               global_step=0, max_steps=400):
    person   = batch["person_latent"].to(device, dtype=weight_dtype)           # (B, 16, 128, 96)
    agnostic = batch["agnostic_latent"].to(device, dtype=weight_dtype)
    # Fill repair zone of agnostic with rough's body estimate. Rough shows the try-on
    # preview with approximate body/arm pixels in the repair zone. By filling agnostic's
    # repair zone with rough, the model sees a MORE COHERENT agnostic and can use the
    # body-ish content as a reference for "what goes here". Garment zone stays grey
    # (model must generate garment). (USE_AGNOSTIC_ROUGH_FILL=1)
    if int(os.environ.get("USE_AGNOSTIC_ROUGH_FILL", "0")):
        _rl_fill = batch["rough_latent"].to(device, dtype=weight_dtype)
        _wm_fill = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_fill.dim() == 3: _wm_fill = _wm_fill.unsqueeze(1)
        _M_ag_fill = (batch["agnostic_mask_latent"].to(device, dtype=weight_dtype) > 0.5).to(weight_dtype)
        _gar_bin = (_wm_fill > 0.5).to(weight_dtype)
        _repair_zone = (_M_ag_fill - _gar_bin).clamp(0, 1)                           # edit minus garment
        agnostic = agnostic * (1 - _repair_zone) + _rl_fill * _repair_zone

    # Iterated Gaussian fill of agnostic: propagate unmasked pixel values into the
    # masked region via repeated blur-then-paste. Gives model a coherent agnostic
    # with body/background extending naturally through the mask. (USE_AGNOSTIC_INPAINT=1)
    if int(os.environ.get("USE_AGNOSTIC_INPAINT", "0")):
        from torchvision.transforms.functional import gaussian_blur as _gb
        _M_ai_hard = (batch["agnostic_mask_latent"].to(device, dtype=weight_dtype) > 0.5).to(weight_dtype)
        _M_ai = _M_ai_hard
        # AGNOSTIC_INPAINT_SOFT=1: feather the paste mask so the detail-level
        # transition at the mask boundary is a smooth ramp, not a hard step.
        # Prevents a visible boundary line in the model output.
        _soft_paste_sig = float(os.environ.get("AGNOSTIC_INPAINT_SOFT_SIG", "0"))
        if _soft_paste_sig > 0:
            _k_sp = int(2 * round(2 * _soft_paste_sig) + 1)
            _M_ai = _gb(_M_ai_hard.float(), kernel_size=[_k_sp, _k_sp],
                         sigma=_soft_paste_sig).to(_M_ai_hard.dtype).clamp(0, 1)
        _keep = (1 - _M_ai)
        _result = agnostic.clone()
        _k = int(os.environ.get("AGNOSTIC_INPAINT_KERNEL", "7"))
        _sig = float(os.environ.get("AGNOSTIC_INPAINT_SIGMA", "2.0"))
        _iters = int(os.environ.get("AGNOSTIC_INPAINT_ITERS", "20"))
        for _ in range(_iters):
            _blurred = _gb(_result.float(), kernel_size=[_k, _k], sigma=_sig).to(_result.dtype)
            _result = _result * _keep + _blurred * _M_ai
        agnostic = _result

    # AGNOSTIC_ZERO_REPAIR=1: zero the agnostic in the repair band (M_full minus
    # warped garment silhouette) so the model CANNOT identity-map the grey
    # agnostic into its repair-band output. Forces the model to synthesize body
    # content using other signals (rough, pose via VL, surrounding person).
    # Uses warped_mask (inference-available), NOT target_mask, so same at inf.
    if int(os.environ.get("AGNOSTIC_ZERO_REPAIR", "0")):
        _M_ag_bin = (batch["agnostic_mask_latent"].to(device, dtype=weight_dtype) > 0.5).to(weight_dtype)
        _wm = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
        _wm_bin = (_wm > 0.5).to(weight_dtype)
        _repair_proxy = (_M_ag_bin - _wm_bin).clamp(0, 1)                         # (B, 1, H, W)
        agnostic = agnostic * (1 - _repair_proxy)                                  # zero in repair band
    rough    = batch["rough_latent"].to(device, dtype=weight_dtype)
    garment  = batch["garment_latent"].to(device, dtype=weight_dtype)
    M_ag     = batch["agnostic_mask_latent"].to(device, dtype=weight_dtype)
    M_full   = (M_ag > 0.5).to(dtype=weight_dtype)
    image_ids = batch["image_id"]
    B, C, H, W = person.shape

    # ── Mask geometry (exp486+: tight band from target_mask, not fuzzy warped_mask) ──
    # garment_prior was previously batch["warped_mask"] (soft, with wide fuzzy edges
    # from the warp process itself — not the true contour uncertainty). That diluted
    # supervision near the real edge. Now use binary target_mask for a crisp silhouette
    # and a 3-pixel dilate-erode ring for the uncertain_band.
    tm = batch["target_mask"].to(device, dtype=weight_dtype)                   # (B, 1, 128, 96), binary-ish
    if tm.dim() == 3:
        tm = tm.unsqueeze(1)
    garment_prior = (tm > 0.5).to(weight_dtype)                                 # crisp silhouette
    tm_dil = F.max_pool2d(garment_prior, kernel_size=5, stride=1, padding=2)
    tm_ero = -F.max_pool2d(-garment_prior, kernel_size=5, stride=1, padding=2)
    uncertain_band = (tm_dil - tm_ero).clamp(0, 1).to(weight_dtype)             # thin ring around true contour
    # repair_band: edit region minus target garment = where body/arm/neck must be
    # reconstructed. Previously got ≈0 weight → disease concentrated here. Now
    # explicitly supervised.
    repair_band = (M_full - garment_prior).clamp(0, 1).to(weight_dtype)
    # keep_mask: high outside the edit region
    keep_mask = 1.0 - M_full                                                   # (B, 1, 128, 96)

    # ── 3-region repair split via densepose (USE_DP_SPLIT=1) ──
    # repair_band = body_repair (skin/neck — must generate real body)
    #             ∪ bg_repair   (empty wings behind old garment — pure background)
    # The model collapses bg_repair to BG easily, but leaves a sharp residue ring
    # at its boundary. Splitting lets us weight bg_repair strongly (pull to GT
    # BG pixel-for-pixel) while keeping body_repair at moderate weight.
    # Compute body_latent once if densepose is available (USE_DP_SPLIT / USE_BG_HINT / USE_FLOW_REGION_SPLIT)
    body_repair = torch.zeros_like(repair_band)
    bg_repair   = torch.zeros_like(repair_band)
    body_latent = torch.zeros_like(repair_band)   # for img-resolution use below
    if "densepose" in batch:
        dp_raw = batch["densepose"].to(device, dtype=weight_dtype)            # (B, 3, 1024, 768)
        body_img = (dp_raw.sum(dim=1, keepdim=True) > 0.02).to(weight_dtype)  # (B, 1, 1024, 768)
        body_latent = F.interpolate(body_img, size=(H, W), mode="area")       # (B, 1, 128, 96) soft
        body_latent = (body_latent > 0.5).to(weight_dtype)                     # re-binarize
        body_repair = (repair_band * body_latent).clamp(0, 1)
        bg_repair   = (repair_band * (1.0 - body_latent)).clamp(0, 1)

    # ── v6 routing classes from target parse + warped_mask ──
    # M_edit = dilate(warped, r_out)   (edit support, inference-available)
    # M_core = erode(warped,  r_in)    (confident garment synthesis core)
    # M_g = M_edit ∩ parse_garment     (new garment)
    # M_s = M_edit ∩ parse_skin        (exposed skin: face/arm/neck)
    # M_b = M_edit − M_g − M_s         (revealed bg / other)
    # M_k = 1 − M_edit                 (keep untouched)
    use_v6 = int(os.environ.get("USE_V6", "0")) and "parse_garment" in batch
    M_g_v6 = torch.zeros_like(repair_band)
    M_s_v6 = torch.zeros_like(repair_band)
    M_b_v6 = torch.zeros_like(repair_band)
    M_k_v6 = torch.ones_like(repair_band)
    M_edit_v6 = torch.zeros_like(repair_band)
    M_core_v6 = torch.zeros_like(repair_band)
    if use_v6:
        r_out = int(os.environ.get("V6_R_OUT", "7"))     # latent px
        r_in  = int(os.environ.get("V6_R_IN",  "2"))
        _wm_v6 = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_v6.dim() == 3: _wm_v6 = _wm_v6.unsqueeze(1)
        _wm_bin_v6 = (_wm_v6 > 0.5).to(weight_dtype)
        M_edit_v6 = F.max_pool2d(_wm_bin_v6, 2*r_out+1, 1, r_out).clamp(0, 1)
        _M_core_t = -F.max_pool2d(-_wm_bin_v6, 2*r_in+1, 1, r_in)
        M_core_v6 = (_M_core_t > 0.5).to(weight_dtype)

        _ps = batch["parse_skin"].to(device, dtype=weight_dtype)
        _pb = batch["parse_bg"].to(device, dtype=weight_dtype)
        if _ps.dim() == 3: _ps = _ps.unsqueeze(1)
        if _pb.dim() == 3: _pb = _pb.unsqueeze(1)
        _ps_b = (_ps > 0.5).to(weight_dtype)
        _pb_b = (_pb > 0.5).to(weight_dtype)

        # SAFE routing: warped defines garment class (inference-available).
        # Parse only subdivides the ring (M_edit − warped) into skin/bg. If
        # parse is missing or disagrees, ring pixels fall back to M_other
        # which is treated as keep-like (preserve source), avoiding the
        # oracle-parse geometry dependency inside the garment core.
        ring_v6 = (M_edit_v6 - _wm_bin_v6).clamp(0, 1)
        M_g_v6 = _wm_bin_v6                                                    # warped = garment
        M_s_v6 = (ring_v6 * _ps_b).clamp(0, 1)                                  # skin only in ring
        M_b_v6 = (ring_v6 * _pb_b * (1.0 - M_s_v6)).clamp(0, 1)                 # bg only in ring
        M_other_v6 = (ring_v6 - M_s_v6 - M_b_v6).clamp(0, 1)                    # hair/pants/unknown
        M_k_v6 = (1.0 - M_edit_v6).clamp(0, 1)

        # ── Kill garment-core torso template: zero agnostic inside M_core ──
        # Uses eroded warped_mask (inference-available) so train and inference
        # do the same input surgery. Removes torso template — the model must
        # actually synthesize garment, not denoise a body blob.
        if int(os.environ.get("V6_ZERO_G_CORE", "1")):
            agnostic = agnostic * (1.0 - M_core_v6)                            # zero confident garment core

    # NO neutralization — feed raw agnostic and rough as conditioning.
    # The model learns routing from the soft masks via the loss structure.

    # CFG conditioning dropout: randomly zero garment slot to enable classifier-free guidance
    cfg_dropout = float(os.environ.get("CFG_DROPOUT", "0.0"))
    if cfg_dropout > 0 and torch.rand(1).item() < cfg_dropout:
        garment = torch.zeros_like(garment)

    # Per-sample cached tensors
    pose = torch.stack([pose_cache[iid] for iid in image_ids]).to(device, dtype=weight_dtype)

    # Per-sample prompt embeds (pad to max text seq across the batch)
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
    pe_batch = torch.cat(pe_pad, dim=0).to(device, dtype=weight_dtype)
    pm_batch = torch.cat(pm_pad, dim=0).to(device, dtype=torch.long)
    txt_seq_lens = pm_batch.sum(dim=1).tolist()

    # ── Flow matching — rough-based SDEdit forward (exp421) ──
    # C_t interpolates between rough and noise (NOT person and noise).
    # v_target still points toward person so the model learns rough→person corrections.
    # At inference, C_lat starts from noised rough and the model refines.
    sigma = torch.distributions.Beta(sigma_beta_alpha, sigma_beta_beta).sample((B,)).to(device=device, dtype=weight_dtype)
    sigma_cap = float(os.environ.get("SIGMA_CAP", "1.0"))
    if sigma_cap < 1.0:
        sigma = sigma * sigma_cap
    s = sigma.view(B, 1, 1, 1)
    noise = torch.randn_like(person)

    # Progressive noise curriculum (NOISE_CURRICULUM env var):
    # Blend between SDEdit-style (start from rough) and noise-init over training.
    # noise_blend=0 → pure SDEdit (C_t interpolates rough→noise)
    # noise_blend=1 → pure noise-init (C_t interpolates person→noise)
    noise_curriculum = int(os.environ.get("NOISE_CURRICULUM", "0"))
    if noise_curriculum:
        # Linear ramp from 0 to 1 over training
        warmup_frac = float(os.environ.get("CURRICULUM_WARMUP", "0.5"))
        progress = min(1.0, global_step / (max_steps * warmup_frac))
        # Interpolate starting point between rough and person
        init_point = progress * person + (1 - progress) * rough
        C_t      = init_point + s * M_full * (noise - init_point)
        v_target = M_full * (noise - person)
    elif int(os.environ.get("USE_PURE_NOISE", "0")):
        # Pure-noise slot 0 — matches Qwen native pipeline (no clean-pixel hybrid).
        # σ=1 → full Gaussian noise; σ=0 → person. Supervision still masked to edit region.
        C_t      = person + s * (noise - person)                              # no mask on init
        v_target = M_full * (noise - person)                                   # masked target
    else:
        # Standard flow matching: slot 0 = noised person (NOT rough).
        C_t      = person + s * M_full * (noise - person)
        v_target = M_full * (noise - person)

    # ── Rough augmentation (USE_ROUGH_AUG=1) ──
    # Teach model to extract only low-freq (shape/color) from rough, not high-freq artifacts.
    # Per-step random blur σ in [0, ROUGH_BLUR_MAX] + per-step random noise scale.
    # Model sees rough at many quality levels → must rely on signal that survives all levels
    # = low-freq silhouette/color. High-freq patterns in rough become inconsistent across
    # training steps, so model cannot memorize them.
    if int(os.environ.get("USE_ROUGH_AUG", "0")):
        from torchvision.transforms.functional import gaussian_blur
        blur_max = float(os.environ.get("ROUGH_BLUR_MAX", "3.0"))
        noise_max = float(os.environ.get("ROUGH_NOISE_MAX", "0.2"))
        # Random per-step (not per-sample) for simplicity
        r_blur_sig = torch.rand(1).item() * blur_max
        r_noise_scl = torch.rand(1).item() * noise_max
        if r_blur_sig > 0.1:
            bk = int(2 * round(2 * r_blur_sig) + 1)
            rough = gaussian_blur(rough.float(), kernel_size=[bk, bk], sigma=r_blur_sig).to(rough.dtype)
        if r_noise_scl > 0.01:
            rough_std = rough.float().std(dim=(-1, -2), keepdim=True).clamp(min=1e-3)
            rough = (rough.float() + torch.randn_like(rough.float()) * r_noise_scl * rough_std).to(rough.dtype)

    # ── Rough fixed blur (USE_ROUGH_BLUR_FIXED=1) ──
    # Always blur rough at fixed σ — both train and inference will use this σ.
    # Removes ALL high-freq from rough slot. Model forced to only use silhouette/color.
    if int(os.environ.get("USE_ROUGH_BLUR_FIXED", "0")):
        from torchvision.transforms.functional import gaussian_blur
        fixed_sig = float(os.environ.get("ROUGH_BLUR_FIXED_SIG", "4.0"))
        bk = int(2 * round(2 * fixed_sig) + 1)
        rough = gaussian_blur(rough.float(), kernel_size=[bk, bk], sigma=fixed_sig).to(rough.dtype)

    # ── Garment low-frequency bias at high sigma (USE_GAR_BLUR=1) ──
    # At high sigma, model sees blurred garment (low-freq shape/color only);
    # at low sigma, clean garment (full detail). Forces silhouette-first
    # commitment without removing supervision. Train-only; inference unchanged.
    if int(os.environ.get("USE_GAR_BLUR", "0")):
        from torchvision.transforms.functional import gaussian_blur
        blur_sigma_val = float(os.environ.get("GAR_BLUR_SIGMA", "2.0"))
        blur_k = int(2 * round(2 * blur_sigma_val) + 1)
        garment_blurred = gaussian_blur(
            garment.float(), kernel_size=[blur_k, blur_k], sigma=blur_sigma_val
        ).to(garment.dtype)
        s_mix = sigma.view(B, 1, 1, 1).to(garment.dtype)
        garment = s_mix * garment_blurred + (1 - s_mix) * garment

    # ── Optionally mask rough by silhouette (USE_ROUGH_MASKED=1) ──
    # Rough contributes info only inside the warped_mask region. Outside, rough is
    # zero → can't leak artifacts into repair zone regardless of attention patterns.
    if int(os.environ.get("USE_ROUGH_MASKED", "0")):
        _wm_r = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_r.dim() == 3: _wm_r = _wm_r.unsqueeze(1)
        if int(os.environ.get("ROUGH_MASK_SOFT", "0")):
            # Soft mask: Gaussian-smoothed binary. Prevents hard edge in rough input.
            from torchvision.transforms.functional import gaussian_blur as _gb_rm
            _wm_r_bin = (_wm_r > 0.5).to(weight_dtype)
            _rm_sig = float(os.environ.get("ROUGH_MASK_SOFT_SIG", "3.0"))
            _k_rm = int(2 * round(2 * _rm_sig) + 1)
            _wm_r_b = _gb_rm(_wm_r_bin.float(), kernel_size=[_k_rm, _k_rm], sigma=_rm_sig).to(_wm_r_bin.dtype)
        else:
            _wm_r_b = (_wm_r > 0.5).to(weight_dtype)
        rough = rough * _wm_r_b                                                    # (B, 16, H, W)

    # ── Pack 5 positions, slot order driven by SLOT_ORDER env var ──
    C_p_    = pack_latents(C_t,      B, C, H, W)   # (B, 3072, 64)
    agn_p   = pack_latents(agnostic, B, C, H, W)
    pose_p  = pack_latents(pose,     B, C, H, W)
    rough_p = pack_latents(rough,    B, C, H, W)
    gar_p   = pack_latents(garment,  B, C, H, W)
    vt_p    = pack_latents(v_target, B, C, H, W)

    # ── Sigma-scheduled conditioning scales (USE_SIGMA_SCHED=1) ──
    # Gentle bias: all slots visible at all sigma (min 0.8, max 1.2).
    # Early (high sigma): structure slightly amplified (for contour)
    # Late  (low sigma):  detail slightly amplified (for identity)
    # Garment+rough stay visible at high sigma so early steps can choose the contour.
    if int(os.environ.get("USE_SIGMA_SCHED", "0")):
        s_vec = sigma.view(B, 1, 1).float()                                     # (B, 1, 1)
        sched_lo = float(os.environ.get("SIGMA_SCHED_LO", "0.8"))               # scale at weak end
        sched_hi = float(os.environ.get("SIGMA_SCHED_HI", "1.2"))               # scale at strong end
        span = sched_hi - sched_lo
        struct_scale = (sched_lo + span * s_vec).to(agn_p.dtype)                # lo→hi as sigma 0→1
        detail_scale = (sched_hi - span * s_vec).to(gar_p.dtype)                # hi→lo as sigma 0→1
        agn_p   = agn_p   * struct_scale
        pose_p  = pose_p  * struct_scale
        rough_p = rough_p * detail_scale
        gar_p   = gar_p   * detail_scale

    # ── Garment-slot noise at high sigma (USE_GAR_NOISE=1) ──
    # Gentler than amplitude schedule: adds Gaussian perturbation scaled by sigma
    # so at high sigma the model is forced to rely on rough/agnostic/pose for
    # silhouette (not starved — signal still present, just noisier). At low sigma
    # (final steps) garment is clean for identity. Train-only; no inference mirror.
    if int(os.environ.get("USE_GAR_NOISE", "0")):
        noise_max = float(os.environ.get("GAR_NOISE_MAX", "0.3"))
        s_vec_n = sigma.view(B, 1, 1).float() * noise_max                       # (B,1,1) in [0, noise_max]
        gar_std = gar_p.float().std(dim=(-1, -2), keepdim=True).clamp(min=1e-3)
        gar_noise = torch.randn_like(gar_p.float()) * s_vec_n * gar_std
        gar_p = (gar_p.float() + gar_noise).to(gar_p.dtype)

    # Silhouette slot: use warped_mask_128 (from garment warping pre-process, NOT GT).
    # Warped_mask is available at inference for any new sample. target_mask is GT
    # and would be leakage. Broadcast 1→16 channels and pack.
    if "silhouette" in SLOT_ORDER:
        if int(os.environ.get("USE_VAE_SILHOUETTE", "0")):
            # Use a VAE-encoded silhouette LATENT (pre-computed from a grey-on-white
            # silhouette image). In-distribution latent, not a bitmap broadcast.
            _sil_lats = []
            for iid in image_ids:
                _lp = os.path.join(BASE, "my_vton_cache/latents", f"{iid}_warped_silhouette_latent.pt")
                _sil_lats.append(torch.load(_lp, weights_only=True))
            sil = torch.stack(_sil_lats).to(device, dtype=weight_dtype)                   # (B, 16, 128, 96)
            sil_p = pack_latents(sil, B, C, H, W)
        else:
            _wm = batch["warped_mask"].to(device, dtype=weight_dtype)                    # (B, 1, 128, 96)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            if int(os.environ.get("SILHOUETTE_SOFT", "0")):
                from torchvision.transforms.functional import gaussian_blur as _gb2
                _wm_bin = (_wm > 0.5).to(weight_dtype)
                _soft_sig = float(os.environ.get("SILHOUETTE_SOFT_SIG", "2.0"))
                _k = int(2 * round(2 * _soft_sig) + 1)
                _wm_b = _gb2(_wm_bin.float(), kernel_size=[_k, _k], sigma=_soft_sig).to(_wm_bin.dtype)
            else:
                _wm_b = (_wm > 0.5).to(weight_dtype)
            _sil_scale = float(os.environ.get("SILHOUETTE_SCALE", "1.0"))
            # ── USE_BG_HINT=1: make silhouette a 2-signal map ──
            # ch 0  = +1 where garment silhouette (warped_mask) → "generate garment here"
            # ch 1  = +1 where bg wing (agnostic ∩ ¬densepose_body ∩ ¬warped) → "this is BG"
            # ch 2..15 = 0
            # Model is told explicitly which pixels are BG, so it doesn't need to infer
            # from color alone which region of the wide agnostic should collapse to BG.
            if int(os.environ.get("USE_BG_HINT", "0")) and "densepose" in batch:
                sil = torch.zeros((B, C, H, W), device=device, dtype=weight_dtype)
                sil[:, 0:1] = _wm_b * _sil_scale                                         # garment+
                wm_bin_lat = (_wm > 0.5).to(weight_dtype)
                # bg_hint = agnostic ∩ ¬body ∩ ¬warped (reuse body_latent computed above)
                bg_hint = (M_full * (1.0 - body_latent) * (1.0 - wm_bin_lat)).clamp(0, 1)
                _bg_scale = float(os.environ.get("BG_HINT_SCALE", "1.0"))
                sil[:, 1:2] = bg_hint * _bg_scale
            else:
                sil = (_wm_b * _sil_scale).expand(B, C, H, W).to(dtype=weight_dtype)
            sil_p = pack_latents(sil, B, C, H, W)
    else:
        sil_p = None
    # Body-rough slot: rough * (1 - warped_mask) = rough's body/background content only.
    # Separates garment rough (in main rough slot) from body context for repair zone.
    if "body_rough" in SLOT_ORDER:
        _wm_br = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_br.dim() == 3: _wm_br = _wm_br.unsqueeze(1)
        _wm_br_b = (_wm_br > 0.5).to(weight_dtype)
        _br = batch["rough_latent"].to(device, dtype=weight_dtype) * (1 - _wm_br_b)
        br_p = pack_latents(_br, B, C, H, W)
    else:
        br_p = None
    slot_tensors = {"agnostic": agn_p, "pose": pose_p, "rough": rough_p, "garment": gar_p,
                     "silhouette": sil_p, "body_rough": br_p}
    cond_seq = [slot_tensors[n] for n in SLOT_ORDER]
    hidden = torch.cat([C_p_] + cond_seq, dim=1)                                # (B, 3072 * NUM_SLOTS, 64)
    img_shapes = [[(1, H//2, W//2)] * NUM_SLOTS] * B

    ATTN_MASK_HOLDER["mask"] = None

    # USE_REPAIR_ATTN_MASK=1: block C-slot repair-band queries from attending to
    # agnostic-slot out-of-mask keys (background-leak mitigation).
    if int(os.environ.get("USE_REPAIR_ATTN_MASK", "0")):
        # Find agnostic slot's index in the spatial-slot sequence (1 = first slot after C)
        if "agnostic" in SLOT_ORDER:
            agn_seq_idx = SLOT_ORDER.index("agnostic") + 1   # +1 because C is at slot 0
        else:
            agn_seq_idx = 1
        wm_for_mask = batch["warped_mask"].to(device, dtype=weight_dtype)
        if wm_for_mask.dim() == 3: wm_for_mask = wm_for_mask.unsqueeze(1)
        ATTN_MASK_HOLDER["mask"] = build_repair_attn_mask(
            M_full, wm_for_mask, pe_batch.shape[1], NUM_SLOTS,
            B, device, weight_dtype, agnostic_slot_idx_in_seq=agn_seq_idx,
        )

    # ── Main forward ──
    out = transformer(
        hidden_states              = hidden,
        timestep                   = sigma,
        encoder_hidden_states      = pe_batch,
        encoder_hidden_states_mask = pm_batch,
        img_shapes                 = img_shapes,
        txt_seq_lens               = txt_seq_lens,
        return_dict                = False,
    )[0]

    pred_C = out[:, :C_p_.size(1), :]                                             # (B, 3072, 64)

    # ── Soft-weighted flow loss ──
    # USE_SIGMA_SPATIAL_SCHED=1: smooth spatial crossover across sigma.
    # High sigma → boundary pressure high, interior pressure low (tolerate wrong interior details).
    # Low  sigma → interior pressure high, boundary pressure lower (detail/identity dominant).
    # Total weight mass per sigma stays ~const (unlike exp450 SNR dim which starved gradient).
    w_keep, w_garment, w_uncertain = loss_weights
    if int(os.environ.get("USE_SIGMA_SPATIAL_SCHED", "0")):
        s_map      = sigma.view(B, 1, 1, 1).float()                                  # (B,1,1,1), 1=high noise
        int_region = (garment_prior - uncertain_band).clamp(min=0.0)                 # garment interior
        bdry_region= uncertain_band                                                   # transition band
        w_keep_s   = float(os.environ.get("W_KEEP",     "0.05"))
        w_bdry_lo  = float(os.environ.get("W_BDRY_LO",  "0.3"))
        w_bdry_hi  = float(os.environ.get("W_BDRY_HI",  "2.0"))
        w_int_lo   = float(os.environ.get("W_INT_LO",   "0.2"))
        w_int_hi   = float(os.environ.get("W_INT_HI",   "2.0"))
        w_bdry = w_bdry_lo + (w_bdry_hi - w_bdry_lo) * s_map                         # ↑ with σ
        w_int  = w_int_lo  + (w_int_hi  - w_int_lo)  * (1.0 - s_map)                 # ↑ toward clean
        weight_map = w_keep_s * keep_mask + w_bdry * bdry_region + w_int * int_region
    else:
        if int(os.environ.get("USE_DP_SPLIT", "0")) and "densepose" in batch:
            w_body_rep = float(os.environ.get("W_BODY_REPAIR", "0.3"))
            w_bg_rep   = float(os.environ.get("W_BG_REPAIR",   "2.0"))
            weight_map = (w_keep * keep_mask
                          + w_garment * garment_prior
                          + w_uncertain * uncertain_band
                          + w_body_rep * body_repair
                          + w_bg_rep   * bg_repair)
        else:
            w_repair = float(os.environ.get("W_REPAIR", "0.3"))
            weight_map = (w_keep * keep_mask
                          + w_garment * garment_prior
                          + w_uncertain * uncertain_band
                          + w_repair * repair_band)
        # Optionally smooth the weight_map boundaries to prevent model from
        # memorizing hard zone transitions as visible output edges.
        if int(os.environ.get("WEIGHT_MAP_SOFT", "0")):
            from torchvision.transforms.functional import gaussian_blur as _gb_wm
            _wm_sig = float(os.environ.get("WEIGHT_MAP_SOFT_SIG", "2.0"))
            _k_wm = int(2 * round(2 * _wm_sig) + 1)
            weight_map = _gb_wm(weight_map.float(), kernel_size=[_k_wm, _k_wm],
                                 sigma=_wm_sig).to(weight_map.dtype)
    Wmap_p = pack_latents(weight_map.expand(B, C, H, W), B, C, H, W).mean(dim=-1, keepdim=True)
    sq_err = ((pred_C.float() - vt_p.float()) ** 2).mean(dim=-1, keepdim=True)

    # ── Region-split flow loss (USE_FLOW_REGION_SPLIT=1) ──
    # Per-region normalized MSE (mean over region), combined with explicit weights.
    # Unlike the single weighted MSE (which mixes region sizes), each term has its
    # own gradient path and the weights are directly comparable.
    # Requires densepose (USE_DP_SPLIT=1 or USE_BG_HINT=1) for body_latent.
    if int(os.environ.get("USE_FLOW_REGION_SPLIT", "0")) or use_v6:
        def _pack_mask(m):
            return pack_latents(m.expand(B, C, H, W), B, C, H, W).mean(dim=-1, keepdim=True)
        def _reg_mse(mask_p):
            denom = mask_p.sum() + 1e-6
            return (sq_err * mask_p).sum() / denom

        if use_v6:
            # When USE_V6, main transformer keeps its full region-weighted flow
            # loss (everywhere) — same as baseline. v6 heads add SPECIALIZED
            # residual refinement on top via L_repair_v6 / L_route_v6.
            # This gives the main transformer normal denoising training while
            # the heads contribute class-specific corrections.
            m_core_p   = _pack_mask(garment_prior).float()
            m_repair_p = _pack_mask(repair_band).float()
            m_ub_p     = _pack_mask(uncertain_band).float()
            m_keep_p   = _pack_mask(keep_mask).float()
            wc  = float(os.environ.get("W_FLOW_CORE",     "1.0"))
            wr  = float(os.environ.get("W_FLOW_REPAIR",   "0.1"))
            wub = float(os.environ.get("W_FLOW_UB",       "0.3"))
            wk  = float(os.environ.get("W_FLOW_KEEP",     "0.05"))
            L_flow = (wc * _reg_mse(m_core_p) + wr * _reg_mse(m_repair_p)
                    + wub * _reg_mse(m_ub_p) + wk * _reg_mse(m_keep_p))
        elif "densepose" in batch:
            # 5-way split: core / body_repair / bg_repair / uncertain / keep
            m_core_p = _pack_mask(garment_prior).float()
            m_body_p = _pack_mask(body_repair).float()
            m_bg_p   = _pack_mask(bg_repair).float()
            m_ub_p   = _pack_mask(uncertain_band).float()
            m_keep_p = _pack_mask(keep_mask).float()

            wc  = float(os.environ.get("W_FLOW_CORE",        "1.0"))
            wbr = float(os.environ.get("W_FLOW_BODY_REPAIR", "0.3"))
            wbg = float(os.environ.get("W_FLOW_BG_REPAIR",   "0.3"))
            wub = float(os.environ.get("W_FLOW_UB",          "0.3"))
            wk  = float(os.environ.get("W_FLOW_KEEP",        "0.05"))
            L_flow = (wc * _reg_mse(m_core_p) + wbr * _reg_mse(m_body_p) + wbg * _reg_mse(m_bg_p)
                    + wub * _reg_mse(m_ub_p) + wk * _reg_mse(m_keep_p))
        else:
            # 4-way split: core / repair / uncertain / keep (no densepose)
            m_core_p   = _pack_mask(garment_prior).float()
            m_repair_p = _pack_mask(repair_band).float()
            m_ub_p     = _pack_mask(uncertain_band).float()
            m_keep_p   = _pack_mask(keep_mask).float()

            wc  = float(os.environ.get("W_FLOW_CORE",     "1.0"))
            wr  = float(os.environ.get("W_FLOW_REPAIR",   "0.1"))
            wub = float(os.environ.get("W_FLOW_UB",       "0.3"))
            wk  = float(os.environ.get("W_FLOW_KEEP",     "0.3"))
            L_flow = (wc * _reg_mse(m_core_p) + wr * _reg_mse(m_repair_p)
                    + wub * _reg_mse(m_ub_p) + wk * _reg_mse(m_keep_p))
    # SNR weighting: downweight high sigma (broad coarse), upweight low sigma (refinement)
    # weight = 1-sigma: 0 at sigma=1, 1 at sigma=0
    elif int(os.environ.get("SNR_WEIGHT", "0")):
        snr_w = (1.0 - sigma).view(B, 1, 1).float()
        L_flow = (sq_err * Wmap_p.float() * snr_w).mean()
    else:
        L_flow = (sq_err * Wmap_p.float()).mean()

    # ── Unpack pred_v for spatial losses ──
    v_pred_lat = unpack_latents(pred_C, B, C, H, W)                                # (B, 16, 128, 96)
    x0_pred    = C_t - s * v_pred_lat

    # ── Specialized heads: produce δ (repair residual) + route logits from v_pred features ──
    # Each head has its own parameters → their gradients don't cross.
    # - RepairHead: L1 δ residual supervised in ring (trains repair_head only)
    # - RoutingHead: CE 4-class supervised globally (trains routing_head only)
    # - Main transformer v_pred: flow MSE supervised in M_g (trains main only)
    delta_s_v6 = torch.zeros_like(v_pred_lat)
    delta_b_v6 = torch.zeros_like(v_pred_lat)
    route_logits = None
    if use_v6 and "hidden" in _HIDDEN_HOLDER:
        hidden_full = _HIDDEN_HOLDER["hidden"]                                       # (B, N_total, 3072)
        # Slice off image-token portion (first C_p_.size(1) tokens after txt). For
        # this transformer the image tokens are at positions [:N_img] of the merged
        # output; norm_out output keeps the same layout. We use C_p_ size = 3072.
        hidden_C = hidden_full[:, :C_p_.size(1), :]
        v6_heads = _get_v6_heads(device, weight_dtype, hidden_dim=hidden_C.shape[-1])
        v6_out = v6_heads(hidden_C)
        delta_s_lat = unpack_latents(v6_out["delta_s_packed"], B, C, H, W)           # (B, 16, H, W)
        delta_b_lat = unpack_latents(v6_out["delta_b_packed"], B, C, H, W)
        delta_s_v6 = delta_s_lat
        delta_b_v6 = delta_b_lat
        # Routing logits: (B, N, 16) → (B, 4, H, W)  (16 dims = 4 classes × 2×2 patch)
        route_packed = v6_out["route_logits"]
        H2, W2 = H // 2, W // 2
        route_logits = route_packed.view(B, H2, W2, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 4, H, W)

    # ── Timestep-dependent early weight ──
    # sigma ∈ [0, 1]. High sigma = early denoising. We penalize strongly when sigma > 0.5.
    # w_early decays from 1.0 at sigma=1 to 0.0 at sigma=0.3, zero below.
    w_early = ((sigma.view(B, 1, 1, 1) - 0.3) / 0.7).clamp(0, 1).float()         # (B, 1, 1, 1)

    # ── Interior mask (garment interior, away from boundary) ──
    interior_mask = (garment_prior > 0.7).float()                                  # confident garment interior
    int_area = interior_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)

    # ── A. Early ALLOCATION penalty: interior should not dominate boundary ──
    # Doesn't suppress v_pred magnitude — constrains WHERE the update budget goes.
    # If interior energy >> boundary energy early, that's broad coarse sludge.
    # If boundary gets proportional share, the model is doing structured repair first.
    v_abs = v_pred_lat.float().abs().mean(dim=1, keepdim=True)                     # (B, 1, H, W)
    # Boundary = thin ring around garment edge (from garment_prior)
    gp_dilated = F.max_pool2d(garment_prior, kernel_size=5, stride=1, padding=2)
    gp_eroded  = -F.max_pool2d(-garment_prior, kernel_size=5, stride=1, padding=2)
    boundary_mask = (gp_dilated - gp_eroded).clamp(0, 1)
    E_int  = (v_abs * interior_mask).sum(dim=(-2, -1))                             # (B, 1)
    E_bdry = (v_abs * boundary_mask).sum(dim=(-2, -1))                             # (B, 1)
    r_max = float(os.environ.get("ALLOC_R_MAX", "1.0"))
    ratio = E_int / (E_bdry + 1e-6)
    L_early_alloc = (w_early.view(B, -1) * F.relu(ratio - r_max)).mean()

    # ── B. Early BROAD RATIO: active fraction in interior relative to boundary ──
    # Broad interior activation is only bad if disproportionate to boundary activation.
    energy = v_abs
    gar_energy = energy * garment_prior.float()
    gar_flat = gar_energy.flatten(2)
    gar_sorted = gar_flat.sort(dim=-1).values
    n_gar = (garment_prior.float().flatten(2).sum(dim=-1, keepdim=True)).clamp(min=1).long()
    idx_75 = (n_gar * 75 // 100).clamp(0, gar_sorted.shape[-1] - 1)
    tau = gar_sorted.gather(-1, idx_75).unsqueeze(-1)
    active = (energy > tau).float()
    f_int  = (active * interior_mask).sum(dim=(-2, -1)) / (interior_mask.sum(dim=(-2, -1)) + 1e-6)
    f_bdry = (active * boundary_mask).sum(dim=(-2, -1)) / (boundary_mask.sum(dim=(-2, -1)) + 1e-6)
    rho_max = float(os.environ.get("BROAD_RHO_MAX", "1.5"))
    frac_ratio = f_int / (f_bdry + 1e-6)
    L_early_broad = (w_early.view(B, -1) * F.relu(frac_ratio - rho_max)).mean()

    # ── Anti-sludge in uncertain band (kept, L1 sparsity) ──
    L_antisludge = ((v_abs * uncertain_band.float()).sum() / (uncertain_band.float().sum() + 1e-6))

    # ── TV smoothness in uncertain band ──
    ub_exp = uncertain_band.expand_as(v_pred_lat).float()
    v_ub = v_pred_lat.float() * ub_exp
    grad_y = (v_ub[:, :, 1:, :] - v_ub[:, :, :-1, :]).abs()
    grad_x = (v_ub[:, :, :, 1:] - v_ub[:, :, :, :-1]).abs()
    L_tv = grad_y.mean() + grad_x.mean()

    # ── Direct latent x0 recon in uncertain band ──
    diff_l1 = (x0_pred.float() - person.float()).abs().mean(dim=1, keepdim=True)
    L_recon_ub = ((diff_l1 * uncertain_band.float()).sum() / (uncertain_band.float().sum() + 1e-6))

    # ── L_late_shell: x0 recon in repair zone, weighted by (1 - sigma) ──
    # Strong at low σ specifically. Targets the "shell" = repair band residue that
    # persists in the final few denoising steps. (USE_LATE_SHELL=1)
    L_late_shell = torch.tensor(0.0, device=device)
    lambda_late_shell = float(os.environ.get("LAMBDA_LATE_SHELL", "0.0"))
    if lambda_late_shell > 0:
        _rb = repair_band.float()                                                   # (B,1,H,W)
        _ls_power = float(os.environ.get("LATE_SHELL_POWER", "1.0"))
        _late_w = ((1.0 - sigma).clamp(min=0.0).view(B, 1, 1, 1).float()) ** _ls_power
        _num = (diff_l1 * _rb * _late_w).sum()
        _den = (_rb * _late_w).sum().clamp(min=1e-6)
        L_late_shell = (_num / _den).to(L_flow.device, dtype=torch.float32)

    # ── v6 specialized-head auxiliary losses ──
    # These train ONLY the new heads (RepairHead, RoutingHead), not the main transformer.
    L_repair_v6 = torch.tensor(0.0, device=device)
    L_route_v6  = torch.tensor(0.0, device=device)
    if use_v6:
        # δ_s supervised in M_s; δ_b supervised in M_b ∪ M_other.
        # Each head's output trains only on its own region — no cross-region bleed.
        delta_target = (person - agnostic).detach()                                 # (B, 16, H, W) GT residual
        l1_s = (delta_s_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
        l1_b = (delta_b_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
        L_s_loss = (l1_s * M_s_v6.float()).sum() / (M_s_v6.float().sum() + 1e-6)
        L_b_loss = (l1_b * (M_b_v6 + M_other_v6).float()).sum() / ((M_b_v6 + M_other_v6).float().sum() + 1e-6)
        # Encourage δ ≈ 0 outside their respective regions (prevents head leak)
        keep_for_s = (1.0 - M_s_v6).float()
        keep_for_b = (1.0 - (M_b_v6 + M_other_v6)).float()
        L_s_keep = (delta_s_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_s).sum() / (keep_for_s.sum() + 1e-6)
        L_b_keep = (delta_b_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_b).sum() / (keep_for_b.sum() + 1e-6)
        w_rs   = float(os.environ.get("W_V6_DELTA_S",     "0.5"))
        w_rb   = float(os.environ.get("W_V6_DELTA_B",     "0.5"))
        w_keep_d = float(os.environ.get("W_V6_DELTA_KEEP", "1.0"))
        L_repair_v6 = (w_rs * L_s_loss + w_rb * L_b_loss
                     + w_keep_d * (L_s_keep + L_b_keep)).to(L_flow.device, dtype=torch.float32)

        # Routing head CE against Y_route (4-class: 0=g, 1=s, 2=b, 3=k / other absorbed into k)
        # Build Y_route (B, H, W) long tensor from masks
        y_route = torch.zeros((B, H, W), device=device, dtype=torch.long)
        y_route[(M_g_v6[:, 0] > 0.5)] = 0
        y_route[(M_s_v6[:, 0] > 0.5)] = 1
        y_route[(M_b_v6[:, 0] > 0.5)] = 2
        # M_k and M_other both treated as "keep/other" class = 3
        y_route[(M_k_v6[:, 0]    > 0.5)] = 3
        y_route[(M_other_v6[:, 0] > 0.5)] = 3
        L_route_v6 = F.cross_entropy(route_logits.float(), y_route).to(L_flow.device, dtype=torch.float32)

    # ── L_tv_edge: TV smoothness on x0_pred at silhouette boundary ──
    # Penalizes sharp gradients across the silhouette line so the model can't
    # memorize a visible boundary. Wide dilation to cover the observed line region.
    L_tv_edge = torch.tensor(0.0, device=device)
    lambda_tv_edge = float(os.environ.get("LAMBDA_TV_EDGE", "0.0"))
    if lambda_tv_edge > 0:
        # Dilated boundary band around garment_prior
        _gp_d = F.max_pool2d(garment_prior.float(), kernel_size=11, stride=1, padding=5)
        _gp_e = -F.max_pool2d(-garment_prior.float(), kernel_size=11, stride=1, padding=5)
        _edge_band = (_gp_d - _gp_e).clamp(0, 1)                                     # wide ring
        _x0f = x0_pred.float()
        _gy = (_x0f[:, :, 1:, :] - _x0f[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        _gx = (_x0f[:, :, :, 1:] - _x0f[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        _eb_y = _edge_band[:, :, :-1, :]
        _eb_x = _edge_band[:, :, :, :-1]
        L_tv_edge = ((_gy * _eb_y).sum() / (_eb_y.sum() + 1e-6)
                      + (_gx * _eb_x).sum() / (_eb_x.sum() + 1e-6))
        L_tv_edge = L_tv_edge.to(L_flow.device, dtype=torch.float32)

    # ── L_anti_grey: penalize pred for matching the agnostic's masked grey in repair zone ──
    # Model tends to output the agnostic grey-mask color in repair zone (safest guess when
    # content varies across samples). Compute per-sample grey reference from agnostic's
    # masked pixels, penalize pred being too close in the repair band.
    L_anti_grey = torch.tensor(0.0, device=device)
    lambda_anti_grey = float(os.environ.get("LAMBDA_ANTI_GREY", "0.0"))
    if lambda_anti_grey > 0:
        m_ag = M_full.float()
        grey_ref = ((agnostic.float() * m_ag).sum(dim=(-2, -1), keepdim=True)
                    / (m_ag.sum(dim=(-2, -1), keepdim=True) + 1e-6))                 # (B, 16, 1, 1)
        dist_to_grey = (x0_pred.float() - grey_ref).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        margin = float(os.environ.get("ANTI_GREY_MARGIN", "0.3"))
        rb = repair_band.float()
        # Penalty when distance < margin (pred too close to grey)
        L_anti_grey = (F.relu(margin - dist_to_grey) * rb).sum() / (rb.sum() + 1e-6)

    # ── Image-space L1 loss (soft-weighted by garment_prior + offset) ──
    # ── Sigma-scheduled image loss weighting (USE_SIGMA_LOSS_SCHED=1) ──
    # High sigma (early): punish silhouette errors hard (boundary + transition zone)
    # Low sigma (late): punish identity errors hard (garment interior)
    # Keeps both signals present but shifts emphasis across the trajectory.
    if int(os.environ.get("USE_SIGMA_LOSS_SCHED", "0")):
        s_map = sigma.view(B, 1, 1, 1).float()                                  # (B,1,1,1)
        silhouette_mask = (boundary_mask + 0.5 * uncertain_band).clamp(0, 1.5)
        identity_mask = (garment_prior - uncertain_band).clamp(min=0.0)
        if int(os.environ.get("USE_IMG_CROSSOVER", "0")):
            # Smooth crossover w/ non-zero floors: supervision present everywhere,
            # emphasis shifts smoothly. High σ: boundary high, interior low.
            # Low  σ: interior high, boundary low. Mid σ: comparable.
            bdry_lo = float(os.environ.get("IMG_BDRY_LO", "0.3"))
            bdry_hi = float(os.environ.get("IMG_BDRY_HI", "2.0"))
            int_lo  = float(os.environ.get("IMG_INT_LO",  "0.2"))
            int_hi  = float(os.environ.get("IMG_INT_HI",  "2.0"))
            w_bdry_img = bdry_lo + (bdry_hi - bdry_lo) * s_map
            w_int_img  = int_lo  + (int_hi  - int_lo)  * (1.0 - s_map)
            img_weight_map = (
                0.05 * keep_mask
                + w_bdry_img * silhouette_mask
                + w_int_img  * identity_mask
            )
        else:
            silh_early_w = float(os.environ.get("SILH_EARLY_W", "2.0"))
            id_late_w    = float(os.environ.get("ID_LATE_W",    "3.0"))
            img_weight_map = (
                silh_early_w * s_map * silhouette_mask +    # silhouette punish at high sigma
                id_late_w    * (1 - s_map) * identity_mask +# identity punish at low sigma
                0.05 * keep_mask                            # small constant outside weight
            )
    else:
        if int(os.environ.get("USE_DP_SPLIT", "0")) and "densepose" in batch:
            _img_wbody = float(os.environ.get("IMG_WEIGHT_BODY_REPAIR", "0.3"))
            _img_wbg   = float(os.environ.get("IMG_WEIGHT_BG_REPAIR",   "2.0"))
            img_weight_map = (garment_prior + 0.3 * uncertain_band
                              + _img_wbody * body_repair
                              + _img_wbg   * bg_repair
                              + 0.05 * keep_mask)
        else:
            _img_wrep = float(os.environ.get("IMG_WEIGHT_REPAIR", "0.3"))
            img_weight_map = (garment_prior + 0.3 * uncertain_band + _img_wrep * repair_band + 0.05 * keep_mask)
    x0_5d = x0_pred.to(vae_device, dtype=weight_dtype).unsqueeze(2)
    m_v = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
    s_v = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
    denorm = x0_5d * s_v + m_v
    with torch.amp.autocast("cuda", dtype=weight_dtype):
        decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0]
    pred_img = decoded.clamp(-1, 1)
    Hi, Wi = pred_img.shape[2], pred_img.shape[3]
    weight_map_img = F.interpolate(img_weight_map, size=(Hi, Wi), mode="bilinear",
                                   align_corners=False).to(vae_device, weight_dtype)

    # ── Boundary-ring match weight (W_BOUNDARY_MATCH) ──
    # The visible halo sits at the inner N-pixel ring of the agnostic boundary
    # (pred meets real source). Add extra L1 weight on that ring to force pred
    # to match GT (= source) exactly there, eliminating the edge discontinuity.
    _w_bmatch = float(os.environ.get("W_BOUNDARY_MATCH", "0.0"))
    if _w_bmatch > 0:
        _br = int(os.environ.get("BOUNDARY_MATCH_RING", "6"))     # px at img resolution
        _M_img_hard = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest")
        _M_eroded = -F.max_pool2d(-_M_img_hard, 2*_br+1, 1, _br)
        _inner_ring = (_M_img_hard - _M_eroded).clamp(0, 1)         # (B,1,Hi,Wi)
        _inner_ring = _inner_ring.to(vae_device, weight_dtype)
        weight_map_img = weight_map_img + _w_bmatch * _inner_ring

    person_imgs = torch.stack([person_image_cache[iid] for iid in image_ids]).to(vae_device, dtype=weight_dtype)

    # ── USE_IMG_REGION_SPLIT=1: per-region normalized L1 for image loss ──
    # L_img = W_IC*|pred-gt|_core + W_IR*|pred-gt|_repair + W_IB*|pred-gt|_ub
    #       + W_IK*|pred-gt|_keep   (each normalized by region pixel count)
    # In keep region, gt == source, so W_IK > 0 preserves source.
    # Low W_IR reduces repair freedom; high W_IC pushes garment fidelity.
    if use_v6:
        pix_err = (pred_img - person_imgs).abs().mean(dim=1, keepdim=True)               # (B,1,Hi,Wi)
        def _up(m):
            return F.interpolate(m.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _g_i     = _up(M_g_v6)
        _s_i     = _up(M_s_v6)
        _b_i     = _up(M_b_v6)
        _other_i = _up(M_other_v6)
        _k_i     = _up(M_k_v6)
        _ub_i    = _up(uncertain_band)

        def _reg_l1(mask):
            denom = mask.sum() + 1e-6
            return (pix_err * mask).sum() / denom

        L_img_g     = _reg_l1(_g_i)
        L_img_s     = _reg_l1(_s_i)
        L_img_b     = _reg_l1(_b_i)
        L_img_other = _reg_l1(_other_i)
        L_img_k     = _reg_l1(_k_i)
        L_img_ub    = _reg_l1(_ub_i)

        wig  = float(os.environ.get("W_IMG_V6_G",     "1.0"))     # garment (high)
        wis  = float(os.environ.get("W_IMG_V6_S",     "0.5"))     # skin repair
        wib  = float(os.environ.get("W_IMG_V6_B",     "0.5"))     # bg repair
        wio  = float(os.environ.get("W_IMG_V6_OTHER", "1.0"))     # ring fallback
        wik  = float(os.environ.get("W_IMG_V6_K",     "1.0"))     # keep preserves source
        wiub = float(os.environ.get("W_IMG_V6_UB",    "2.0"))     # boundary
        L_img = (wig * L_img_g + wis * L_img_s + wib * L_img_b
               + wio * L_img_other + wik * L_img_k + wiub * L_img_ub)
        L_img = L_img.to(L_flow.device, dtype=torch.float32)
    elif int(os.environ.get("USE_IMG_REGION_SPLIT", "0")):
        pix_err = (pred_img - person_imgs).abs().mean(dim=1, keepdim=True)               # (B,1,Hi,Wi)
        def _up(m):
            return F.interpolate(m.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _core_i   = _up(garment_prior)
        _ub_i     = _up(uncertain_band)
        _repair_i = _up(repair_band)
        _keep_i   = _up(keep_mask)

        def _reg_l1(mask):
            denom = mask.sum() + 1e-6
            return (pix_err * mask).sum() / denom

        L_img_core   = _reg_l1(_core_i)
        L_img_ub     = _reg_l1(_ub_i)
        L_img_repair = _reg_l1(_repair_i)
        L_img_keep   = _reg_l1(_keep_i)

        wic = float(os.environ.get("W_IMG_CORE",     "1.0"))
        wir = float(os.environ.get("W_IMG_REPAIR",   "0.05"))
        wib = float(os.environ.get("W_IMG_UB",       "0.3"))
        wik = float(os.environ.get("W_IMG_KEEP",     "0.3"))
        L_img = wic * L_img_core + wir * L_img_repair + wib * L_img_ub + wik * L_img_keep
        L_img = L_img.to(L_flow.device, dtype=torch.float32)
    else:
        L_img = ((pred_img - person_imgs).abs() * weight_map_img).mean()
        L_img = L_img.to(L_flow.device, dtype=torch.float32)

    # ── v6 boundary loss on composed x_hat_0 ──
    # Compose: M_k*agn + M_g*x0_g + M_s*(agn+δ_s) + (M_b+M_other)*(agn+δ_b)
    # Boundary loss penalizes mismatch at class boundaries (collar, sleeve hem,
    # silhouette edges) where transitions are perceptually critical.
    L_v6_boundary = torch.tensor(0.0, device=device)
    if use_v6:
        x_repair_s_lat = agnostic + delta_s_v6
        x_repair_b_lat = agnostic + delta_b_v6
        ring_v6_full = (M_s_v6 + M_b_v6 + M_other_v6).clamp(0, 1).float()
        x_hat_0 = (M_k_v6 * agnostic + M_g_v6 * x0_pred
                 + M_s_v6 * x_repair_s_lat
                 + (M_b_v6 + M_other_v6) * x_repair_b_lat).float()
        _Mg_dil = F.max_pool2d(M_g_v6.float(), 3, 1, 1)
        _ring_dil = F.max_pool2d(ring_v6_full, 3, 1, 1)
        bnd_v6 = (_Mg_dil * _ring_dil).clamp(0, 1)
        wub_v6 = float(os.environ.get("W_V6_UB", "2.0"))
        bnd_diff = (x_hat_0 - person.float()).abs().mean(dim=1, keepdim=True)
        L_v6_boundary = wub_v6 * (bnd_diff * bnd_v6).sum() / (bnd_v6.sum() + 1e-6)
        L_v6_boundary = L_v6_boundary.to(L_flow.device, dtype=torch.float32)
    L_v6_repair_dummy = torch.tensor(0.0, device=device)
    L_v6_keep_dummy   = torch.tensor(0.0, device=device)

    # ── TV smoothness in the inner agnostic ring (LAMBDA_TV_AGN_RING) ──
    # Penalizes sharp color gradients AT the agnostic boundary ring, preventing
    # the visible halo/edge-line by making pred smooth across that transition.
    L_tv_ring = torch.tensor(0.0, device=device)
    lambda_tv_ring = float(os.environ.get("LAMBDA_TV_AGN_RING", "0.0"))
    if lambda_tv_ring > 0:
        _tvr = int(os.environ.get("TV_AGN_RING_PX", "6"))
        _M_img_hard_tv = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest")
        _M_eroded_tv = -F.max_pool2d(-_M_img_hard_tv, 2*_tvr+1, 1, _tvr)
        _ring_tv = (_M_img_hard_tv - _M_eroded_tv).clamp(0, 1).to(vae_device, weight_dtype)
        pred_f = pred_img.float()
        _gy = (pred_f[:, :, 1:, :] - pred_f[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        _gx = (pred_f[:, :, :, 1:] - pred_f[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        _ring_y = _ring_tv[:, :, 1:, :].float()
        _ring_x = _ring_tv[:, :, :, 1:].float()
        L_tv_ring = ((_gy * _ring_y).sum() / (_ring_y.sum() + 1e-6)
                   + (_gx * _ring_x).sum() / (_ring_x.sum() + 1e-6))
        L_tv_ring = L_tv_ring.to(L_flow.device, dtype=torch.float32)

    # ── L_no_bg_leak: penalize pred matching out-of-mask "background" in repair band ──
    # The halo/edge-line forms because repair-band pred gets dragged toward the
    # background color (white) that surrounds the person outside the agnostic mask.
    # We compute the mean per-image pixel value in the "outside mask" region of the
    # cached person image (as a proxy for "what the out-of-agnostic area looks like"),
    # and penalize pred being CLOSE to that mean value in the repair band.
    # Push pred AWAY from bg_mean in the repair zone specifically.
    L_no_bg_leak = torch.tensor(0.0, device=device)
    lambda_no_bg = float(os.environ.get("LAMBDA_NO_BG_LEAK", "0.0"))
    if lambda_no_bg > 0:
        # Compute bg mean per sample from person_imgs outside agnostic mask.
        # person_imgs is (B, 3, Hi, Wi) in [-1, 1]; M_full is at latent res so upsample.
        M_full_img = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        outside = (1.0 - M_full_img).expand_as(person_imgs)                # (B, 3, Hi, Wi)
        denom = outside.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        bg_mean = (person_imgs * outside).sum(dim=(1, 2, 3), keepdim=True) / denom  # (B, 1, 1, 1)
        repair_img_mask_bg = F.interpolate(repair_band.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        # Distance from bg_mean in repair band (per-pixel). We want to MAXIMIZE this.
        dist_from_bg = (pred_img - bg_mean).abs().mean(dim=1, keepdim=True)   # (B, 1, Hi, Wi)
        # Penalize CLOSENESS to bg — i.e., loss = -distance (so gradient pushes dist up)
        L_no_bg_leak = -(dist_from_bg * repair_img_mask_bg).sum() / (repair_img_mask_bg.sum() + 1e-6)
        L_no_bg_leak = L_no_bg_leak.to(L_flow.device, dtype=torch.float32)

    # VGG perceptual loss (optional, enabled by USE_PERCEPTUAL env var)
    L_percep = torch.tensor(0.0, device=device)
    if int(os.environ.get("USE_PERCEPTUAL", "0")):
        vgg = get_vgg_features(vae_device, weight_dtype)
        L_percep = perceptual_loss(pred_img, person_imgs, weight_map_img, vgg)
        L_percep = L_percep.to(L_flow.device, dtype=torch.float32)

    # ── L_adv: PatchGAN adversarial loss on repair zone (USE_ADV=1) ──
    # Discriminator learns real vs pred on the repair zone. Generator gets pulled
    # toward the real-image manifold, breaking MSE averaging's low-variance haze.
    # Hinge loss (standard for stability).
    L_adv = torch.tensor(0.0, device=device)
    lambda_adv = float(os.environ.get("LAMBDA_ADV", "0.0"))
    if lambda_adv > 0:
        D, D_opt = _get_discriminator(vae_device, weight_dtype)
        repair_img_mask = F.interpolate(repair_band.float().to(vae_device),
                                         size=(Hi, Wi), mode="nearest")
        # Focus D on repair zone by zeroing outside (keeps resolution)
        real_patch = (person_imgs * repair_img_mask).detach()
        fake_patch = (pred_img * repair_img_mask).detach()
        # D step: train D to distinguish
        with torch.enable_grad():
            D.train()
            d_real = D(real_patch.to(dtype=weight_dtype))
            d_fake = D(fake_patch.to(dtype=weight_dtype))
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            D_opt.zero_grad()
            d_loss.backward()
            D_opt.step()
        # G step: adversarial loss pulls pred toward real
        D.eval()
        fake_for_g = pred_img * repair_img_mask
        d_fake_g = D(fake_for_g.to(dtype=weight_dtype))
        L_adv = (-d_fake_g.mean()).to(L_flow.device, dtype=torch.float32)

    # ── L_anti_rough_hf: discourage copying rough HF content (exp486+) ──
    # Decodes rough latent, computes highpass of both pred and rough, and penalizes
    # their positive correlation in the edit region. Pushes pred's HF content to be
    # UNCORRELATED with rough's HF artifacts (stripes/floral/texture residue).
    L_anti_rough = torch.tensor(0.0, device=device)
    lambda_anti_rough = float(os.environ.get("LAMBDA_ANTI_ROUGH_HF", "0.0"))
    if lambda_anti_rough > 0:
        from torchvision.transforms.functional import gaussian_blur
        rough_5d = rough.to(vae_device, dtype=weight_dtype).unsqueeze(2)
        rough_denorm = rough_5d * s_v + m_v
        with torch.amp.autocast("cuda", dtype=weight_dtype):
            rough_dec = vae.decode(rough_denorm, return_dict=False)[0][:, :, 0]
        rough_img = rough_dec.clamp(-1, 1)
        k_hp = 7; sig_hp = 3.0
        pred_f = pred_img.float()
        rough_f = rough_img.float()
        pred_hp  = pred_f  - gaussian_blur(pred_f,  [k_hp, k_hp], sig_hp)
        rough_hp = rough_f - gaussian_blur(rough_f, [k_hp, k_hp], sig_hp)
        mask_edit_img = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest").to(vae_device)
        # Positive per-pixel product = aligned HF patterns. Penalize only positive side.
        prod = (pred_hp * rough_hp * mask_edit_img.expand_as(pred_hp))
        L_anti_rough = F.relu(prod.mean()).to(L_flow.device, dtype=torch.float32)

    # ── Gate losses (exp420: weak prior + entropy on α) ──
    L_gate_prior = torch.tensor(0.0, device=device)
    L_gate_entropy = torch.tensor(0.0, device=device)
    # Collect alphas from all installed GarmentRepairGate modules
    gate_alphas = []
    if GARMENT_GATES:
        for gate in GARMENT_GATES:
            if hasattr(gate, 'last_alpha') and gate.last_alpha is not None:
                a = gate.last_alpha.float()                                        # (B, N_c, 1)
                gate_alphas.append(a)
        if gate_alphas:
            all_alpha = torch.cat(gate_alphas, dim=0)                              # (n_gates*B, N_c, 1)
            # Weak prior: garment_prior_tok → encourage α high where garment expected
            gp_tok = pack_latents(garment_prior.expand(B, C, H, W), B, C, H, W).mean(dim=-1, keepdim=True)
            # soft target = 0.7 where garment, 0.3 elsewhere
            alpha_prior = 0.3 + 0.4 * gp_tok.float()                              # (B, N_c, 1)
            # BCE per gate, averaged
            # Manual BCE (autocast-safe): -t*log(p) - (1-t)*log(1-p)
            def _safe_bce(pred, target):
                p = pred.float().clamp(1e-6, 1-1e-6)
                t = target.float()
                return -(t * p.log() + (1 - t) * (1 - p).log()).mean()
            L_gate_prior = sum(
                _safe_bce(a, alpha_prior.expand_as(a))
                for a in gate_alphas
            ) / len(gate_alphas)
            # Entropy penalty: encourage confident (0 or 1), not mushy 0.5
            L_gate_entropy = sum(
                -(a * (a + 1e-6).log() + (1 - a) * (1 - a + 1e-6).log()).mean()
                for a in gate_alphas
            ) / len(gate_alphas)

            # NO-EDIT zone penalty: severely punish any α > 0 in keep_mask region
            # keep_mask (1 outside edit region) packed to token level
            keep_tok = pack_latents(keep_mask.expand(B, C, H, W), B, C, H, W).mean(dim=-1, keepdim=True)
            L_noedit = sum(
                (a.float() * keep_tok.float()).sum() / (keep_tok.float().sum() + 1e-6)
                for a in gate_alphas
            ) / len(gate_alphas)

    # ── Total loss ──
    lambda_recon = float(os.environ.get("LAMBDA_RECON", "0.3"))
    lambda_antisludge = float(os.environ.get("LAMBDA_ANTISLUDGE", "0.3"))
    lambda_tv = float(os.environ.get("LAMBDA_TV", "0.03"))
    lambda_alloc = float(os.environ.get("LAMBDA_ALLOC", "0.1"))
    lambda_broad_ratio = float(os.environ.get("LAMBDA_BROAD_RATIO", "0.1"))
    lambda_percep = float(os.environ.get("LAMBDA_PERCEPTUAL", "0.1"))
    w_flow = float(os.environ.get("W_FLOW", "1.0"))
    loss = (w_flow * L_flow
            + img_loss_weight * L_img
            + lambda_recon * L_recon_ub
            + lambda_antisludge * L_antisludge
            + lambda_tv * L_tv
            + lambda_alloc * L_early_alloc
            + lambda_broad_ratio * L_early_broad
            + lambda_percep * L_percep
            + lambda_anti_rough * L_anti_rough
            + lambda_anti_grey * L_anti_grey
            + lambda_adv * L_adv
            + lambda_late_shell * L_late_shell
            + lambda_tv_edge * L_tv_edge
            + lambda_tv_ring * L_tv_ring
            + lambda_no_bg * L_no_bg_leak
            + float(os.environ.get("LAMBDA_V6_REPAIR", "1.0")) * L_repair_v6
            + float(os.environ.get("LAMBDA_V6_ROUTE",  "0.5")) * L_route_v6
            + L_v6_boundary)

    return loss, {"flow": L_flow.item(),
                  "img": L_img.item(),
                  "recon": L_recon_ub.item(),
                  "anti": L_antisludge.item(),
                  "tv": L_tv.item(),
                  "alloc": L_early_alloc.item(),
                  "broad": L_early_broad.item(),
                  "percep": L_percep.item(),
                  "antirough": L_anti_rough.item(),
                  "antigrey": L_anti_grey.item(),
                  "adv": L_adv.item(),
                  "late": L_late_shell.item(),
                  "sigma": sigma.mean().item()}


# ─────────────────────────── Main ───────────────────────────

INFERENCE_TEMPLATE = os.path.join(BASE, "vtonautoresearch", "inference_template.py")

def main():
    args = Args()
    args.sigma_beta_alpha = float(os.environ.get("SIGMA_BETA_ALPHA", str(args.sigma_beta_alpha)))
    args.sigma_beta_beta = float(os.environ.get("SIGMA_BETA_BETA", str(args.sigma_beta_beta)))
    device = torch.device(args.device_transformer); dtype = torch.bfloat16
    vae_device = torch.device("cuda:1")

    run_name = f"vton_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = asdict(args)
    cfg["slot_order"] = SLOT_ORDER
    cfg["slot_order_idx"] = SLOT_ORDER_IDX
    # Region weights (read from env or default) — picked up by generate_panel.py
    cfg["w_out"]  = float(os.environ.get("W_OUTSIDE",  "0.05"))
    cfg["w_core"] = float(os.environ.get("W_CORE",     "1.0"))
    cfg["w_rep"]  = float(os.environ.get("W_REPAIR",   "0.25"))
    cfg["w_bdy"]  = float(os.environ.get("W_BOUNDARY", "1.0"))
    cfg["lambda_repair"] = float(os.environ.get("LAMBDA_REPAIR", "0.25"))
    cfg["cfg_dropout"] = float(os.environ.get("CFG_DROPOUT", "0.0"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = os.path.join(args.output_dir, "train.log")
    log = logging.getLogger("vton"); log.setLevel(logging.INFO); log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path)); log.addHandler(logging.StreamHandler())
    torch.manual_seed(args.seed)
    # Strict determinism (for reproducibility across runs)
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if int(os.environ.get("DETERMINISTIC", "0")):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    # ── Stage 1: load QwenImageEditPlusPipeline on GPU 1, keep VAE + text_encoder, drop transformer ──
    log.info("Loading QwenImageEditPlusPipeline on cuda:1 (text encoder + VAE only)...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=dtype, transformer=None,
    )
    pipe = pipe.to(vae_device)

    # VAE-decode degraded_rough_latent → PIL per sample (for Qwen2.5-VL semantic pass)
    log.info("VAE-decoding rough latents to PIL...")
    rough_pils = precompute_rough_pils(args.latent_cache_dir, pipe.vae, vae_device, dtype)

    # Per-sample Qwen2.5-VL encode: [agnostic_pil, pose_pil, rough_pil, garment_pil] + fixed prompt
    log.info("Encoding per-sample prompts via Qwen2.5-VL...")
    prompt_cache = precompute_prompt_embeds(pipe, args.latent_cache_dir, LOCAL_CACHE, rough_pils, vae_device, dtype)
    for sid, (pe, pm) in prompt_cache.items():
        log.info(f"  prompt[{sid}]: pe={tuple(pe.shape)} pm={tuple(pm.shape)} {pe.dtype}")
    # move prompt cache to GPU 0 for training
    prompt_cache = {k: (v[0].to(device, dtype=dtype), v[1].to(device, dtype=torch.long))
                    for k, v in prompt_cache.items()}

    # Free the text encoder and pipeline VAE
    del pipe
    torch.cuda.empty_cache()

    # Load pose latents (precomputed VAE-encoded densepose RGB, one per sample)
    log.info("Loading pose latents from local cache...")
    pose_cache = load_pose_latents(LOCAL_CACHE, device, dtype)
    for sid, lat in pose_cache.items():
        log.info(f"  pose[{sid}]: {tuple(lat.shape)} {lat.dtype}")

    # Load VAE on cuda:1 for image-space L1 loss
    log.info("Loading VAE on cuda:1 for image-space L1 loss...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=dtype)
    vae.to(vae_device).eval()
    vae.requires_grad_(False)

    # Precompute decoded person image per sample (cached at startup, no per-step recompute)
    log.info("Precomputing decoded person images for image-space loss...")
    person_image_cache = {}
    m_v = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    s_v = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    for sid in TRAIN_IDS:
        plat = torch.load(os.path.join(args.latent_cache_dir, f"{sid}_person_latent.pt"),
                          weights_only=True).unsqueeze(0).unsqueeze(2).to(vae_device, dtype)  # (1,16,1,128,96)
        denorm = plat * s_v + m_v
        with torch.no_grad():
            decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0]                      # (1,3,Hi,Wi)
        person_image_cache[sid] = decoded.clamp(-1, 1)[0].to(vae_device, dtype=dtype).detach()
    for sid, img in person_image_cache.items():
        log.info(f"  person_image[{sid}]: {tuple(img.shape)} {img.dtype}")

    # ── Stage 2: load transformer + tryon LoRA on GPU 0 ──
    log.info("Loading transformer on cuda:0...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
    transformer = get_peft_model(
        transformer,
        LoraConfig(r=args.rank, lora_alpha=args.alpha,
                   init_lora_weights=args.init_lora_weights,
                   target_modules=args.lora_targets, lora_dropout=0.0),
        adapter_name="tryon")
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(device)

    # Install HoleyAttnProcessor when attention masking is enabled
    if int(os.environ.get("USE_REPAIR_ATTN_MASK", "0")):
        n_proc = 0
        for mod in transformer.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = HoleyAttnProcessor()
                n_proc += 1
        log.info(f"HoleyAttnProcessor installed on {n_proc} attention blocks")

    # exp420: install GarmentRepairGate on transformer blocks
    global GARMENT_GATES
    garment_gates, gate_hooks = [], []
    if int(os.environ.get("USE_GARMENT_GATE", "0")):
        # Figure out garment slot index (position after C in SLOT_ORDER)
        gar_idx = SLOT_ORDER.index("garment") if "garment" in SLOT_ORDER else -1
        if gar_idx >= 0:
            n_c = 3072  # tokens per slot at 128×96 with 2×2 packing
            every_n = int(os.environ.get("GATE_EVERY_N", "3"))
            garment_gates, gate_hooks = install_garment_gates(
                transformer, n_c, gar_idx, device, dtype,
                n_heads=4, head_dim=32, every_n=every_n)
            log.info(f"GarmentRepairGate installed on {len(garment_gates)} blocks "
                     f"(every {every_n}, gar_idx={gar_idx})")

    # Trainable params: tryon LoRA + garment gates
    tryon_params = [p for _, p in transformer.named_parameters() if p.requires_grad]
    gate_params = [p for g in garment_gates for p in g.parameters()]
    log.info(f"tryon_params: {sum(p.numel() for p in tryon_params):,}")
    log.info(f"gate_params:  {sum(p.numel() for p in gate_params):,}")
    GARMENT_GATES = garment_gates

    # Load pretrained LoRA weights if specified (for continued training or frozen gate-only)
    lora_path = os.environ.get("LORA_INIT_PATH", "")
    if lora_path and os.path.exists(lora_path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as sf_load
        set_peft_model_state_dict(transformer, sf_load(lora_path), adapter_name="tryon")
        log.info(f"Loaded pretrained LoRA from {lora_path}")

    if int(os.environ.get("FREEZE_LORA", "0")) and gate_params:
        for p in tryon_params:
            p.requires_grad_(False)
        tryon_params = []
        log.info("LoRA FROZEN — training gate params only")

    param_groups = []
    # Optional per-block LR split — higher LR for early transformer blocks so they
    # develop useful adaptations instead of being drowned by late-block gradients.
    # EARLY_BLOCK_CUTOFF: boundary block idx (inclusive). LR_EARLY_MULT: multiplier
    # applied to args.lr for blocks [0..cutoff]. Blocks > cutoff use args.lr.
    # Default (cutoff=-1): single group, all params at args.lr (exp524 behavior).
    early_cutoff = int(os.environ.get("EARLY_BLOCK_CUTOFF", "-1"))
    early_mult   = float(os.environ.get("LR_EARLY_MULT", "1.0"))
    if tryon_params and early_cutoff >= 0 and early_mult != 1.0:
        early_params, late_params = [], []
        for n, p in transformer.named_parameters():
            if not p.requires_grad: continue
            parts = n.split(".")
            try:
                idx = int(parts[parts.index("transformer_blocks") + 1])
            except (ValueError, IndexError):
                late_params.append(p); continue
            if idx <= early_cutoff:
                early_params.append(p)
            else:
                late_params.append(p)
        if early_params:
            param_groups.append({"params": early_params, "lr": args.lr * early_mult})
        if late_params:
            param_groups.append({"params": late_params, "lr": args.lr})
        log.info(f"per-block LR: blocks 0..{early_cutoff} at lr={args.lr*early_mult:.2e}, "
                 f"blocks {early_cutoff+1}..59 at lr={args.lr:.2e}; "
                 f"{sum(p.numel() for p in early_params):,} early params, "
                 f"{sum(p.numel() for p in late_params):,} late params")
    elif tryon_params:
        param_groups.append({"params": tryon_params, "lr": args.lr})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": args.lr * 3})
    # v6 specialized heads (Linear on 3072-dim Qwen features). Hooks norm_out
    # to capture pre-proj features. Heads have separate gradient paths from
    # the main transformer, so cross-region bleed is bounded by feature
    # sharing only — no direct mask-weight averaging on a single output.
    if int(os.environ.get("USE_V6", "0")):
        v6_heads = _get_v6_heads(device, dtype, hidden_dim=transformer.inner_dim)
        param_groups.append({"params": list(v6_heads.parameters()), "lr": args.lr * 10})
        log.info(f"v6_heads: {sum(p.numel() for p in v6_heads.parameters()):,} params "
                 f"at lr={args.lr*10:.2e} (to_s + to_b + to_route)")
        # Register hook on transformer.norm_out to capture (B, N_total, 3072) features
        transformer.norm_out.register_forward_hook(_v6_hidden_hook)
        log.info("v6: registered forward_hook on transformer.norm_out")
    if not param_groups:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay)

    train_ds = VTONDataset(args, split=args.train_split)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True, prefetch_factor=2)

    img_loss_weight = float(os.environ.get("IMG_LOSS_WEIGHT", "0.1"))
    # Soft-mask flow weights: (w_keep, w_garment, w_uncertain)
    w_keep      = float(os.environ.get("W_KEEP", "0.05"))
    w_garment   = float(os.environ.get("W_GARMENT", "1.0"))
    w_uncertain = float(os.environ.get("W_UNCERTAIN", "0.3"))
    loss_weights = (w_keep, w_garment, w_uncertain)

    log.info(f"Architecture: exp419 (soft-mask routing + anti-sludge) — "
             f"{NUM_SLOTS}-slot slot_order={SLOT_ORDER}; "
             f"raw agnostic+rough (no neutralization); "
             f"soft masks: garment_prior(warped_mask_128) + uncertain_band + keep; "
             f"loss = L_flow(w_keep={w_keep}, w_gar={w_garment}, w_ub={w_uncertain}) "
             f"+ {img_loss_weight}*L_img + "
             f"{os.environ.get('LAMBDA_RECON','0.3')}*L_recon_ub + "
             f"{os.environ.get('LAMBDA_ANTISLUDGE','0.1')}*L_antisludge + "
             f"{os.environ.get('LAMBDA_TV','0.01')}*L_tv")
    optimizer.zero_grad(); global_step = 0; micro_step = 0
    accum = {"flow": 0.0, "img": 0.0, "recon": 0.0,
             "anti": 0.0, "tv": 0.0, "alloc": 0.0, "broad": 0.0, "percep": 0.0,
             "antirough": 0.0, "antigrey": 0.0, "adv": 0.0, "late": 0.0, "sigma": 0.0}
    t_start = time.time()

    import signal
    stop_flag = {"stop": False}
    def _graceful(sig, frame):
        stop_flag["stop"] = True
        log.info(f"Received signal {sig} — finishing current step and saving.")
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    for epoch in range(99999):
        transformer.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            if time.time() - t_start >= TIME_BUDGET or stop_flag["stop"] or global_step >= MAX_STEPS: break
            try:
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss, metrics = train_step(
                        transformer,
                        pose_cache, prompt_cache,
                        vae, person_image_cache, vae_device, img_loss_weight,
                        loss_weights,
                        batch, device, dtype,
                        sigma_beta_alpha=args.sigma_beta_alpha,
                        sigma_beta_beta=args.sigma_beta_beta,
                        global_step=global_step,
                        max_steps=MAX_STEPS)
                (loss / args.grad_accum).backward()
                for k in accum: accum[k] += metrics.get(k, 0.0)
                micro_step += 1
                if micro_step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(tryon_params, args.max_grad_norm)
                    optimizer.step(); optimizer.zero_grad(); global_step += 1
                    if global_step % args.logging_steps == 0:
                        n = args.logging_steps * args.grad_accum
                        log.info(f"step {global_step} flow={accum['flow']/n:.5f} "
                                 f"img={accum['img']/n:.4f} recon={accum['recon']/n:.4f} "
                                 f"anti={accum['anti']/n:.4f} tv={accum['tv']/n:.5f} "
                                 f"alloc={accum['alloc']/n:.4f} broad={accum['broad']/n:.4f} "
                                 f"percep={accum['percep']/n:.4f} "
                                 f"antirough={accum.get('antirough', 0.0)/n:.5f} "
                                 f"σ={accum['sigma']/n:.3f}")
                        accum = {k: 0.0 for k in accum}
            except Exception as e:
                log.error(f"Error step {global_step}: {e}")
                log.error(traceback.format_exc())
                if "cuda" in str(e).lower(): sys.exit(1)
                continue
        if time.time() - t_start >= TIME_BUDGET or stop_flag["stop"] or global_step >= MAX_STEPS: break
        log.info(f"Epoch {epoch} done at step {global_step}")

    # ── Save for inference ──
    final_path = os.path.join(args.output_dir, "final"); os.makedirs(final_path, exist_ok=True)
    save_file(get_peft_model_state_dict(transformer, adapter_name="tryon"),
              os.path.join(final_path, "tryon_lora.safetensors"))
    if int(os.environ.get("USE_V6", "0")) and _V6_HEADS is not None:
        torch.save({k: v.cpu() for k, v in _V6_HEADS.state_dict().items()},
                   os.path.join(final_path, "v6_heads.pt"))
        log.info(f"Saved v6_heads.pt")
    torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in prompt_cache.items()},
               os.path.join(final_path, "prompt_cache.pt"))
    torch.save({k: v.cpu() for k, v in pose_cache.items()},
               os.path.join(final_path, "pose_latent_cache.pt"))
    # Patch inference template with this run's slot order + rough-neutralize flag
    with open(INFERENCE_TEMPLATE) as f:
        template_src = f.read()
    template_src = template_src.replace(
        "DEFAULT_SLOT_ORDER = [0, 1, 2, 3]",
        f"DEFAULT_SLOT_ORDER = {SLOT_ORDER_IDX}  # {SLOT_ORDER}"
    )
    neutralize_rough_val = int(os.environ.get("NEUTRALIZE_ROUGH", "1"))
    template_src = template_src.replace(
        "NEUTRALIZE_ROUGH = 1",
        f"NEUTRALIZE_ROUGH = {neutralize_rough_val}"
    )
    lrr_val = float(os.environ.get("LAMBDA_REPAIR_ROUGH", "0.25"))
    template_src = template_src.replace(
        "LAMBDA_REPAIR_ROUGH = 0.25",
        f"LAMBDA_REPAIR_ROUGH = {lrr_val}"
    )
    proxy_val = float(os.environ.get("PROXY_CORE_THRESH", "0.6"))
    template_src = template_src.replace(
        "diff_in_agn > 0.6 * max_diff",
        f"diff_in_agn > {proxy_val} * max_diff"
    )
    attn_mode_map = {"none": 0, "one_way": 1, "both": 2}
    attn_mode_val = attn_mode_map.get(os.environ.get("ATTN_MASK_MODE", "none"), 0)
    template_src = template_src.replace(
        "ATTN_MASK_MODE = 0",
        f"ATTN_MASK_MODE = {attn_mode_val}"
    )
    pin_ring_val = int(os.environ.get("PIN_RING_ROUGH", "0"))
    template_src = template_src.replace(
        "PIN_RING_ROUGH = 0   # Set by train.py at save time",
        f"PIN_RING_ROUGH = {pin_ring_val}"
    )
    sigma_sched_val = int(os.environ.get("USE_SIGMA_SCHED", "0"))
    template_src = template_src.replace(
        "USE_SIGMA_SCHED = 0  # Set by train.py at save time",
        f"USE_SIGMA_SCHED = {sigma_sched_val}"
    )
    sched_lo = float(os.environ.get("SIGMA_SCHED_LO", "0.8"))
    sched_hi = float(os.environ.get("SIGMA_SCHED_HI", "1.2"))
    template_src = template_src.replace(
        "SIGMA_SCHED_LO = 0.8 # Set by train.py at save time",
        f"SIGMA_SCHED_LO = {sched_lo}"
    )
    template_src = template_src.replace(
        "SIGMA_SCHED_HI = 1.2 # Set by train.py at save time",
        f"SIGMA_SCHED_HI = {sched_hi}"
    )
    pure_noise_val = int(os.environ.get("USE_PURE_NOISE", "0"))
    template_src = template_src.replace(
        "USE_PURE_NOISE = 0   # Set by train.py at save time",
        f"USE_PURE_NOISE = {pure_noise_val}"
    )
    rough_masked_val = int(os.environ.get("USE_ROUGH_MASKED", "0"))
    template_src = template_src.replace(
        "USE_ROUGH_MASKED = 0  # Set by train.py at save time",
        f"USE_ROUGH_MASKED = {rough_masked_val}"
    )
    agn_mean_fill_val = int(os.environ.get("USE_AGNOSTIC_MEAN_FILL", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_MEAN_FILL = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_MEAN_FILL = {agn_mean_fill_val}"
    )
    agn_rough_fill_val = int(os.environ.get("USE_AGNOSTIC_ROUGH_FILL", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_ROUGH_FILL = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_ROUGH_FILL = {agn_rough_fill_val}"
    )
    agn_inpaint_val = int(os.environ.get("USE_AGNOSTIC_INPAINT", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_INPAINT = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_INPAINT = {agn_inpaint_val}"
    )
    sil_scale_val = float(os.environ.get("SILHOUETTE_SCALE", "1.0"))
    template_src = template_src.replace(
        "SILHOUETTE_SCALE = 1.0  # Set by train.py at save time",
        f"SILHOUETTE_SCALE = {sil_scale_val}"
    )
    sil_soft_val = int(os.environ.get("SILHOUETTE_SOFT", "0"))
    template_src = template_src.replace(
        "SILHOUETTE_SOFT = 0  # Set by train.py at save time",
        f"SILHOUETTE_SOFT = {sil_soft_val}"
    )
    vae_sil_val = int(os.environ.get("USE_VAE_SILHOUETTE", "0"))
    template_src = template_src.replace(
        "USE_VAE_SILHOUETTE = 0  # Set by train.py at save time",
        f"USE_VAE_SILHOUETTE = {vae_sil_val}"
    )
    agn_inp_soft_val = float(os.environ.get("AGNOSTIC_INPAINT_SOFT_SIG", "0.0"))
    template_src = template_src.replace(
        "AGNOSTIC_INPAINT_SOFT_SIG = 0.0  # Set by train.py at save time",
        f"AGNOSTIC_INPAINT_SOFT_SIG = {agn_inp_soft_val}"
    )
    agn_zero_rep_val = int(os.environ.get("AGNOSTIC_ZERO_REPAIR", "0"))
    template_src = template_src.replace(
        "AGNOSTIC_ZERO_REPAIR = 0  # Set by train.py at save time",
        f"AGNOSTIC_ZERO_REPAIR = {agn_zero_rep_val}"
    )
    use_repair_mask_val = int(os.environ.get("USE_REPAIR_ATTN_MASK", "0"))
    template_src = template_src.replace(
        "USE_REPAIR_ATTN_MASK = 0  # Set by train.py at save time",
        f"USE_REPAIR_ATTN_MASK = {use_repair_mask_val}"
    )
    bg_hint_val = int(os.environ.get("USE_BG_HINT", "0"))
    template_src = template_src.replace(
        "USE_BG_HINT = 0  # Set by train.py at save time",
        f"USE_BG_HINT = {bg_hint_val}"
    )
    bg_hint_scale_val = float(os.environ.get("BG_HINT_SCALE", "1.0"))
    template_src = template_src.replace(
        "BG_HINT_SCALE = 1.0  # Set by train.py at save time",
        f"BG_HINT_SCALE = {bg_hint_scale_val}"
    )
    v6_zero_g_val = int(os.environ.get("V6_ZERO_G_CORE", "0")) if int(os.environ.get("USE_V6", "0")) else 0
    template_src = template_src.replace(
        "V6_ZERO_G_CORE = 0  # Set by train.py at save time",
        f"V6_ZERO_G_CORE = {v6_zero_g_val}"
    )
    v6_r_in_val = int(os.environ.get("V6_R_IN", "2"))
    template_src = template_src.replace(
        "V6_R_IN = 2  # Set by train.py at save time",
        f"V6_R_IN = {v6_r_in_val}"
    )
    v6_r_out_val = int(os.environ.get("V6_R_OUT", "7"))
    template_src = template_src.replace(
        "V6_R_OUT = 7  # Set by train.py at save time",
        f"V6_R_OUT = {v6_r_out_val}"
    )
    sil_soft_sig_val = float(os.environ.get("SILHOUETTE_SOFT_SIG", "2.0"))
    template_src = template_src.replace(
        "SILHOUETTE_SOFT_SIG = 2.0  # Set by train.py at save time",
        f"SILHOUETTE_SOFT_SIG = {sil_soft_sig_val}"
    )
    rough_mask_soft_val = int(os.environ.get("ROUGH_MASK_SOFT", "0"))
    template_src = template_src.replace(
        "ROUGH_MASK_SOFT = 0  # Set by train.py at save time",
        f"ROUGH_MASK_SOFT = {rough_mask_soft_val}"
    )
    rough_mask_soft_sig_val = float(os.environ.get("ROUGH_MASK_SOFT_SIG", "3.0"))
    template_src = template_src.replace(
        "ROUGH_MASK_SOFT_SIG = 3.0  # Set by train.py at save time",
        f"ROUGH_MASK_SOFT_SIG = {rough_mask_soft_sig_val}"
    )
    rough_blur_fixed_val = int(os.environ.get("USE_ROUGH_BLUR_FIXED", "0"))
    rough_blur_sig_val   = float(os.environ.get("ROUGH_BLUR_FIXED_SIG", "4.0"))
    template_src = template_src.replace(
        "USE_ROUGH_BLUR_FIXED = 0  # Set by train.py at save time",
        f"USE_ROUGH_BLUR_FIXED = {rough_blur_fixed_val}"
    )
    template_src = template_src.replace(
        "ROUGH_BLUR_FIXED_SIG = 4.0  # Set by train.py at save time",
        f"ROUGH_BLUR_FIXED_SIG = {rough_blur_sig_val}"
    )
    with open(os.path.join(final_path, "inference.py"), "w") as f:
        f.write(template_src)

    # Copy the loss comparison / experiment plan doc into the run dir if present
    loss_doc = os.path.join(BASE, "vtonautoresearch", "loss_comparison_exp395_to_exp409.md")
    if os.path.exists(loss_doc):
        shutil.copy2(loss_doc, os.path.join(final_path, "loss_comparison.md"))

    elapsed = time.time() - t_start
    print(f"\n---\ntraining_seconds: {elapsed:.0f}\nnum_steps: {global_step}\noutput_dir: {args.output_dir}")
    log.info("Done.")


if __name__ == "__main__":
    main()
