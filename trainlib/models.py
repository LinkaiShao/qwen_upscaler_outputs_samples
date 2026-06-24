"""Pure model classes (split from train.py — no shared globals)."""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
from trainlib.data import pack_latents, unpack_latents, vae_decode_to_pil

_VGG_MODEL=None


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


class GarmentNet(nn.Module):
    """Garment-only helper encoder. Reads garment_pixel (B, 3, 1024, 768) and
    produces a residual to add at transformer.norm_out's C-slot tokens.

    Output shape: (B, N_C=3072, hidden_dim=3072) — same as the C-slot portion of
    norm_out. Final layer zero-init so initial residual ≈ 0 (no early disruption
    to the frozen tryon LoRA's predictions).

    Architecture: 4 stride-2 convs + 1 stride-1 conv → flatten → linear projection.
    1024×768 input → 64×48 spatial (3072 tokens). Each token gets a 3072-dim residual.
    """
    def __init__(self, in_ch=3, hidden_dim=3072, ch_mult=None):
        super().__init__()
        if ch_mult is None:
            size_tag = os.environ.get("GARMENT_NET_SIZE", "")
            big = int(os.environ.get("GARMENT_NET_BIG", "0"))
            if size_tag == "huge":
                ch_mult = (96, 192, 384, 768)
            elif big:
                ch_mult = (64, 128, 256, 512)
            else:
                ch_mult = (32, 64, 128, 256)
        c1, c2, c3, c4 = ch_mult
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=4, stride=2, padding=1), nn.GELU(),  # 512×384
            nn.Conv2d(c1,    c2, kernel_size=4, stride=2, padding=1), nn.GELU(),  # 256×192
            nn.Conv2d(c2,    c3, kernel_size=4, stride=2, padding=1), nn.GELU(),  # 128×96
            nn.Conv2d(c3,    c4, kernel_size=4, stride=2, padding=1), nn.GELU(),  # 64×48
            nn.Conv2d(c4,    c4, kernel_size=3, stride=1, padding=1), nn.GELU(),  # 64×48 refine
        )
        self.proj = nn.Linear(c4, hidden_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, garment_pixel):
        # garment_pixel: (B, 3, 1024, 768) in [0, 1] float. Center to [-1, 1].
        x = garment_pixel * 2.0 - 1.0
        h = self.enc(x)                                      # (B, c4, 64, 48)
        B, Ch, H, W = h.shape
        h = h.flatten(2).transpose(1, 2).contiguous()        # (B, 3072, c4)
        out = self.proj(h)                                   # (B, 3072, hidden_dim)
        return out


class GarmentNetOutput(nn.Module):
    """Output-space helper. Reads garment_pixel (B, 3, 1024, 768) and produces
    a full-resolution image-space correction (B, 3, 1024, 768). The correction
    is applied to the decoded pred image, gated by warped_pixel_mask, AFTER the
    frozen base produces its prediction. This avoids any latent-space perturbation
    of the frozen backbone (no VAE receptive-field spillage).

    Architecture: encoder-decoder (no skip connections to keep weight count modest).
    Final ConvTranspose zero-init → initial correction = 0 (no early disruption).
    """
    def __init__(self, in_ch=3, ch_mult=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = ch_mult
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),  # 512×384
            nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),  # 256×192
            nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),  # 128×96
            nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),  # 64×48
            nn.Conv2d(c4,    c4, 3, 1, 1), nn.GELU(),  # bottleneck
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1), nn.GELU(),  # 128×96
            nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.GELU(),  # 256×192
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.GELU(),  # 512×384
            nn.ConvTranspose2d(c1, in_ch, 4, 2, 1),          # 1024×768
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, garment_pixel):
        x = garment_pixel * 2.0 - 1.0
        return self.dec(self.enc(x))   # (B, 3, 1024, 768) — initial correction ≈ 0


class GarmentLatentEnhancer(nn.Module):
    """Adds a learnable residual to garment_latent itself. CNN maps
    garment_pixel (B, 3, 1024, 768) → latent-shaped delta (B, 16, 128, 96)
    that is added to the garment_latent before it enters slot 3.

    This stays compatible with the frozen base because the base was already
    trained to consume garment_latent in this slot — we just provide a
    slightly enhanced version. No new slots, no token-count changes,
    no perturbation of non-garment regions.

    Final 1×1 conv zero-init → initial delta = 0 (no early disruption).
    """
    def __init__(self, in_ch=3, ch_mult=(32, 64, 128, 256), out_ch=16):
        super().__init__()
        c1, c2, c3, c4 = ch_mult
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),  # 512×384
            nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),  # 256×192
            nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),  # 128×96 (matches latent res)
            nn.Conv2d(c3,    c4, 3, 1, 1), nn.GELU(),
        )
        self.out = nn.Conv2d(c4, out_ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, garment_pixel):
        x = garment_pixel * 2.0 - 1.0
        return self.out(self.enc(x))   # (B, 16, 128, 96) — initial delta ≈ 0


class GarmentNetAdaLN(nn.Module):
    """AdaLN modulation garment net (paradigm #2). Garment-only encoder produces
    per-token (γ, β) modulation pairs at hidden_dim. Applied at transformer.norm_out:

        modulated = (1 + γ · gate) · normed + β · gate

    where `gate` is the packed warped_mask (binary 0/1 per token), so modulation
    is exactly 0 outside the garment silhouette → ZERO bleeding into repair area.

    γ, β init to 0 (no early disruption — modulation starts as identity).
    """
    def __init__(self, in_ch=3, hidden_dim=3072, ch_mult=None):
        super().__init__()
        if ch_mult is None:
            size_tag = os.environ.get("GARMENT_NET_SIZE", "")
            if size_tag == "huge":
                ch_mult = (96, 192, 384, 768)
            elif int(os.environ.get("GARMENT_NET_BIG", "0")):
                ch_mult = (64, 128, 256, 512)
            else:
                ch_mult = (32, 64, 128, 256)
        c1, c2, c3, c4 = ch_mult
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),  # 512×384
            nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),  # 256×192
            nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),  # 128×96
            nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),  # 64×48
            nn.Conv2d(c4,    c4, 3, 1, 1), nn.GELU(),  # 64×48 refine
        )
        # Two zero-init heads: γ and β
        self.to_gamma = nn.Linear(c4, hidden_dim, bias=True)
        self.to_beta  = nn.Linear(c4, hidden_dim, bias=True)
        for h in (self.to_gamma, self.to_beta):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, garment_pixel):
        x = garment_pixel * 2.0 - 1.0
        h = self.enc(x)                                       # (B, c4, 64, 48)
        B, Ch, H, W = h.shape
        h = h.flatten(2).transpose(1, 2).contiguous()         # (B, 3072, c4)
        gamma = self.to_gamma(h)                              # (B, 3072, hidden_dim)
        beta  = self.to_beta(h)
        return gamma, beta


class GarmentNetCrossAttn(nn.Module):
    """Cross-attention K,V injection garment net (paradigm #1, IP-Adapter style).
    Encoder produces N_g tokens at inner_dim. Shared to_k_g / to_v_g project
    these tokens to K, V that every transformer-block's attention processor
    uses for an EXTRA cross-attention computation, gated by warped_mask
    packed to image-token level so only inside-garment image queries get
    garment context (no bleeding into repair area).
    """
    def __init__(self, in_ch=3, inner_dim=3072, ch_mult=None):
        super().__init__()
        if ch_mult is None:
            size_tag = os.environ.get("GARMENT_NET_SIZE", "")
            if size_tag == "huge":
                ch_mult = (96, 192, 384, 768)
            elif int(os.environ.get("GARMENT_NET_BIG", "0")):
                ch_mult = (64, 128, 256, 512)
            else:
                ch_mult = (32, 64, 128, 256)
        c1, c2, c3, c4 = ch_mult
        # 1024×768 → 16×12 (192 tokens) via 6 stride-2 convs
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GELU(),  # 512×384
            nn.Conv2d(c1,    c2, 4, 2, 1), nn.GELU(),  # 256×192
            nn.Conv2d(c2,    c3, 4, 2, 1), nn.GELU(),  # 128×96
            nn.Conv2d(c3,    c4, 4, 2, 1), nn.GELU(),  # 64×48
            nn.Conv2d(c4,    c4, 4, 2, 1), nn.GELU(),  # 32×24
            nn.Conv2d(c4,    c4, 4, 2, 1), nn.GELU(),  # 16×12 → 192 tokens
        )
        self.tok_proj = nn.Linear(c4, inner_dim)
        # Shared global to_k_g, to_v_g used by every attention block.
        self.to_k_g = nn.Linear(inner_dim, inner_dim, bias=True)
        self.to_v_g = nn.Linear(inner_dim, inner_dim, bias=True)
        # Zero-init V projection so initial cross-attn contribution = 0.
        nn.init.zeros_(self.to_v_g.weight)
        nn.init.zeros_(self.to_v_g.bias)

    def forward(self, garment_pixel):
        x = garment_pixel * 2.0 - 1.0
        h = self.enc(x)                                       # (B, c4, 16, 12)
        h = h.flatten(2).transpose(1, 2).contiguous()         # (B, 192, c4)
        tokens = self.tok_proj(h)                              # (B, 192, inner_dim)
        K_g = self.to_k_g(tokens)                              # (B, 192, inner_dim)
        V_g = self.to_v_g(tokens)                              # (B, 192, inner_dim)
        return K_g, V_g


class GarmentSlotEncoder(nn.Module):
    """Produces a packed-slot tensor (B, 3072, 64) — same shape as agnostic/garment/silhouette
    slots — from garment_pixel. Appended to the conditioning sequence as a new slot
    'garment_aux'. The frozen LoRA's joint attention will compute Q@K^T over all
    tokens including the new ones, so the existing image tokens can attend to a
    fresh, garment-specialised representation.

    Final 1×1 conv zero-init → initial slot tokens ≈ 0 so the model behaves
    indistinguishably from the frozen base at step 0.
    """
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
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, garment_pixel):
        x = garment_pixel * 2.0 - 1.0
        h = self.out(self.enc(x))                                # (B, 16, 128, 96)
        # Pack 2×2 patches → (B, 3072, 64), matches latent slot format
        B, Ch, H, W = h.shape
        return h.view(B, Ch, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), Ch*4)


def _make_qwen_block(dim, num_heads, head_dim):
    """Real diffusers QwenImageTransformerBlock — joint text+image stream w/ AdaLN modulation."""
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    return QwenImageTransformerBlock(
        dim=dim, num_attention_heads=num_heads, attention_head_dim=head_dim,
    )


class QwenGarmentNetAdaLN(nn.Module):
    """Qwen-backbone garment encoder using diffusers' QwenImageTransformerBlock.

    Input: precomputed `garment_latent` (B, 16, 128, 96).
    2×2 packs into (B, 3072, 64). Linear projects to enc_dim. N joint-stream Qwen
    blocks (with a learnable dummy text stream + learnable temb). Final γ, β
    heads zero-init for identity start.
    """
    def __init__(self, in_lat_ch=16, hidden_dim=3072, enc_dim=None, n_layers=None, num_heads=None, head_dim=None):
        if n_layers is None:
            n_layers = int(os.environ.get("GARMENT_NET_QWEN_LAYERS", "2"))
        super().__init__()
        big = int(os.environ.get("GARMENT_NET_BIG", "0"))
        if enc_dim is None:
            enc_dim = 1024 if big else 512
        if num_heads is None:
            num_heads = 16 if enc_dim >= 1024 else 8
        if head_dim is None:
            head_dim = enc_dim // num_heads
        assert num_heads * head_dim == enc_dim, f"heads*head_dim must equal enc_dim ({num_heads}*{head_dim} != {enc_dim})"
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Dummy text stream + learnable temb so the joint-stream block has valid inputs.
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.dummy_text, std=0.02)
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        self.to_gamma = nn.Linear(enc_dim, hidden_dim, bias=True)
        self.to_beta  = nn.Linear(enc_dim, hidden_dim, bias=True)
        for h in (self.to_gamma, self.to_beta):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        gamma = self.to_gamma(x)
        beta  = self.to_beta(x)
        return gamma, beta


class QwenGarmentNetBlockCopyAdaLN(nn.Module):
    """Block-copy variant — N QwenImageTransformerBlock instances at the SAME dim/heads
    as the main 60-block transformer (3072/24/128). Initialized from main-model
    block weights via load_blocks_from_main() after construction; then trained
    independently. β-only AdaLN head, zero-init.

    Input: garment_latent (B, 16, 128, 96).
    """
    def __init__(self, in_lat_ch=16, hidden_dim=3072, n_layers=None):
        if n_layers is None:
            n_layers = int(os.environ.get("GARMENT_NET_QWEN_LAYERS", "2"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128  # match main model
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.dummy_text, std=0.02)
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        self.to_gamma = nn.Linear(enc_dim, hidden_dim, bias=True)
        self.to_beta  = nn.Linear(enc_dim, hidden_dim, bias=True)
        for h in (self.to_gamma, self.to_beta):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def load_blocks_from_main(self, main_transformer, indices):
        """Copy state_dict from main_transformer.transformer_blocks[indices[i]] into self.blocks[i]."""
        assert len(indices) == len(self.blocks), \
            f"need {len(self.blocks)} indices, got {len(indices)}"
        for tgt_i, src_i in enumerate(indices):
            sd = main_transformer.transformer_blocks[src_i].state_dict()
            self.blocks[tgt_i].load_state_dict(sd, strict=True)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        gamma = self.to_gamma(x)
        beta  = self.to_beta(x)
        return gamma, beta


class QwenGarmentNetCrossAttn(nn.Module):
    """Qwen-backbone garment encoder (cross_attn variant) using diffusers' QwenImageTransformerBlock."""
    def __init__(self, in_lat_ch=16, inner_dim=3072, enc_dim=None, n_layers=None, num_heads=None, head_dim=None, n_tokens=192):
        if n_layers is None:
            n_layers = int(os.environ.get("GARMENT_NET_QWEN_LAYERS", "2"))
        super().__init__()
        big = int(os.environ.get("GARMENT_NET_BIG", "0"))
        if enc_dim is None:
            enc_dim = 1024 if big else 512
        if num_heads is None:
            num_heads = 16 if enc_dim >= 1024 else 8
        if head_dim is None:
            head_dim = enc_dim // num_heads
        assert num_heads * head_dim == enc_dim
        self.enc_dim = enc_dim
        self.n_tokens = n_tokens
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.dummy_text, std=0.02)
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        self.pool = nn.Linear(64 * 48, n_tokens, bias=True)
        self.to_k_g = nn.Linear(enc_dim, inner_dim, bias=True)
        self.to_v_g = nn.Linear(enc_dim, inner_dim, bias=True)
        nn.init.zeros_(self.to_v_g.weight); nn.init.zeros_(self.to_v_g.bias)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        K_g = self.to_k_g(x)
        V_g = self.to_v_g(x)
        return K_g, V_g


class QwenGarmentEncoder(nn.Module):
    """Block-copy garment encoder, feature-only output (no β/γ heads).
    Designed for the cross-attn injection path."""
    def __init__(self, in_lat_ch=16, n_layers=None):
        if n_layers is None:
            n_layers = int(os.environ.get("GARMENT_XATTN_LAYERS", "2"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128  # match main model
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.dummy_text, std=0.02)
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks), \
            f"need {len(self.blocks)} indices, got {len(indices)}"
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        return self.out_norm(x)   # (B, 3072, 3072)


class QwenControlNet(nn.Module):
    """ControlNet branch for Qwen-Image-Edit transformer.

    Standard ControlNet pattern: N blocks copied from main, each producing a
    zero-init residual added to the corresponding main transformer block's
    hidden_states (via forward_hook). The residual is added ONLY to the C-slot
    tokens (the denoising target slot) inside main's concatenated hidden_states.

    Input:  agnostic_latent (B, 16, 128, 96)
    Output: list of N residuals, each (B, 64*48, hidden_dim) — applied to C-slot.

    Hooks should be installed externally on main.transformer_blocks[copy_to_blocks[i]]
    (defaults to blocks 0..N-1 → 1:1 mapping).
    """
    def __init__(self, in_lat_ch=16, hidden_dim=3072, n_layers=None):
        if n_layers is None:
            n_layers = int(os.environ.get("CONTROLNET_LAYERS", "12"))
        super().__init__()
        self.n_layers = n_layers
        enc_dim, num_heads, head_dim = hidden_dim, 24, 128  # match main
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.dummy_text, std=0.02)
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        # Per-block zero-init residual projection. Same dim out so we can add
        # directly to main's hidden_states without further reshaping.
        self.zero_projs = nn.ModuleList([
            nn.Linear(enc_dim, hidden_dim, bias=True) for _ in range(n_layers)
        ])
        for proj in self.zero_projs:
            nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks), \
            f"need {len(self.blocks)} indices, got {len(indices)}"
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def forward(self, agnostic_latent):
        """Returns list of N residuals, each (B, 64*48, hidden_dim).

        Caller is responsible for adding each residual to the C-slot region of
        the corresponding main block's hidden_states output (via forward_hook).
        """
        B, C, H, W = agnostic_latent.shape
        x = agnostic_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        residuals = []
        for blk, zproj in zip(self.blocks, self.zero_projs):
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
            residuals.append(zproj(x))  # zero-init at start
        return residuals


class GarmentCrossAttn(nn.Module):
    """Cross-attention F → G. Q from main hidden F, K,V from garment encoder G.
    Zero-init out_proj + learnable scalar gate (fp32, init logit=-3 → sigmoid≈0.047,
    small but non-zero for clean gradient signal early)."""
    def __init__(self, dim=3072, num_heads=24, head_dim=128, gate_init_logit=-3.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.Wq = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.Wk = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.Wv = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # FP32 gate to avoid bf16 quantization of small Adam updates
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit), dtype=torch.float32))

    def forward(self, F_h, G):
        B, N_F, _ = F_h.shape
        N_G = G.shape[1]
        Q = self.Wq(F_h).view(B, N_F, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(G  ).view(B, N_G, self.num_heads, self.head_dim).transpose(1, 2)
        A = F.scaled_dot_product_attention(Q, K, V)
        A = A.transpose(1, 2).reshape(B, N_F, self.num_heads * self.head_dim)
        out = self.out_proj(A)
        # GARMENT_XATTN_FORCE_GATE_1: skip sigmoid clamp, use 1.0 directly so
        # the scalar gate doesn't suppress the branch (5_29/instructions5).
        if int(os.environ.get("GARMENT_XATTN_FORCE_GATE_1", "0")):
            gate = torch.tensor(1.0, dtype=out.dtype, device=out.device)
        else:
            gate = torch.sigmoid(self.gate_logit).to(out.dtype)
        return gate * out


class OOTDInjector(nn.Module):
    """Per-injection-block module holding dedicated to_k_g, to_v_g and a learnable
    scalar gate. Init: to_k_g/to_v_g copied from the main block's BASE attn weights
    (frozen-LoRA-aware copy uses base_layer); gate logit init=-5 → sigmoid≈0.007 so
    initial garment contribution is near zero (no early disruption to v6b baseline)."""
    def __init__(self, dim=3072, num_heads=24, head_dim=128, gate_init_logit=-4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_k_g = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.to_v_g = nn.Linear(dim, num_heads * head_dim, bias=True)
        # Per-block learned scalar gate (fp32 to avoid bf16 quantization of small
        # parameters). sigmoid(-4) ≈ 0.018: low initial influence, learns to amplify.
        self.gate_logit = nn.Parameter(
            torch.tensor(float(gate_init_logit), dtype=torch.float32))

    def init_from_main_block(self, main_block):
        """Copy effective weights (base + frozen LoRA) from main block i's attn.to_k/to_v."""
        attn = main_block.attn
        # Effective Linear weight for a peft-wrapped Linear: W_base + sum scaling*(B@A)
        def _effective_w_b(linear_mod):
            base = linear_mod.base_layer if hasattr(linear_mod, "base_layer") else linear_mod
            W = base.weight.data.detach().clone()
            b = base.bias.data.detach().clone() if base.bias is not None else None
            if hasattr(linear_mod, "lora_A") and hasattr(linear_mod, "lora_B"):
                for adapter_name in linear_mod.lora_A.keys():
                    A = linear_mod.lora_A[adapter_name].weight.data
                    B = linear_mod.lora_B[adapter_name].weight.data
                    scaling = linear_mod.scaling[adapter_name] if hasattr(linear_mod, "scaling") else 1.0
                    W = W + scaling * (B @ A).to(W.dtype)
            return W, b

        Wk, bk = _effective_w_b(attn.to_k)
        Wv, bv = _effective_w_b(attn.to_v)
        self.to_k_g.weight.data.copy_(Wk.to(self.to_k_g.weight.dtype))
        if bk is not None and self.to_k_g.bias is not None:
            self.to_k_g.bias.data.copy_(bk.to(self.to_k_g.bias.dtype))
        # GARMENT_OOTD_VG_ZERO_INIT=1: zero V projection so contribution starts at 0
        # regardless of gate value. Lets gate operate freely (e.g., init logit=0 → 0.5)
        # without the model first having to find a non-tiny signal through a near-zero gate.
        if int(os.environ.get("GARMENT_OOTD_VG_ZERO_INIT", "0")):
            nn.init.zeros_(self.to_v_g.weight)
            if self.to_v_g.bias is not None:
                nn.init.zeros_(self.to_v_g.bias)
        else:
            self.to_v_g.weight.data.copy_(Wv.to(self.to_v_g.weight.dtype))
            if bv is not None and self.to_v_g.bias is not None:
                self.to_v_g.bias.data.copy_(bv.to(self.to_v_g.bias.dtype))


class QwenSlotEnricher(nn.Module):
    """Garment encoder (block-copy from main) + projection back to 64-dim slot.
    Output is a residual added to the packed garment slot. Zero-init projection
    keeps initial residual = 0 so baseline is preserved exactly at start."""
    def __init__(self, in_lat_ch=16, n_layers=None, slot_dim=64):
        if n_layers is None:
            n_layers = int(os.environ.get("QWEN_SLOT_ENRICH_LAYERS", "4"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        # Project back to slot dim (per-token 64-dim packed-latent representation)
        self.to_slot_residual = nn.Linear(enc_dim, slot_dim, bias=True)
        # Zero-init so initial residual = 0 (baseline preserved exactly)
        nn.init.zeros_(self.to_slot_residual.weight)
        nn.init.zeros_(self.to_slot_residual.bias)

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks)
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        return self.to_slot_residual(x)   # (B, 3072, 64) — slot-dim residual


class QwenAuxSlotEncoder(nn.Module):
    def __init__(self, in_lat_ch=16, n_layers=None, slot_dim=64):
        if n_layers is None:
            n_layers = int(os.environ.get("QWEN_AUX_SLOT_LAYERS", "4"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128
        self.enc_dim = enc_dim
        self.patch_proj = nn.Linear(in_lat_ch * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        self.to_slot = nn.Linear(enc_dim, slot_dim, bias=True)
        nn.init.zeros_(self.to_slot.weight)
        nn.init.zeros_(self.to_slot.bias)

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks)
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def forward(self, garment_latent):
        B, C, H, W = garment_latent.shape
        x = garment_latent.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, C * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        return self.to_slot(x)   # (B, 3072, 64) — slot tokens


class QwenLatentRefiner(nn.Module):
    def __init__(self, n_layers=None):
        if n_layers is None:
            n_layers = int(os.environ.get("QWEN_REFINER_LAYERS", "4"))
        super().__init__()
        enc_dim, num_heads, head_dim = 3072, 24, 128
        # Input: 33 channels (pred 16 + gar 16 + mask 1) × 4 (2x2 patch) = 132
        self.patch_proj = nn.Linear(33 * 4, enc_dim, bias=True)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 * 48, enc_dim))
        self.dummy_text = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.temb = nn.Parameter(torch.zeros(1, enc_dim))
        self.blocks = nn.ModuleList([
            _make_qwen_block(enc_dim, num_heads, head_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.RMSNorm(enc_dim)
        self.to_residual = nn.Linear(enc_dim, 64, bias=True)
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def load_blocks_from_main(self, main_xfmr, indices):
        assert len(indices) == len(self.blocks)
        for tgt_i, src_i in enumerate(indices):
            self.blocks[tgt_i].load_state_dict(
                main_xfmr.transformer_blocks[src_i].state_dict(), strict=True)

    def forward(self, pred_lat, garment_lat, warped_mask):
        # All inputs (B, *, 128, 96)
        B, _, H, W = pred_lat.shape
        if warped_mask.dim() == 3: warped_mask = warped_mask.unsqueeze(1)
        # Concat along channel dim → (B, 33, 128, 96)
        x_in = torch.cat([pred_lat, garment_lat, warped_mask], dim=1)
        # 2x2 patchify → (B, 64, 48, 33, 2, 2)
        x = x_in.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 64 * 48, 33 * 4)
        x = self.patch_proj(x) + self.pos_embed
        text = self.dummy_text.expand(B, -1, -1).contiguous()
        text_mask = torch.ones(B, 1, dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).contiguous()
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        x = self.out_norm(x)
        residual_packed = self.to_residual(x)              # (B, 3072, 64)
        return residual_packed                              # caller unpacks


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
