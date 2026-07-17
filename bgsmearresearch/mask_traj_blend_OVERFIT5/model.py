"""mask_traj_blend.py — MASKED TRAJECTORY BLENDING (user's #1 design).

A separate garment velocity denoiser predicts garment-region velocity; frozen run37 owns
everything else. At EVERY denoise step (training AND deploy rollout):
    v_final = v_run37 * (1 - M) + v_garment * M
where M = warped garment mask at latent/token res. In the garment region v_final = v_garment;
elsewhere v_final = v_run37 (bg/skin untouched -> gradient can't reach them). run37 is a FROZEN
constant (detached); only the garment denoiser trains, and only through the M region.

Clean diagnostic: if garment can't improve while bg/skin are LOCKED by construction, the problem
is the garment branch not learning detail, not contamination.

Guarded by USE_MASK_TRAJ_BLEND (default off -> all other runs byte-identical).
"""
import os
import math
import glob
import torch
import torch.nn as nn
from trainlib.models import _make_qwen_block
from trainlib.data import pack_latents, unpack_latents
import trainlib.state as state

BASE = "/home/link/Desktop/Code/fashion gen testing"
LAT = f"{BASE}/my_vton_cache/latents"


def timestep_embedding(t, dim):
    """Sinusoidal timestep/σ embedding -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GarmentVelocityDenoiser(nn.Module):
    """Copied Qwen blocks over [C_t(16) + garment_latent(16) + warped_mask(1) + warped_rgb(3)] = 36ch
    + σ conditioning -> garment-region velocity (16ch). Velocity head ZERO-INIT so v_garment=0 at
    step0 (v_final == v_run37 exactly), then it learns."""
    def __init__(self, n_blocks=4, dim=3072, heads=24, head_dim=128, in_ch=36):
        super().__init__()
        self.dim = dim
        self.in_ch = in_ch
        self.blocks = nn.ModuleList([_make_qwen_block(dim, heads, head_dim) for _ in range(n_blocks)])
        self.patch_proj = nn.Linear(in_ch * 4, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 * 48, dim)); nn.init.normal_(self.pos_embed, std=0.01)
        self.dummy_text = nn.Parameter(torch.zeros(1, 8, dim)); nn.init.normal_(self.dummy_text, std=0.02)
        self.temb_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.vel_head = nn.Linear(dim, 16 * 4)
        nn.init.zeros_(self.vel_head.weight); nn.init.zeros_(self.vel_head.bias)   # zero-init -> v_garment=0 at init

    def forward(self, C_t, garment_latent, warped_mask, warped_rgb, sigma):
        dt = self.patch_proj.weight.dtype
        C_t = C_t.to(dt); garment_latent = garment_latent.to(dt)
        wm = warped_mask.to(dt)
        if wm.dim() == 3: wm = wm.unsqueeze(1)
        wr = warped_rgb.to(dt)
        B, _, H, W = C_t.shape
        x_in = torch.cat([C_t, garment_latent, wm, wr], dim=1)                     # (B, 36, H, W)
        x = x_in.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, (H // 2) * (W // 2), self.in_ch * 4)
        x = self.patch_proj(x) + self.pos_embed[:, : (H // 2) * (W // 2)]
        _sig = sigma.reshape(-1)
        _sig = _sig[:1].expand(B) if _sig.numel() == 1 else _sig[:B]
        temb = self.temb_mlp(timestep_embedding(_sig, self.dim).to(dt))
        text = self.dummy_text.expand(B, -1, -1).to(dt)
        text_mask = torch.ones(B, text.shape[1], dtype=torch.long, device=x.device)
        for blk in self.blocks:
            text, x = blk(x, text, text_mask, temb, image_rotary_emb=None)
        v = self.vel_head(x)                                                       # (B, 3072, 64)
        return unpack_latents(v, B, 16, H, W)                                      # (B, 16, H, W)


_WRONG_LOG = False


def _load_wrong(batch, gar, wr, device, dtype):
    """Deploy WRONG: swap garment_latent + warped_rgb to a different reserved id (deterministic)."""
    global _WRONG_LOG
    sid = batch.get("image_id")
    if isinstance(sid, (list, tuple)):
        sid = sid[0]
    pool = [x.strip() for x in os.environ.get("BRIDGE_WRONG_SIDS", "").split(",") if x.strip()]
    cand = [x for x in pool if x != sid] or pool
    if not cand:
        return gar, wr
    wsid = cand[sum(ord(c) for c in str(sid)) % len(cand)]
    try:
        wg = torch.load(f"{LAT}/{wsid}_garment_latent.pt", map_location="cpu", weights_only=True).to(device, dtype)
        if wg.dim() == 3: wg = wg.unsqueeze(0)
        wg = wg.expand(gar.shape[0], -1, -1, -1)
        nwr = wr
        _p = f"{LAT}/{wsid}_warped_rgb_128.pt"
        if os.path.exists(_p):
            nwr = (torch.load(_p, map_location="cpu", weights_only=True).float() * 2.0 - 1.0).to(device, dtype)
            if nwr.dim() == 3: nwr = nwr.unsqueeze(0)
            nwr = nwr.expand(wr.shape[0], -1, -1, -1)
        if not _WRONG_LOG:
            print(f"[mask_traj] DEPLOY WRONG garment {sid} <- {wsid}", flush=True); _WRONG_LOG = True
        return wg, nwr
    except Exception as e:
        print(f"[mask_traj] wrong-load failed: {e}", flush=True)
        return gar, wr


_DBG_DONE = False


def blend_velocity(v_run37_p, C_t, sigma, batch, B, C, H, W, device, dtype):
    """v_final = v_run37 * (1-M) + v_garment * M  (packed token space). run37 detached (frozen)."""
    global _DBG_DONE
    # two-track x_base pass -> return PURE run37 (v_final == v_run37).
    if getattr(state, "_MASK_TRAJ_BYPASS", False):
        return v_run37_p
    den = state._GARMENT_VEL_DENOISER
    gar = batch["garment_latent"].to(device, dtype)
    wm = batch["warped_mask"].to(device, dtype)
    if wm.dim() == 3: wm = wm.unsqueeze(1)
    wr = batch.get("warped_garment_rgb")
    wr = wr.to(device, dtype) if wr is not None else torch.zeros(B, 3, H, W, device=device, dtype=dtype)
    if int(os.environ.get("MASK_TRAJ_DEPLOY_WRONG", "0")):                         # WRONG: different garment
        gar, wr = _load_wrong(batch, gar, wr, device, dtype)
    if int(os.environ.get("MASK_TRAJ_ZERO", "0")):                                 # ZERO: denoiser runs on EMPTY garment
        gar = torch.zeros_like(gar); wr = torch.zeros_like(wr)                     # (keep C_t/mask/σ) -> "no-garment" velocity
    v_gar = den(C_t, gar, wm, wr, sigma)                                           # (B,16,H,W) fp32
    v_gar_p = pack_latents(v_gar, B, C, H, W).to(v_run37_p.dtype)                  # (B,3072,64)
    wm_bin = (wm > 0.5).to(dtype).expand(B, C, H, W)
    M_tok = pack_latents(wm_bin, B, C, H, W).mean(-1, keepdim=True).to(v_run37_p.dtype)   # (B,3072,1) soft
    v_final = v_run37_p.detach() * (1.0 - M_tok) + v_gar_p * M_tok
    if int(os.environ.get("MASK_TRAJ_DEBUG", "0")) and not _DBG_DONE:
        with torch.no_grad():
            bg = (M_tok[..., 0] < 1e-4)                                            # tokens fully outside garment
            fg = (M_tok[..., 0] > 0.5)
            d_bg = float((v_final - v_run37_p)[bg].abs().max()) if bg.any() else -1.0
            d_fg = float((v_final - v_run37_p)[fg].abs().mean()) if fg.any() else -1.0
            print(f"[mask_traj_dbg] M-blend: bg-token max|v_final-v_run37|={d_bg:.3e} (MUST be 0 -> bg locked) | "
                  f"fg-token mean|Δ|={d_fg:.3e} (grows as denoiser learns) | M_tok.sum={float(M_tok.sum()):.1f} "
                  f"v_gar.absmean={float(v_gar.detach().abs().mean()):.3e}", flush=True)
        _DBG_DONE = True
    return v_final
