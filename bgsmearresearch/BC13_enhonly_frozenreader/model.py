"""garment_adapter.py — UNIFIED garment adapter for tonight's 6 runs.

One module, `GARMENT_ADAPTER_MODE` selects the injection style. All modes take the CLEAN aligned
garment signal (warped_rgb 3 + garment_latent 16 + warped_mask 1 = 20ch, person-coords) and inject a
ZERO-INIT, mask-gated contribution so step0 == run37 exactly. NO K/V-append, NO OOTD K/V.

Modes:
  input_hidden  (Run1/Run2) : block-0 pre-hook, C_hidden += M * zero_init_proj([warped_rgb, garment, mask])  (3072-d hidden).
  latent_token  (Run3)      : add M * zero_init_64d_adapter(...) to the packed C-slot (64-d) BEFORE the transformer.
  detail_slot   (Run4)      : add a learned residual detail slot to the standard garment conditioning SLOT (64-d tokens).
  spatial_adaln (Run5)      : per-token FiLM at blocks 12,20,28,36: hidden = hidden*(1+M*gamma) + M*beta (zero-init gamma/beta).
  controlnet    (Run6)      : Qwen side branch -> masked zero-init residuals into blocks 8,16,24,32.

Branch-off (eval + '10% base solo'): state._GARMENT_ADAPTER_BYPASS=True -> every hook returns run37 unchanged.
Schedule (state._GAR_SCHED['branch_on']) also gates the contribution during training.
"""
import os
import torch
import torch.nn as nn
from copy import deepcopy
from trainlib.models import _make_qwen_block
from trainlib.data import pack_latents, unpack_latents
import trainlib.state as state

ADAPTER_IN_CH = 20   # warped_rgb(3) + garment_latent(16) + warped_mask(1)


_LAT = "/home/link/Desktop/Code/fashion gen testing/my_vton_cache/latents"
_WRONG_LOGGED = False


def _load_wrong_adapter(batch, gar, wr, device, dtype):
    """Eval WRONG: swap garment_latent + warped_rgb fed to the adapter to a DIFFERENT reserved id."""
    global _WRONG_LOGGED
    sid = batch.get("image_id")
    if isinstance(sid, (list, tuple)):
        sid = sid[0]
    pool = [x.strip() for x in os.environ.get("BRIDGE_WRONG_SIDS", "").split(",") if x.strip()]
    cand = [x for x in pool if x != sid] or pool
    if not cand:
        return gar, wr
    wsid = cand[sum(ord(c) for c in str(sid)) % len(cand)]
    try:
        wg = torch.load(f"{_LAT}/{wsid}_garment_latent.pt", map_location="cpu", weights_only=True).to(device, dtype)
        if wg.dim() == 3: wg = wg.unsqueeze(0)
        gar = wg.expand(gar.shape[0], -1, -1, -1)
        _p = f"{_LAT}/{wsid}_warped_rgb_128.pt"
        if os.path.exists(_p):
            nwr = (torch.load(_p, map_location="cpu", weights_only=True).float() * 2.0 - 1.0).to(device, dtype)
            if nwr.dim() == 3: nwr = nwr.unsqueeze(0)
            wr = nwr.expand(wr.shape[0], -1, -1, -1)
        if not _WRONG_LOGGED:
            print(f"[garment_adapter] EVAL WRONG garment {sid} <- {wsid}", flush=True); _WRONG_LOGGED = True
    except Exception as e:
        print(f"[garment_adapter] wrong-load failed: {e}", flush=True)
    return gar, wr


def build_adapter_input(batch, device, dtype, B, H, W):
    """Clean aligned garment signal (B, 20, H, W): warped_rgb(3) + garment_latent(16) + warped_mask(1)."""
    gar = batch["garment_latent"].to(device, dtype)
    wr = batch.get("warped_garment_rgb")
    wr = wr.to(device, dtype) if wr is not None else torch.zeros(B, 3, H, W, device=device, dtype=dtype)
    if int(os.environ.get("GARMENT_ADAPTER_DEPLOY_WRONG", "0")):
        gar, wr = _load_wrong_adapter(batch, gar, wr, device, dtype)
    wm = batch["warped_mask"].to(device, dtype)
    if wm.dim() == 3: wm = wm.unsqueeze(1)
    return torch.cat([wr, gar, (wm > 0.5).to(dtype)], dim=1)


def mask_tok(batch, device, dtype, B, C, H, W):
    """Per-token warped-garment gate (B, N, 1) — the M in every injection."""
    wm = batch["warped_mask"].to(device, dtype)
    if wm.dim() == 3: wm = wm.unsqueeze(1)
    wm = (wm > 0.5).to(dtype).expand(B, C, H, W)
    return pack_latents(wm, B, C, H, W).mean(dim=-1, keepdim=True)


def _active():
    """True when the adapter should contribute (not bypassed by eval or the base-solo schedule)."""
    if getattr(state, "_GARMENT_ADAPTER_BYPASS", False) or int(os.environ.get("GARMENT_ADAPTER_BYPASS", "0")):
        return False   # eval branch-off / base-solo -> hooks return run37's OWN output (NOT adapter=0)
    sch = getattr(state, "_GAR_SCHED", None)
    if sch is not None and not sch.get("branch_on", True):
        return False
    return True


class GarmentAdapter(nn.Module):
    def __init__(self, mode="input_hidden", dim=3072, heads=24, head_dim=128, in_ch=ADAPTER_IN_CH,
                 pack_dim=64, n_gnet_blocks=2, cn_blocks=None):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.in_ch = in_ch
        self.pack_dim = pack_dim
        if mode in ("input_hidden",):
            self.proj = nn.Linear(in_ch * 4, dim)
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        elif mode == "latent_token":
            self.proj = nn.Linear(in_ch * 4, pack_dim)              # -> 64-d packed residual
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        elif mode == "detail_slot":
            self.slot = nn.Linear(in_ch * 4, pack_dim)              # learned residual detail slot tokens (64-d)
            nn.init.zeros_(self.slot.weight); nn.init.zeros_(self.slot.bias)
        elif mode in ("spatial_adaln", "controlnet"):
            # small Qwen garment encoder producing per-token features
            self.patch = nn.Linear(in_ch * 4, dim)
            self.pos = nn.Parameter(torch.zeros(1, 64 * 48, dim)); nn.init.normal_(self.pos, std=0.01)
            self.temb = nn.Parameter(torch.zeros(1, dim)); nn.init.trunc_normal_(self.temb, std=0.02)
            self.dummy_text = nn.Parameter(torch.zeros(1, 8, dim)); nn.init.normal_(self.dummy_text, std=0.02)
            self.blocks = nn.ModuleList([_make_qwen_block(dim, heads, head_dim) for _ in range(n_gnet_blocks)])
            if mode == "spatial_adaln":
                tgt = cn_blocks or [12, 20, 28, 36]
                self.gamma = nn.ModuleDict({str(b): nn.Linear(dim, dim) for b in tgt})
                self.beta = nn.ModuleDict({str(b): nn.Linear(dim, dim) for b in tgt})
                for m in list(self.gamma.values()) + list(self.beta.values()):
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)     # zero FiLM at init
            else:  # controlnet
                tgt = cn_blocks or [8, 16, 24, 32]
                self.res = nn.ModuleDict({str(b): nn.Linear(dim, dim) for b in tgt})
                for m in self.res.values():
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)     # zero residual at init
        elif mode == "state_enhancer":
            # STATE-AWARE ENHANCER (next-version RunA/B/C).
            # Delta = CrossAttn(Q=run37 hidden after block i, K=V=garment_net features);
            # H_i' = H_i + M_garment * ZeroOut(Delta). Edits what run37 is ALREADY drawing in
            # the garment region (the first N_C target/velocity tokens), NOT "predicts garment".
            self.heads = heads; self.head_dim = head_dim
            # (a) garment encoder G from [warped_rgb, garment_latent, warped_mask] -> per-token feats
            self.patch = nn.Linear(in_ch * 4, dim)
            self.pos = nn.Parameter(torch.zeros(1, 64 * 48, dim)); nn.init.normal_(self.pos, std=0.01)
            self.temb = nn.Parameter(torch.zeros(1, dim)); nn.init.trunc_normal_(self.temb, std=0.02)
            self.dummy_text = nn.Parameter(torch.zeros(1, 8, dim)); nn.init.normal_(self.dummy_text, std=0.02)
            self.blocks = nn.ModuleList([_make_qwen_block(dim, heads, head_dim) for _ in range(n_gnet_blocks)])
            # (b) cross-attention: Q = run37 hidden (garment target tokens), K/V = G
            self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, heads * head_dim, bias=False)
            self.out_proj = nn.Linear(heads * head_dim, dim)
            nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)   # zero Delta at init -> step0 == run37
            # ENH_OUT_INIT_STD>0: small NONZERO out_proj init so delta starts non-degenerate.
            # Zero-init + small-init encoder => near-zero regime where the enhancer gradient VANISHES
            # (~1e-7, can't bootstrap in a 1h overfit). A tiny nonzero init breaks the degeneracy so
            # the gradient is normal-sized; step0 != run37 by a small perturbation (fine for the credit test).
            _ois = float(os.environ.get("ENH_OUT_INIT_STD", "0.0"))
            if _ois > 0:
                nn.init.normal_(self.out_proj.weight, std=_ois); nn.init.zeros_(self.out_proj.bias)
            self.q_norm = nn.LayerNorm(head_dim)   # QK-norm: bound attention logits (stability — no NaN blowup)
            self.k_norm = nn.LayerNorm(head_dim)

    def _encode(self, garment_in):
        """Run the small garment encoder (spatial_adaln / controlnet) -> per-token features (B, N, dim)."""
        garment_in = garment_in.to(self.patch.weight.dtype)
        B, _, H, W = garment_in.shape
        x = garment_in.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, (H // 2) * (W // 2), self.in_ch * 4)
        x = self.patch(x) + self.pos[:, : (H // 2) * (W // 2)]
        text = self.dummy_text.expand(B, -1, -1).to(x.dtype)
        tmask = torch.ones(B, text.shape[1], dtype=torch.long, device=x.device)
        temb = self.temb.expand(B, -1).to(x.dtype)
        for blk in self.blocks:
            text, x = blk(x, text, tmask, temb, image_rotary_emb=None)
        return x

    # input_hidden / latent_token / detail_slot: direct projection of the patchified aligned garment
    def project(self, garment_in):
        garment_in = garment_in.to(self.proj.weight.dtype if hasattr(self, "proj") else self.slot.weight.dtype)
        B, _, H, W = garment_in.shape
        x = garment_in.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, (H // 2) * (W // 2), self.in_ch * 4)
        head = self.proj if hasattr(self, "proj") else self.slot
        return head(x)

    # state_enhancer: Delta = CrossAttn(Q=H, K=V=G); returns M * ZeroOut(Delta) (0 at init).
    def xattn(self, H, G, M):
        """H (B,N_C,dim) run37 garment target tokens; G (B,N_g,dim) garment feats; M (B,N_C,1) gate."""
        import torch.nn.functional as _F
        H = H.to(self.q_proj.weight.dtype); G = G.to(self.k_proj.weight.dtype)
        B, Nc, _ = H.shape; Ng = G.shape[1]; h, d = self.heads, self.head_dim
        q = self.q_proj(H).view(B, Nc, h, d).transpose(1, 2)   # (B,h,Nc,d)
        k = self.k_proj(G).view(B, Ng, h, d).transpose(1, 2)   # (B,h,Ng,d)
        v = self.v_proj(G).view(B, Ng, h, d).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)                 # QK-norm -> bounded logits (no attention blowup)
        a = _F.scaled_dot_product_attention(q, k, v)           # (B,h,Nc,d)
        a = a.transpose(1, 2).reshape(B, Nc, h * d)
        delta = self.out_proj(a)                               # zero-init -> 0 at start
        return (M.to(delta.dtype) * delta)


# ── hooks ────────────────────────────────────────────────────────────────────
def install_input_hidden_hook(transformer, adapter, holder):
    """Run1/2: block-0 forward_pre_hook adds M * zero_init_proj(aligned garment) to the C-slot hidden."""
    _inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    blocks = _inner.transformer_blocks if hasattr(_inner, "transformer_blocks") else _inner.blocks

    def pre_hook(module, args, kwargs):
        if not _active():
            return None
        g = holder.get("adapter_in"); M = holder.get("adapter_M"); N_C = holder.get("N_C")
        if g is None or M is None or N_C is None:
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None or hs.dim() != 3 or hs.shape[1] < N_C:
            return None
        res = adapter.project(g)                                        # (B, N_C, dim), zero at init
        add = (M * res).to(hs.dtype)
        hs_new = torch.cat([hs[:, :N_C] + add[:, :N_C], hs[:, N_C:]], dim=1)
        if kwargs.get("hidden_states", None) is not None:
            kwargs = dict(kwargs); kwargs["hidden_states"] = hs_new; return (args, kwargs)
        return ((hs_new,) + tuple(args[1:]), kwargs)

    h = blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    return [h]


def install_spatial_adaln_hooks(transformer, adapter, holder, tgt_blocks):
    """Run5: per-token FiLM at target blocks: hidden = hidden*(1+M*gamma) + M*beta."""
    _inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    blocks = _inner.transformer_blocks if hasattr(_inner, "transformer_blocks") else _inner.blocks
    handles = []

    def make(bk):
        def hook(module, inp, out):
            if not _active():
                return out
            feat = holder.get("adapter_feat"); M = holder.get("adapter_M"); N_C = holder.get("N_C")
            if feat is None or M is None or N_C is None:
                return out
            text_out, img = out if isinstance(out, tuple) and len(out) == 2 else (None, out)
            if not torch.is_tensor(img) or img.dim() != 3 or img.shape[1] < N_C:
                return out
            g = adapter.gamma[str(bk)](feat[:, :N_C]); b = adapter.beta[str(bk)](feat[:, :N_C])
            Fc = img[:, :N_C]
            Fc = Fc * (1.0 + (M * g).to(Fc.dtype)) + (M * b).to(Fc.dtype)
            img_new = torch.cat([Fc, img[:, N_C:]], dim=1)
            return (text_out, img_new) if text_out is not None else img_new
        return hook
    for bk in tgt_blocks:
        handles.append(blocks[bk].register_forward_hook(make(bk)))
    return handles


def install_state_enhancer_hooks(transformer, adapter, holder, tgt_blocks):
    """After target block(s), H_i' = H_i + M_garment * CrossAttn(Q=H_i, K=V=G).
    CRITICAL: the delta is added by WRAPPING block.forward (NOT a register_forward_hook), so the
    injection lives INSIDE the gradient-checkpointed block function. A post-block forward_hook's
    delta is severed by checkpointing recompute (enhancer gets ZERO gradient -> delta stuck at
    zero-init -> false "inert"). The wrapper is recomputed during backward exactly like the LoRA,
    so gradient reaches the enhancer. Edits run37's OWN garment-region target tokens (first N_C)."""
    _inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    blocks = _inner.transformer_blocks if hasattr(_inner, "transformer_blocks") else _inner.blocks
    handles = []

    def make_wrapper(_orig_forward):
        def wrapped(*args, **kwargs):
            out = _orig_forward(*args, **kwargs)
            if not _active():
                return out
            G = holder.get("adapter_feat"); M = holder.get("adapter_M"); N_C = holder.get("N_C")
            if G is None or M is None or N_C is None:
                return out
            text_out, img = out if isinstance(out, tuple) and len(out) == 2 else (None, out)
            if not torch.is_tensor(img) or img.dim() != 3 or img.shape[1] < N_C:
                return out
            H = img[:, :N_C]                                            # run37 garment target tokens
            delta = adapter.xattn(H, G, M)                             # (B,N_C,dim), 0 at init
            _dm = (delta.float() ** 2).mean()                         # branch-credit delta magnitude
            _prev = getattr(state, "_ENH_DELTA_SQMEAN", None)
            state._ENH_DELTA_SQMEAN = _dm if _prev is None else (_prev + _dm)
            img_new = torch.cat([H + delta.to(img.dtype), img[:, N_C:]], dim=1)
            return (text_out, img_new) if text_out is not None else img_new
        return wrapped

    class _Restore:                                                    # so run.py cleanup has .remove()
        def __init__(self, blk, orig): self.blk = blk; self.orig = orig
        def remove(self): self.blk.forward = self.orig
    for bk in tgt_blocks:
        _blk = blocks[bk]; _orig = _blk.forward
        _blk.forward = make_wrapper(_orig)
        handles.append(_Restore(_blk, _orig))
    return handles


def install_controlnet_hooks(transformer, adapter, holder, tgt_blocks):
    """Run6: masked zero-init residual into target blocks from the side-branch feature."""
    _inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    blocks = _inner.transformer_blocks if hasattr(_inner, "transformer_blocks") else _inner.blocks
    handles = []

    def make(bk):
        def hook(module, inp, out):
            if not _active():
                return out
            feat = holder.get("adapter_feat"); M = holder.get("adapter_M"); N_C = holder.get("N_C")
            if feat is None or M is None or N_C is None:
                return out
            text_out, img = out if isinstance(out, tuple) and len(out) == 2 else (None, out)
            if not torch.is_tensor(img) or img.dim() != 3 or img.shape[1] < N_C:
                return out
            r = adapter.res[str(bk)](feat[:, :N_C])
            Fc = img[:, :N_C] + (M * r).to(img.dtype)
            img_new = torch.cat([Fc, img[:, N_C:]], dim=1)
            return (text_out, img_new) if text_out is not None else img_new
        return hook
    for bk in tgt_blocks:
        handles.append(blocks[bk].register_forward_hook(make(bk)))
    return handles
