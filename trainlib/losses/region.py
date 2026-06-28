import os, sys
import torch
import torch.nn.functional as F
from trainlib import state
from trainlib.conditioning.masks import build_v6_masks
from trainlib.rollouts.halo_eval import deployed_halo_eval
from trainlib.models import (GarmentCrossAttn, GarmentLatentEnhancer, GarmentNet, GarmentNetAdaLN, GarmentNetCrossAttn, GarmentNetOutput, GarmentRepairGate, GarmentSlotEncoder, OOTDInjector, PatchDiscriminator, QwenAuxSlotEncoder, QwenControlNet, QwenGarmentEncoder, QwenGarmentNetAdaLN, QwenGarmentNetBlockCopyAdaLN, QwenGarmentNetCrossAttn, QwenLatentRefiner, QwenSlotEnricher, V6Heads, _make_qwen_block, perceptual_loss)
from trainlib.data import (pack_latents, unpack_latents, vae_decode_to_pil, precompute_rough_pils, precompute_prompt_embeds, load_pose_latents, VTONDataset, collate_fn)
from trainlib.builders import (get_vgg_features, _get_v6_heads, _v6_hidden_hook, _get_invalid_token, CrossAttnGarmentProcessor, QwenGarmentBranch, OOTDQwenAttnProcessor, _get_garment_branch, _get_qwen_slot_enricher, _get_qwen_aux_slot, _get_qwen_refiner, _get_garment_encoder, _get_garment_xattn, _get_controlnet, _make_controlnet_block_hook, _proj_out_pre_hook, _get_garment_net, _garment_inject_hook, _get_discriminator, _get_critic, HoleyAttnProcessor, AgnosticCtrlProcessor, build_repair_attn_mask, install_garment_gates)
from trainlib.constants import *


"""Per-region image reconstruction (img_g/s/b/other/k/ub), harmony, bg-ring isolation."""

def region_image_losses(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    B = ctx.get('B')
    Hi = ctx.get('Hi')
    L_flow = ctx.get('L_flow')
    L_img = ctx.get('L_img')
    L_img_b = ctx.get('L_img_b')
    L_img_core = ctx.get('L_img_core')
    L_img_g = ctx.get('L_img_g')
    L_img_k = ctx.get('L_img_k')
    L_img_keep = ctx.get('L_img_keep')
    L_img_other = ctx.get('L_img_other')
    L_img_repair = ctx.get('L_img_repair')
    L_img_s = ctx.get('L_img_s')
    L_img_ub = ctx.get('L_img_ub')
    M_b_v6 = ctx.get('M_b_v6')
    M_band = ctx.get('M_band')
    M_edit_v6 = ctx.get('M_edit_v6')
    M_full = ctx.get('M_full')
    M_g_v6 = ctx.get('M_g_v6')
    M_k_v6 = ctx.get('M_k_v6')
    M_other_v6 = ctx.get('M_other_v6')
    M_s_v6 = ctx.get('M_s_v6')
    W_band = ctx.get('W_band')
    Wi = ctx.get('Wi')
    _Hl = ctx.get('_Hl')
    _M_eroded = ctx.get('_M_eroded')
    _M_img_hard = ctx.get('_M_img_hard')
    _Riso = ctx.get('_Riso')
    _Wl = ctx.get('_Wl')
    _area = ctx.get('_area')
    _b_err = ctx.get('_b_err')
    _b_i = ctx.get('_b_i')
    _bg_field = ctx.get('_bg_field')
    _bg_l = ctx.get('_bg_l')
    _bgnp = ctx.get('_bgnp')
    _br = ctx.get('_br')
    _charb = ctx.get('_charb')
    _clean = ctx.get('_clean')
    _core_i = ctx.get('_core_i')
    _d = ctx.get('_d')
    _edit_h = ctx.get('_edit_h')
    _edit_l = ctx.get('_edit_l')
    _g_dil = ctx.get('_g_dil')
    _g_i = ctx.get('_g_i')
    _g_id = ctx.get('_g_id')
    _g_st = ctx.get('_g_st')
    _gar_l = ctx.get('_gar_l')
    _gtbg_l = ctx.get('_gtbg_l')
    _gw = ctx.get('_gw')
    _hole_i = ctx.get('_hole_i')
    _inner_ring = ctx.get('_inner_ring')
    _k_i = ctx.get('_k_i')
    _keep_i = ctx.get('_keep_i')
    _maxd = ctx.get('_maxd')
    _ml = ctx.get('_ml')
    _nb = ctx.get('_nb')
    _other_i = ctx.get('_other_i')
    _out_bg = ctx.get('_out_bg')
    _pbw = ctx.get('_pbw')
    _pbw_i = ctx.get('_pbw_i')
    _person_l = ctx.get('_person_l')
    _reg_l1 = ctx.get('_reg_l1')
    _repair_i = ctx.get('_repair_i')
    _s_i = ctx.get('_s_i')
    _sb = ctx.get('_sb')
    _scale = ctx.get('_scale')
    _skin_l = ctx.get('_skin_l')
    _sm = ctx.get('_sm')
    _ub_hi = ctx.get('_ub_hi')
    _ub_i = ctx.get('_ub_i')
    _ub_lo = ctx.get('_ub_lo')
    _up = ctx.get('_up')
    _w_bmatch = ctx.get('_w_bmatch')
    _w_harm = ctx.get('_w_harm')
    _wc = ctx.get('_wc')
    _wden = ctx.get('_wden')
    _wfp = ctx.get('_wfp')
    _wfp_alt = ctx.get('_wfp_alt')
    _white = ctx.get('_white')
    _wsuf = ctx.get('_wsuf')
    b = ctx.get('b')
    batch = ctx.get('batch')
    correction = ctx.get('correction')
    d_lat = ctx.get('d_lat')
    denom = ctx.get('denom')
    device = ctx.get('device')
    garment_prior = ctx.get('garment_prior')
    gnet = ctx.get('gnet')
    gp_imgs = ctx.get('gp_imgs')
    iid = ctx.get('iid')
    image_ids = ctx.get('image_ids')
    img_weight_map = ctx.get('img_weight_map')
    k = ctx.get('k')
    keep_mask = ctx.get('keep_mask')
    m = ctx.get('m')
    mask = ctx.get('mask')
    person = ctx.get('person')
    person_image_cache = ctx.get('person_image_cache')
    person_imgs = ctx.get('person_imgs')
    pix_err = ctx.get('pix_err')
    pred_img = ctx.get('pred_img')
    repair_band = ctx.get('repair_band')
    sigma = ctx.get('sigma')
    uncertain_band = ctx.get('uncertain_band')
    use_v6 = ctx.get('use_v6')
    vae_device = ctx.get('vae_device')
    weight_dtype = ctx.get('weight_dtype')
    weight_map_img = ctx.get('weight_map_img')
    wib = ctx.get('wib')
    wic = ctx.get('wic')
    wig = ctx.get('wig')
    wik = ctx.get('wik')
    wio = ctx.get('wio')
    wir = ctx.get('wir')
    wis = ctx.get('wis')
    wiub = ctx.get('wiub')
    wm_pix = ctx.get('wm_pix')
    wm_pix_b = ctx.get('wm_pix_b')
    wm_pix_list = ctx.get('wm_pix_list')
    wm_pix_soft = ctx.get('wm_pix_soft')
    wp = ctx.get('wp')
    x0_pred = ctx.get('x0_pred')
    # ── Output-space garment net assistance (paradigm #5) ──
    # Frozen base produces pred_img. Garment net produces image-space correction
    # gated by warped_pixel_mask (warped_fullres_mask). Garment net only sees
    # garment_pixel — never person, agnostic, etc.
    if (int(os.environ.get("USE_GARMENT_NET", "0"))
            and os.environ.get("GARMENT_NET_MODE", "norm_residual") == "output_space"
            and "garment_pixel" in batch):
        gp_imgs = batch["garment_pixel"].to(vae_device, dtype=weight_dtype)            # (B, 3, 1024, 768)
        gnet = _get_garment_net(vae_device, weight_dtype)                              # GarmentNetOutput
        correction = gnet(gp_imgs)                                                     # (B, 3, Hi, Wi), [-1, 1]ish
        # Pixel-space gate (warped fullres). Resize from cache if needed.
        wm_pix_list = []
        for iid in image_ids:
            _wsuf = os.environ.get("WARP_SUFFIX", "")
            _wfp_alt = os.path.join(BASE, "my_vton_cache/latents", f"{iid}_warped_fullres_mask{_wsuf}.pt")
            _wfp = _wfp_alt if _wsuf and os.path.exists(_wfp_alt) else \
                   os.path.join(BASE, "my_vton_cache/latents", f"{iid}_warped_fullres_mask.pt")
            wp = torch.load(_wfp, weights_only=True)
            if wp.dim() == 2: wp = wp.unsqueeze(0)
            wm_pix_list.append(wp)
        wm_pix = torch.stack(wm_pix_list).to(vae_device, dtype=weight_dtype)            # (B, 1, 1024, 768)
        if wm_pix.shape[-2:] != (Hi, Wi):
            wm_pix = F.interpolate(wm_pix, size=(Hi, Wi), mode="nearest")
        wm_pix_b = (wm_pix > 0.5).to(weight_dtype)
        # Soft-feathered gate so gradient is finite at the boundary
        from torchvision.transforms.functional import gaussian_blur as _gb_gn
        wm_pix_soft = _gb_gn(wm_pix_b.float(), kernel_size=[7, 7], sigma=2.0).to(wm_pix_b.dtype)
        pred_img = (pred_img + correction * wm_pix_soft).clamp(-1, 1)

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

    # ── L_band (LATENT): direct boundary-background latent match (USE_BAND_LOSS) ──
    # The revealed-bg ring is a LATENT generation error (VAE round-trip floor is ~50-100x
    # below the model's defect), so penalize x0_pred against the GT person_latent DIRECTLY
    # in the boundary-background band, at latent res (128x96) — NO VAE decode. This is the
    # latent replacement for the old pixel band (which was a weak, indirect teacher). A
    # direct latent L1-to-GT subsumes the old pixel/color/blotch terms at once.
    L_band = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    g_band = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    L_band_hp_diag = torch.zeros((), device=L_flow.device, dtype=torch.float32)   # retired (pixel high-pass)
    if int(os.environ.get("USE_BAND_LOSS", "0")) and all(k in batch for k in ("parse_skin", "parse_garment", "parse_bg")):
        import numpy as _np_bl
        from scipy import ndimage as _ndi_bl
        _Hl, _Wl = x0_pred.shape[-2], x0_pred.shape[-1]               # 128, 96 (latent)
        def _ml(m):                                                   # mask -> (B,1,_Hl,_Wl) on device
            m = m.to(device, torch.float32)
            if m.dim() == 3: m = m.unsqueeze(1)
            if m.shape[1] != 1: m = m[:, :1]
            if tuple(m.shape[-2:]) != (_Hl, _Wl):
                m = F.interpolate(m, size=(_Hl, _Wl), mode="nearest")
            return m
        _skin_l = _ml(batch["parse_skin"]); _gar_l = _ml(batch["parse_garment"]); _bg_l = _ml(batch["parse_bg"])
        _person_l = (_skin_l > 0.5) | (_gar_l > 0.5)                  # person silhouette (latent)
        _edit_l = _ml(M_full) > 0.5                                   # edit region (latent)
        _gtbg_l = _bg_l > 0.5                                         # GT background
        # exact distance INTO background (~person), per sample, in latent cells
        _bgnp = (~_person_l[:, 0]).detach().cpu().numpy()
        _d = _np_bl.stack([_ndi_bl.distance_transform_edt(_bgnp[b]) for b in range(B)])
        d_lat = torch.from_numpy(_d).to(device, torch.float32).unsqueeze(1)        # (B,1,_Hl,_Wl)
        _maxd = float(os.environ.get("BAND_LAT_MAXDIST", "8.0"))      # band width in latent cells
        _scale = float(os.environ.get("BAND_LAT_SCALE", "2.0"))      # boundary-weighting falloff
        M_band = ((d_lat > 0) & (d_lat <= _maxd) & _edit_l & _gtbg_l).float()
        W_band = torch.exp(-d_lat / _scale) * M_band                 # weight toward the boundary
        _area = W_band.sum() + 1e-6
        if float(W_band.sum()) > 1.0:
            # direct Charbonnier between predicted clean latent and GT latent, in the band
            _charb = torch.sqrt(((x0_pred.float() - person.float()) ** 2).mean(1, keepdim=True) + 1e-3 ** 2)
            L_band = ((_charb * W_band).sum() / _area).to(L_flow.device, torch.float32)
        # σ-gate: gaussian peak at 0.30 (the formation window) + low-σ cleanup floor
        _sb = sigma.float()
        _gw = torch.exp(-0.5 * ((_sb - 0.30) / 0.10) ** 2)
        _clean = torch.where(_sb < 0.20, torch.full_like(_sb, 0.25), torch.zeros_like(_sb))
        g_band = torch.maximum(_gw, _clean).mean().to(L_flow.device, torch.float32)

    # ── USE_IMG_REGION_SPLIT=1: per-region normalized L1 for image loss ──
    # L_img = W_IC*|pred-gt|_core + W_IR*|pred-gt|_repair + W_IB*|pred-gt|_ub
    #       + W_IK*|pred-gt|_keep   (each normalized by region pixel count)
    # In keep region, gt == source, so W_IK > 0 preserves source.
    # Low W_IR reduces repair freedom; high W_IC pushes garment fidelity.
    L_harmony = torch.tensor(0.0, device=device)
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

        # ── BG RING ISOLATION (BG_RING_ISOLATION=1): partition the background so each pixel
        #    has exactly ONE target. ring(0-Rpx around garment) is owned by gem→GT + tve only;
        #    field(>Rpx) is owned by img_b→white only. img_b excludes the ring; harmony and
        #    img_ub exclude ALL bg (so no →GT loss collides with bg→white at the silhouette). ──
        _bg_field = _b_i; _nb = None
        if int(os.environ.get("BG_RING_ISOLATION", "0")):
            _Riso = int(os.environ.get("GARMENT_EDGE_RING_PX", "8"))
            _g_dil = F.max_pool2d(_g_i, 2*_Riso+1, 1, _Riso).clamp(0, 1)   # garment dilated R px
            _bg_field = (_b_i * (1.0 - _g_dil)).clamp(0, 1)                # bg >R px from garment = img_b zone
            _nb = (1.0 - _b_i)                                             # ¬bg = exclude bg from harmony/img_ub

        L_img_g     = _reg_l1(_g_i)
        L_img_s     = _reg_l1(_s_i)
        # BG region is a DIFFERENT TASK: match the real surrounding white (copy the
        # neighbours), NOT per-pixel GT. The white = person_imgs (==agnostic) OUTSIDE the
        # hole, bg only. garment/skin still match GT (hard task); bg just copies the flat
        # neighbour colour (easy task). One loss per region, explicitly separated.
        if int(os.environ.get("BG_REGION_MATCH_WHITE", "0")) and "parse_bg" in batch:
            _hole_i = _up(M_full)
            _pbw = batch["parse_bg"].to(device, dtype=weight_dtype)
            if _pbw.dim() == 3: _pbw = _pbw.unsqueeze(1)
            _pbw_i = _up((_pbw > 0.5).float())
            _out_bg = ((1.0 - _hole_i) * _pbw_i).clamp(0, 1)                  # real bg outside hole
            if float(_out_bg.sum()) >= 16:
                _wden = _out_bg.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                _white = (person_imgs.float() * _out_bg).sum(dim=(2, 3), keepdim=True) / _wden  # (B,3,1,1)
                _b_err = (pred_img.float() - _white).abs().mean(dim=1, keepdim=True)
                L_img_b = (_b_err * _bg_field).sum() / (_bg_field.sum() + 1e-6)
                if int(os.environ.get("BG_WHITE_DEBUG", "0")):
                    _wc = ((_white[0].view(3).clamp(-1, 1) + 1) / 2 * 255).int().tolist()
                    print(f"[bg_white] white_rgb={_wc} field_px={int(_bg_field.sum())} "
                          f"full_bg_px={int(_b_i.sum())} excluded_ring_px={int(_b_i.sum()-_bg_field.sum())} "
                          f"L_img_b={float(L_img_b):.4f}", flush=True)
            else:
                L_img_b = _reg_l1(_bg_field)
        else:
            L_img_b = _reg_l1(_bg_field)
        L_img_other = _reg_l1(_other_i)
        L_img_k     = _reg_l1(_k_i)
        L_img_ub    = _reg_l1(_ub_i if _nb is None else (_ub_i * _nb).clamp(0, 1))

        wig  = float(os.environ.get("W_IMG_V6_G",     "1.0"))     # garment (high)
        wis  = float(os.environ.get("W_IMG_V6_S",     "0.5"))     # skin repair
        wib  = float(os.environ.get("W_IMG_V6_B",     "0.5"))     # bg repair
        wio  = float(os.environ.get("W_IMG_V6_OTHER", "1.0"))     # ring fallback
        wik  = float(os.environ.get("W_IMG_V6_K",     "1.0"))     # keep preserves source
        wiub = float(os.environ.get("W_IMG_V6_UB",    "2.0"))     # boundary
        # CLEAN-LOSS σ-staging (USE_V6_IMG_STAGE=1): this single unified recon is the
        # ONLY image-recon term — its region weights replace rsi/rbi/gcl/bnd/bgc/bgf.
        # garment identity emphasized LATE (low σ); boundary/silhouette emphasized
        # EARLY (high σ). Uses per-step mean σ (exact for batch_size=1). 2026-06-08.
        if int(os.environ.get("USE_V6_IMG_STAGE", "0")):
            _sm    = float(sigma.float().mean())   # python float → wig/wiub stay scalars (no device mismatch)
            _g_id  = float(os.environ.get("W_IMG_V6_G_ID",     str(wig)))   # garment @ low σ
            _g_st  = float(os.environ.get("W_IMG_V6_G_STRUCT", "0.3"))      # garment @ high σ
            wig    = _g_id * (1.0 - _sm) + _g_st * _sm
            _ub_lo = float(os.environ.get("W_IMG_V6_UB_LO",    "0.3"))      # boundary @ low σ
            _ub_hi = float(os.environ.get("W_IMG_V6_UB_HI",    str(wiub)))  # boundary @ high σ
            wiub   = _ub_lo + (_ub_hi - _ub_lo) * _sm
        L_img = (wig * L_img_g + wis * L_img_s + wib * L_img_b
               + wio * L_img_other + wik * L_img_k + wiub * L_img_ub)
        L_img = L_img.to(L_flow.device, dtype=torch.float32)
        # CALIB: weighted per-region contributions inside the unified L_img.
        state._LIMG_PARTS.update({"img_g": float(wig*L_img_g), "img_s": float(wis*L_img_s),
                            "img_b": float(wib*L_img_b), "img_other": float(wio*L_img_other),
                            "img_k": float(wik*L_img_k), "img_ub": float(wiub*L_img_ub)})
        # ── HARMONY: global L1 over the ENTIRE edit region (pred vs GT), per-pixel-balanced.
        #    Ties the independently-supervised regions into one coherent image — the per-region
        #    L1s are each normalized by their OWN area, so they lose the global balance. ──
        _w_harm = float(os.environ.get("W_HARMONY", "0.0"))
        if _w_harm > 0:
            _edit_h = _up(M_edit_v6)
            if _nb is not None: _edit_h = (_edit_h * _nb).clamp(0, 1)   # isolation: harmony excludes bg (no →GT vs bg→white)
            L_harmony = ((pix_err * _edit_h).sum() / (_edit_h.sum() + 1e-6)).to(L_flow.device, dtype=torch.float32)
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

    return {k: v for k, v in locals().items() if not k.startswith("__") and k != "ctx"}