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


"""Gate priors, visible-bg mask, garment-edge bg-ring (gem), inner-bg anchor (brs), bg-route-self."""

def background_losses(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    B = ctx.get('B')
    C = ctx.get('C')
    H = ctx.get('H')
    Hi = ctx.get('Hi')
    L_bg_route_self = ctx.get('L_bg_route_self')
    L_boundary_l1 = ctx.get('L_boundary_l1')
    L_flow = ctx.get('L_flow')
    L_garment_crop_l1 = ctx.get('L_garment_crop_l1')
    L_percep = ctx.get('L_percep')
    L_repair_bg_img = ctx.get('L_repair_bg_img')
    L_repair_skin_img = ctx.get('L_repair_skin_img')
    M_full = ctx.get('M_full')
    W = ctx.get('W')
    Wi = ctx.get('Wi')
    _M_bg_pix = ctx.get('_M_bg_pix')
    _M_g_lat = ctx.get('_M_g_lat')
    _M_g_pix = ctx.get('_M_g_pix')
    _M_model_pix = ctx.get('_M_model_pix')
    _M_skin_pix = ctx.get('_M_skin_pix')
    _Mg_d = ctx.get('_Mg_d')
    _Mg_e = ctx.get('_Mg_e')
    _Mg_f = ctx.get('_Mg_f')
    _R_gem = ctx.get('_R_gem')
    _agn = ctx.get('_agn')
    _apply_detail_gate = ctx.get('_apply_detail_gate')
    _band = ctx.get('_band')
    _band_mask = ctx.get('_band_mask')
    _base = ctx.get('_base')
    _bd = ctx.get('_bd')
    _be_band = ctx.get('_be_band')
    _be_n = ctx.get('_be_n')
    _bg_n = ctx.get('_bg_n')
    _bgf_diff = ctx.get('_bgf_diff')
    _body = ctx.get('_body')
    _boundary = ctx.get('_boundary')
    _brs_diff = ctx.get('_brs_diff')
    _col = ctx.get('_col')
    _crop_band = ctx.get('_crop_band')
    _d = ctx.get('_d')
    _den = ctx.get('_den')
    _detail_gate_per_sample = ctx.get('_detail_gate_per_sample')
    _detail_gate_pix = ctx.get('_detail_gate_pix')
    _detail_sigma_max = ctx.get('_detail_sigma_max')
    _diff = ctx.get('_diff')
    _eps = ctx.get('_eps')
    _ew = ctx.get('_ew')
    _field_l = ctx.get('_field_l')
    _gar_bg_ring = ctx.get('_gar_bg_ring')
    _gated = ctx.get('_gated')
    _gd = ctx.get('_gd')
    _ge_diff = ctx.get('_ge_diff')
    _gmag = ctx.get('_gmag')
    _gmean = ctx.get('_gmean')
    _hd = ctx.get('_hd')
    _hi = ctx.get('_hi')
    _i = ctx.get('_i')
    _img = ctx.get('_img')
    _k = ctx.get('_k')
    _lam_bel1 = ctx.get('_lam_bel1')
    _lam_belg = ctx.get('_lam_belg')
    _lambda_bg_chroma = ctx.get('_lambda_bg_chroma')
    _lambda_bg_field = ctx.get('_lambda_bg_field')
    _lambda_bnd = ctx.get('_lambda_bnd')
    _lambda_gcl1 = ctx.get('_lambda_gcl1')
    _lambda_rb = ctx.get('_lambda_rb')
    _lambda_rs = ctx.get('_lambda_rs')
    _m = ctx.get('_m')
    _minv = ctx.get('_minv')
    _mm = ctx.get('_mm')
    _msk = ctx.get('_msk')
    _n = ctx.get('_n')
    _near = ctx.get('_near')
    _num = ctx.get('_num')
    _o = ctx.get('_o')
    _outside = ctx.get('_outside')
    _ov = ctx.get('_ov')
    _pa = ctx.get('_pa')
    _panel = ctx.get('_panel')
    _pb = ctx.get('_pb')
    _pb_bgr = ctx.get('_pb_bgr')
    _pb_g = ctx.get('_pb_g')
    _pb_pix = ctx.get('_pb_pix')
    _pbb = ctx.get('_pbb')
    _pbg = ctx.get('_pbg')
    _pbg_lat = ctx.get('_pbg_lat')
    _pred_bg_mean = ctx.get('_pred_bg_mean')
    _ps_loss = ctx.get('_ps_loss')
    _ps_pix = ctx.get('_ps_pix')
    _rb = ctx.get('_rb')
    _realbg_sel = ctx.get('_realbg_sel')
    _route_bg = ctx.get('_route_bg')
    _safe_bce = ctx.get('_safe_bce')
    _sb = ctx.get('_sb')
    _sc = ctx.get('_sc')
    _sg = ctx.get('_sg')
    _src = ctx.get('_src')
    _sw = ctx.get('_sw')
    _target_bg_field = ctx.get('_target_bg_field')
    _target_bg_mean = ctx.get('_target_bg_mean')
    _tc = ctx.get('_tc')
    _tgt = ctx.get('_tgt')
    _tgt_mask = ctx.get('_tgt_mask')
    _tgt_n = ctx.get('_tgt_n')
    _use_band = ctx.get('_use_band')
    _vis_bg = ctx.get('_vis_bg')
    _w_brs = ctx.get('_w_brs')
    _w_gem = ctx.get('_w_gem')
    _wm_dil = ctx.get('_wm_dil')
    _wm_for_loss = ctx.get('_wm_for_loss')
    a = ctx.get('a')
    alpha_prior = ctx.get('alpha_prior')
    batch = ctx.get('batch')
    device = ctx.get('device')
    f = ctx.get('f')
    garment_prior = ctx.get('garment_prior')
    gate = ctx.get('gate')
    gate_alphas = ctx.get('gate_alphas')
    gp_tok = ctx.get('gp_tok')
    gx = ctx.get('gx')
    gy = ctx.get('gy')
    keep_mask = ctx.get('keep_mask')
    keep_tok = ctx.get('keep_tok')
    lambda_hf_g = ctx.get('lambda_hf_g')
    lambda_percep = ctx.get('lambda_percep')
    p = ctx.get('p')
    person_imgs = ctx.get('person_imgs')
    pred = ctx.get('pred')
    pred_img = ctx.get('pred_img')
    route_logits = ctx.get('route_logits')
    sigma = ctx.get('sigma')
    sigma_mean_s = ctx.get('sigma_mean_s')
    skin_term = ctx.get('skin_term')
    t = ctx.get('t')
    target = ctx.get('target')
    texture_w = ctx.get('texture_w')
    transformer = ctx.get('transformer')
    vae_device = ctx.get('vae_device')
    vgg = ctx.get('vgg')
    weight_dtype = ctx.get('weight_dtype')
    x = ctx.get('x')
    x0_pred = ctx.get('x0_pred')
    # ── Gate losses (exp420: weak prior + entropy on α) ──
    L_gate_prior = torch.tensor(0.0, device=device)
    L_gate_entropy = torch.tensor(0.0, device=device)
    # Collect alphas from all installed GarmentRepairGate modules
    gate_alphas = []
    if state.GARMENT_GATES:
        for gate in state.GARMENT_GATES:
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
    lambda_band = float(os.environ.get("LAMBDA_BAND", "0.5"))   # grey/off-white ring loss; ramp to 1.0 if ring persists

    # SIGMA_WEIGHTED_TEXTURE=1: at high sigma (noisy) the texture losses are noise —
    # the model is mid-denoising and fine textures aren't yet meaningful. Scale
    # texture losses by (1 - mean_sigma): high σ → ~0 weight, low σ → full weight.
    # Structure losses (flow, img, recon, late_shell) stay at constant weight.
    if int(os.environ.get("SIGMA_WEIGHTED_TEXTURE", "0")):
        sigma_mean_s = float(sigma.float().mean().item())
        texture_w = max(0.0, 1.0 - sigma_mean_s)  # ramp 1→0 as σ goes 0→1
        lambda_hf_g_eff   = lambda_hf_g   * texture_w
        lambda_percep_eff = lambda_percep * texture_w
    else:
        lambda_hf_g_eff   = lambda_hf_g
        lambda_percep_eff = lambda_percep

    # ── 5_29 instructions17/3 — wire 5 inert env vars ─────────────────
    # NEW losses added at uniform-σ or σ-gated depending on DETAIL_LOSS_SIGMA_MAX.

    # Build pixel-resolution masks needed for the new losses.
    # M_model_pix = dilated M_full (the model-owned region per DILATE_M_FULL).
    # We reuse the (already-dilated) M_full.
    _M_model_pix = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
    # M_garment_pix from warped_mask
    _M_g_pix = None
    try:
        _wm_for_loss = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_for_loss.dim() == 3: _wm_for_loss = _wm_for_loss.unsqueeze(1)
        _M_g_lat = (_wm_for_loss > 0.5).to(weight_dtype)
        _M_g_pix = F.interpolate(_M_g_lat.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
    except Exception:
        _M_g_pix = None
    # M_repair_skin_pix from parse_skin ∩ M_model ∩ ~M_garment
    _M_skin_pix = None
    if "parse_skin" in batch and _M_g_pix is not None:
        _ps_loss = batch["parse_skin"].to(device, dtype=weight_dtype)
        if _ps_loss.dim() == 3: _ps_loss = _ps_loss.unsqueeze(1)
        _ps_pix = F.interpolate((_ps_loss > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _M_skin_pix = (_M_model_pix * _ps_pix * (1 - _M_g_pix)).clamp(0, 1)
    # M_repair_bg_pix = M_model − M_garment − M_repair_skin
    _M_bg_pix = None
    if _M_g_pix is not None:
        skin_term = _M_skin_pix if _M_skin_pix is not None else torch.zeros_like(_M_model_pix)
        _M_bg_pix = (_M_model_pix * (1 - _M_g_pix) * (1 - skin_term)).clamp(0, 1)
        # Restrict to ACTUAL background: edit−garment−skin still contains torso/pants/non-skin
        # body, which must NOT be pushed toward bg color. Intersect parse_bg (GT bg, train-time).
        if "parse_bg" in batch:
            _pb_bgr = batch["parse_bg"].to(device, dtype=weight_dtype)
            if _pb_bgr.dim() == 3: _pb_bgr = _pb_bgr.unsqueeze(1)
            _pb_bgr = F.interpolate((_pb_bgr > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
            _M_bg_pix = (_M_bg_pix * _pb_bgr).clamp(0, 1)

    # σ-gate for detail losses: fires when σ < DETAIL_LOSS_SIGMA_MAX
    _detail_sigma_max = float(os.environ.get("DETAIL_LOSS_SIGMA_MAX", "1.0"))
    if _detail_sigma_max < 1.0:
        _detail_gate_per_sample = (sigma.float().view(B, 1, 1, 1) < _detail_sigma_max).to(weight_dtype)
        # Single scalar per sample → broadcast to pixel-grid weight
        _detail_gate_pix = _detail_gate_per_sample.to(vae_device, weight_dtype)
    else:
        _detail_gate_pix = None

    def _apply_detail_gate(t):
        if _detail_gate_pix is None: return t
        return t * _detail_gate_pix

    # LAMBDA_GARMENT_CROP_L1 — image-space L1 around garment bbox
    L_garment_crop_l1 = torch.tensor(0.0, device=device)
    _lambda_gcl1 = float(os.environ.get("LAMBDA_GARMENT_CROP_L1", "0.0"))
    if _lambda_gcl1 > 0 and _M_g_pix is not None:
        # Compute per-sample garment bbox from M_g_pix and use it as a crop weight
        # (soft: dilate the garment mask by 32 px to include immediate boundary)
        _crop_band = F.max_pool2d(_M_g_pix.float(), kernel_size=33, stride=1, padding=16).clamp(0, 1)
        _diff = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
        _gated = _apply_detail_gate(_diff * _crop_band)
        L_garment_crop_l1 = (_gated.sum() / (_apply_detail_gate(_crop_band).sum() + 1e-6))
        L_garment_crop_l1 = L_garment_crop_l1.to(L_flow.device, dtype=torch.float32)

    # PERCEPTUAL_REGION=garment override — recompute L_percep masked to garment if requested
    # The standard L_percep was computed above (line 3691) with weight_map_img;
    # if PERCEPTUAL_REGION=garment, mask the perceptual loss to garment region.
    if (int(os.environ.get("USE_PERCEPTUAL", "0"))
            and os.environ.get("PERCEPTUAL_REGION", "") == "garment"
            and _M_g_pix is not None):
        vgg = get_vgg_features(vae_device, weight_dtype)
        # Use only the garment region for perceptual; keep single channel so
        # perceptual_loss can broadcast against VGG feature dims.
        L_percep = perceptual_loss(pred_img, person_imgs, _M_g_pix, vgg)
        L_percep = L_percep.to(L_flow.device, dtype=torch.float32)

    # LAMBDA_REPAIR_SKIN_IMG — image-space L1 in repair_skin region
    L_repair_skin_img = torch.tensor(0.0, device=device)
    _lambda_rs = float(os.environ.get("LAMBDA_REPAIR_SKIN_IMG", "0.0"))
    if _lambda_rs > 0 and _M_skin_pix is not None:
        _diff = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
        L_repair_skin_img = ((_diff * _M_skin_pix).sum()
                             / (_M_skin_pix.sum() + 1e-6))
        L_repair_skin_img = L_repair_skin_img.to(L_flow.device, dtype=torch.float32)

    # LAMBDA_REPAIR_BG_IMG — image-space L1 in repair_bg region
    L_repair_bg_img = torch.tensor(0.0, device=device)
    _lambda_rb = float(os.environ.get("LAMBDA_REPAIR_BG_IMG", "0.0"))
    if _lambda_rb > 0 and _M_bg_pix is not None:
        _diff = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
        L_repair_bg_img = ((_diff * _M_bg_pix).sum()
                           / (_M_bg_pix.sum() + 1e-6))
        L_repair_bg_img = L_repair_bg_img.to(L_flow.device, dtype=torch.float32)

    # ── Visible-background mask (parse_bg ∩ OUTSIDE the edit region) ──
    # The REAL background the model should match. The old LAMBDA_BG_CHROMA used the
    # whole keep region (1 - M_model), which includes body/hair/clothing and biased the
    # bg target 7-48 L* too dark + warm (smudge root-cause diagnostics, 2026-06-08:
    # diagnostics/smudge_root_cause_REPORT.md). person_imgs == real pixels outside the
    # hole, so it is the correct anchor source (agnostic_pixel cache was removed).
    from torchvision.transforms.functional import gaussian_blur as _bg_gb
    _eps = 1e-6
    _vis_bg = None
    if _M_bg_pix is not None and "parse_bg" in batch:
        _pb = batch["parse_bg"].to(device, dtype=weight_dtype)
        if _pb.dim() == 3: _pb = _pb.unsqueeze(1)
        _pb_pix = F.interpolate((_pb > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _vis_bg = (_pb_pix * (1 - _M_model_pix)).clamp(0, 1)
    elif _M_bg_pix is not None and _M_g_pix is not None:
        # Fallback when parse_bg is unavailable: keep region minus garment.
        _vis_bg = ((1 - _M_model_pix) * (1 - _M_g_pix)).clamp(0, 1)

    # FIX 1 — corrected LAMBDA_BG_CHROMA: anchor pred[M_bg].mean() to the mean of the
    # VISIBLE BACKGROUND, restricted to a band near M_repair_bg (local/vignette-aware).
    L_bg_chroma = torch.tensor(0.0, device=device)
    _lambda_bg_chroma = float(os.environ.get("LAMBDA_BG_CHROMA", "0.0"))
    if _lambda_bg_chroma > 0 and _M_bg_pix is not None and _vis_bg is not None:
        _band = int(os.environ.get("BG_TARGET_BAND", "80"))   # image px around repair_bg
        _near = F.max_pool2d(_M_bg_pix.float(), 2*_band+1, 1, _band).clamp(0, 1)
        _band_mask = (_vis_bg * _near).clamp(0, 1)
        # per-sample: use the local band if it holds enough visible-bg px, else full visible bg
        _use_band = (_band_mask.sum(dim=(-2,-1), keepdim=True) >= 100).to(weight_dtype)
        _tgt_mask = _use_band * _band_mask + (1 - _use_band) * _vis_bg
        _tgt_n = _tgt_mask.sum(dim=(-2,-1), keepdim=True).clamp(min=_eps)
        _bg_n  = _M_bg_pix.sum(dim=(-2,-1), keepdim=True).clamp(min=_eps)
        _target_bg_mean = (person_imgs.float() * _tgt_mask).sum(dim=(-2,-1), keepdim=True) / _tgt_n
        _pred_bg_mean   = (pred_img.float()    * _M_bg_pix).sum(dim=(-2,-1), keepdim=True) / _bg_n
        L_bg_chroma = (_pred_bg_mean - _target_bg_mean).abs().mean().to(L_flow.device, dtype=torch.float32)

    # FIX 2 — LAMBDA_BG_FIELD: per-pixel local background-field loss. Build a smooth bg
    # target by normalized large-blur inpainting of the visible bg into the edit region,
    # then L1 the prediction to it inside M_repair_bg. Captures the vignette/gradient a
    # single mean cannot. Applied broadly across sigma (NO detail-σ gate).
    L_bg_field = torch.tensor(0.0, device=device)
    _lambda_bg_field = float(os.environ.get("LAMBDA_BG_FIELD", "0.0"))
    if _lambda_bg_field > 0 and _M_bg_pix is not None and _vis_bg is not None:
        with torch.no_grad():
            _sc = 4
            _src = F.interpolate(person_imgs.float() * _vis_bg, scale_factor=1.0/_sc, mode="area")
            _msk = F.interpolate(_vis_bg,                       scale_factor=1.0/_sc, mode="area")
            _k = int(os.environ.get("BG_FIELD_KSIZE", "61")); _sg = float(os.environ.get("BG_FIELD_SIGMA", "24"))
            if _k % 2 == 0: _k += 1
            _num = _bg_gb(_src, kernel_size=[_k, _k], sigma=_sg)
            _den = _bg_gb(_msk, kernel_size=[_k, _k], sigma=_sg)
            _field_l = _num / (_den + _eps)
            _gmean = (person_imgs.float() * _vis_bg).sum(dim=(-2,-1), keepdim=True) / _vis_bg.sum(dim=(-2,-1), keepdim=True).clamp(min=_eps)
            _field_l = torch.where(_den > 1e-3, _field_l, _gmean.expand_as(_field_l))
            _target_bg_field = F.interpolate(_field_l, size=(Hi, Wi), mode="bilinear", align_corners=False)
        _bgf_diff = (pred_img.float() - _target_bg_field).abs().mean(dim=1, keepdim=True)
        L_bg_field = ((_bgf_diff * _M_bg_pix).sum() / (_M_bg_pix.sum() + _eps)).to(L_flow.device, dtype=torch.float32)

    # ── Garment-edge bg-ring match (W_GARMENT_EDGE_MATCH) — THE VISIBLE EDGE-RING HALO ──
    # MEASURED: the halo is the BACKGROUND pixels within R px of the GARMENT silhouette
    # (warped_mask); 0% overlap with the agnostic boundary (which W_BOUNDARY_MATCH targeted,
    # and which is a no-op in the v6 region-split path anyway). Force pred==GT(=real bg) in
    # that garment-edge bg ring — direct, fast halo removal, works in the v6 path.
    L_garment_edge = torch.tensor(0.0, device=device)
    _w_gem = float(os.environ.get("W_GARMENT_EDGE_MATCH", "0.0"))
    if _w_gem > 0 and _M_g_pix is not None and "parse_bg" in batch:
        _R_gem = int(os.environ.get("GARMENT_EDGE_RING_PX", "15"))
        _wm_dil = F.max_pool2d(_M_g_pix, 2*_R_gem+1, 1, _R_gem).clamp(0, 1)
        _pb_g = batch["parse_bg"].to(device, dtype=weight_dtype)
        if _pb_g.dim() == 3: _pb_g = _pb_g.unsqueeze(1)
        _pb_g = F.interpolate((_pb_g > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _gar_bg_ring = ((_wm_dil - _M_g_pix).clamp(0, 1) * _pb_g * _M_model_pix).clamp(0, 1)
        _ge_diff = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
        L_garment_edge = ((_ge_diff * _gar_bg_ring).sum() / (_gar_bg_ring.sum() + _eps)).to(L_flow.device, dtype=torch.float32)

    # ── INNER-BG anchor (W_BG_ROUTE_SELF) — the "completely different" punishment for the
    #    bg region INSIDE the agnostic hole. The inner generated bg (the halo at the garment
    #    edge) is a DIFFERENT TASK from the rest: it is the only generated bg that survives
    #    paste-back, and the model's own version of it is garbage. So:
    #      • LOCATE it with the v6 route head (bg class @~99% rec/prec, no GT cheat; runs on
    #        hidden states, deploys identically) -> route_bg, which lives inside M_model.
    #      • ANCHOR it to the REAL surrounding background taken from the AGNOSTIC: person_imgs
    #        OUTSIDE the hole == the real kept/paste-back pixels (forward.py:2456), restricted
    #        to parse_bg so it is pure studio-white, not body. NOT the model's own bg, NOT the
    #        GT of the in-hole pixels (that was gem, which regresses).
    #    Pull the inner route-bg toward that real surrounding white => the halo matches its
    #    real neighbours => seam/halo vanish; the inner-bg task is punished on its own terms.
    L_bg_route_self = torch.tensor(0.0, device=device)
    _w_brs = float(os.environ.get("W_BG_ROUTE_SELF", "0.0"))
    if _w_brs > 0 and route_logits is not None and _M_model_pix is not None:
        with torch.no_grad():
            # inner bg to fix = route head's bg, inside the hole (the halo)
            _pbg_lat = torch.softmax(route_logits.float(), dim=1)[:, 2:3]                 # (B,1,H,W) latent
            _pbg = F.interpolate(_pbg_lat, size=(Hi, Wi), mode="bilinear", align_corners=False).to(vae_device)
            _route_bg = ((_pbg > 0.5).float() * _M_model_pix).clamp(0, 1)
            # real surrounding bg color from the AGNOSTIC = person_imgs OUTSIDE the hole,
            # pure-bg only (parse_bg). This is the kept/paste-back studio white.
            _outside = (1.0 - _M_model_pix)
            _realbg_sel = _outside
            if "parse_bg" in batch:
                _pbb = batch["parse_bg"].to(device, dtype=weight_dtype)
                if _pbb.dim() == 3: _pbb = _pbb.unsqueeze(1)
                _pbb = F.interpolate((_pbb > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
                _realbg_sel = (_outside * _pbb).clamp(0, 1)
            if float(_realbg_sel.sum()) < 16: _realbg_sel = _outside
            _den = _realbg_sel.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)               # (B,1,1,1)
            _tgt = (person_imgs.float() * _realbg_sel).sum(dim=(2, 3), keepdim=True) / _den   # (B,3,1,1) real white
        _brs_diff = (pred_img.float() - _tgt).abs().mean(dim=1, keepdim=True)             # (B,1,Hi,Wi)
        L_bg_route_self = ((_brs_diff * _route_bg).sum() / (_route_bg.sum() + _eps)).to(L_flow.device, dtype=torch.float32)

        if int(os.environ.get("BG_ROUTE_DEBUG", "0")):
            with torch.no_grad():
                import numpy as _np
                from PIL import Image as _Im
                _i = 0
                _img = ((pred_img[_i].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
                _agn = ((person_imgs[_i].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
                def _ov(_base, _m, _col):
                    _o = _base.astype("float32").copy()
                    _mm = _m[_i, 0].float().cpu().numpy()[..., None]
                    return (_o * (1 - 0.6 * _mm) + _np.array(_col, "float32") * 0.6 * _mm).clip(0, 255).astype("uint8")
                _rb = _ov(_img, _route_bg, [255, 0, 0])        # red = inner bg punished (route, in-hole)
                _sb = _ov(_agn, _realbg_sel, [0, 255, 0])      # green = REAL surrounding bg sampled (on agnostic/GT)
                _hd = (_brs_diff[_i, 0] * _route_bg[_i, 0]).cpu().numpy()
                _hd = (_hd / (_hd.max() + 1e-6) * 255).astype("uint8")
                _hd = _np.stack([_hd, _np.zeros_like(_hd), 255 - _hd], -1)   # penalty heatmap
                _tc = _tgt[_i].view(3).cpu().numpy()
                _tc = ((_tc.clip(-1, 1) + 1) / 2 * 255).astype("uint8")
                _sw = _np.tile(_tc, (_img.shape[0], 80, 1)).astype("uint8")
                _panel = _np.concatenate([_img, _rb, _sb, _hd, _sw], axis=1)
                import os as _os2
                _os2.makedirs("/tmp/brs_debug", exist_ok=True)
                _n = len([f for f in _os2.listdir("/tmp/brs_debug") if f.endswith(".png")])
                _Im.fromarray(_panel).save(f"/tmp/brs_debug/brs_{_n:03d}.png")
                print(f"[brs_debug] saved brs_{_n:03d}.png  inner_route_bg_px={int(_route_bg.sum())} "
                      f"real_bg_px={int(_realbg_sel.sum())} tgt_rgb={_tc.tolist()} L={float(L_bg_route_self):.4f}", flush=True)

    # BODY-EDGE losses (Phase-11 body-edge diagnostic). The skin/garment<->background
    # BODY silhouette is under-rendered (soft, ~2x GT edge width). L1 tolerates a soft
    # edge; LAMBDA_BODY_EDGE_GRAD penalises gradient softness directly. Band = the body
    # silhouette (boundary of person = ¬parse_bg) inside M_model.
    L_body_edge_l1   = torch.tensor(0.0, device=device)
    L_body_edge_grad = torch.tensor(0.0, device=device)
    _lam_bel1 = float(os.environ.get("LAMBDA_BODY_EDGE_L1", "0.0"))
    _lam_belg = float(os.environ.get("LAMBDA_BODY_EDGE_GRAD", "0.0"))
    if (_lam_bel1 > 0 or _lam_belg > 0) and "parse_bg" in batch and _M_model_pix is not None:
        _pbb = batch["parse_bg"].to(device, dtype=weight_dtype)
        if _pbb.dim() == 3: _pbb = _pbb.unsqueeze(1)
        _pbb = F.interpolate((_pbb > 0.5).float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        _body = (1.0 - _pbb).clamp(0, 1)                                    # person region
        _ew = int(os.environ.get("BODY_EDGE_W", "6"))
        _bd = (F.max_pool2d(_body, 2*_ew+1, 1, _ew)
               + F.max_pool2d(-(_body - 1.0), 2*_ew+1, 1, _ew) - 1.0).clamp(0, 1)   # dilate∪(1-erode)
        _be_band = (_bd * _M_model_pix).clamp(0, 1)                          # body edge ∩ edit region
        _be_n = _be_band.sum() + _eps
        if _lam_bel1 > 0:
            _d = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
            L_body_edge_l1 = ((_d * _be_band).sum() / _be_n).to(L_flow.device, dtype=torch.float32)
        if _lam_belg > 0:
            def _gmag(x):
                x = x.float().mean(dim=1, keepdim=True)
                gx = x[..., :, 1:] - x[..., :, :-1]; gy = x[..., 1:, :] - x[..., :-1, :]
                gx = F.pad(gx, (0, 1, 0, 0)); gy = F.pad(gy, (0, 0, 0, 1))
                return (gx.abs() + gy.abs())
            _gd = (_gmag(pred_img) - _gmag(person_imgs)).abs()              # missing edge energy
            L_body_edge_grad = ((_gd * _be_band).sum() / _be_n).to(L_flow.device, dtype=torch.float32)

    # 5_30: LAMBDA_BOUNDARY_L1 — explicit boundary band L1 (dilate(g,15) - erode(g,15)).
    L_boundary_l1 = torch.tensor(0.0, device=device)
    _lambda_bnd = float(os.environ.get("LAMBDA_BOUNDARY_L1", "0.0"))
    if _lambda_bnd > 0 and _M_g_pix is not None:
        # Approximate dilate(15)-erode(15) using max_pool / -max_pool of -mask at pixel res
        _k = 31  # 2*15+1
        _Mg_f = _M_g_pix.float()
        _Mg_d = F.max_pool2d(_Mg_f, kernel_size=_k, stride=1, padding=_k//2)
        _Mg_e = -F.max_pool2d(-_Mg_f, kernel_size=_k, stride=1, padding=_k//2)
        _boundary = (_Mg_d - _Mg_e).clamp(0, 1)
        _diff = (pred_img.float() - person_imgs.float()).abs().mean(dim=1, keepdim=True)
        L_boundary_l1 = ((_diff * _boundary).sum()
                         / (_boundary.sum() + 1e-6))
        L_boundary_l1 = L_boundary_l1.to(L_flow.device, dtype=torch.float32)

    # v22: agnostic-inner invariance loss. The two halves of the doubled batch
    # share noise/sigma and differ only in agnostic inner fill; their predicted
    # clean latents must match inside the edit region.
    L_inv = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    if int(os.environ.get("USE_INVARIANCE", "0")) and transformer.training and B % 2 == 0:
        _hi = B // 2
        _pa = x0_pred[:_hi].float()
        _pb = x0_pred[_hi:].float()
        _minv = M_full[:_hi].float()
        L_inv = (((_pa - _pb).abs() * _minv).sum()
                 / (_minv.sum() * x0_pred.shape[1] + 1e-6)).to(L_flow.device, dtype=torch.float32)
    lambda_inv = float(os.environ.get("LAMBDA_INVARIANCE", "0.05"))

    return {k: v for k, v in locals().items() if not k.startswith("__") and k != "ctx"}