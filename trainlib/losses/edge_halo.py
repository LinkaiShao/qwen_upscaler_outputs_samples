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


"""Chroma-ownership, edge-halo framework, person/bg-shell keeps, inside/mid-band, v6 boundary, no-bg-leak."""

def edge_halo_losses(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    Hi = ctx.get('Hi')
    L_bg_shell_ab = ctx.get('L_bg_shell_ab')
    L_bg_shell_keep = ctx.get('L_bg_shell_keep')
    L_flow = ctx.get('L_flow')
    L_no_bg_leak = ctx.get('L_no_bg_leak')
    L_percep = ctx.get('L_percep')
    L_person_halo_keep = ctx.get('L_person_halo_keep')
    L_tv_ring = ctx.get('L_tv_ring')
    L_v6_boundary = ctx.get('L_v6_boundary')
    M_b_v6 = ctx.get('M_b_v6')
    M_full = ctx.get('M_full')
    M_full_img = ctx.get('M_full_img')
    M_g_v6 = ctx.get('M_g_v6')
    M_k_v6 = ctx.get('M_k_v6')
    M_other_v6 = ctx.get('M_other_v6')
    M_s_v6 = ctx.get('M_s_v6')
    Wi = ctx.get('Wi')
    _C_gt = ctx.get('_C_gt')
    _C_pred = ctx.get('_C_pred')
    _Cg = ctx.get('_Cg')
    _Cp = ctx.get('_Cp')
    _L_abd = ctx.get('_L_abd')
    _L_cr = ctx.get('_L_cr')
    _M_eroded_tv = ctx.get('_M_eroded_tv')
    _M_img_hard_tv = ctx.get('_M_img_hard_tv')
    _Mg = ctx.get('_Mg')
    _Mg_b = ctx.get('_Mg_b')
    _Mg_dil = ctx.get('_Mg_dil')
    _Mgr = ctx.get('_Mgr')
    _ab_d = ctx.get('_ab_d')
    _ab_diff = ctx.get('_ab_diff')
    _ab_dm = ctx.get('_ab_dm')
    _ab_e = ctx.get('_ab_e')
    _ab_err = ctx.get('_ab_err')
    _area_g = ctx.get('_area_g')
    _bg_img = ctx.get('_bg_img')
    _bg_img_i = ctx.get('_bg_img_i')
    _bg_l = ctx.get('_bg_l')
    _bg_lat = ctx.get('_bg_lat')
    _bg_lat_i = ctx.get('_bg_lat_i')
    _bs_band = ctx.get('_bs_band')
    _bs_band_i = ctx.get('_bs_band_i')
    _bs_dil = ctx.get('_bs_dil')
    _bs_dil_i = ctx.get('_bs_dil_i')
    _bs_dil_px = ctx.get('_bs_dil_px')
    _bs_dil_px_i = ctx.get('_bs_dil_px_i')
    _bs_pix_err = ctx.get('_bs_pix_err')
    _bs_sg = ctx.get('_bs_sg')
    _bs_sg_i = ctx.get('_bs_sg_i')
    _bs_sig_gate = ctx.get('_bs_sig_gate')
    _cg_thr = ctx.get('_cg_thr')
    _eCg = ctx.get('_eCg')
    _eCp = ctx.get('_eCp')
    _eab = ctx.get('_eab')
    _earea = ctx.get('_earea')
    _eaw = ctx.get('_eaw')
    _eband = ctx.get('_eband')
    _ebp = ctx.get('_ebp')
    _ecw = ctx.get('_ecw')
    _edit_d = ctx.get('_edit_d')
    _edit_dil = ctx.get('_edit_dil')
    _edit_i = ctx.get('_edit_i')
    _edit_img = ctx.get('_edit_img')
    _efeat = ctx.get('_efeat')
    _eg01 = ctx.get('_eg01')
    _egd = ctx.get('_egd')
    _ege = ctx.get('_ege')
    _eglab = ctx.get('_eglab')
    _egp = ctx.get('_egp')
    _egw = ctx.get('_egw')
    _ehinge = ctx.get('_ehinge')
    _ek = ctx.get('_ek')
    _ep01 = ctx.get('_ep01')
    _eplab = ctx.get('_eplab')
    _eterms = ctx.get('_eterms')
    _g01 = ctx.get('_g01')
    _g01i = ctx.get('_g01i')
    _g01m = ctx.get('_g01m')
    _ga = ctx.get('_ga')
    _gate = ctx.get('_gate')
    _glab = ctx.get('_glab')
    _glab_i = ctx.get('_glab_i')
    _glabm = ctx.get('_glabm')
    _gt01 = ctx.get('_gt01')
    _gt_lab = ctx.get('_gt_lab')
    _gx = ctx.get('_gx')
    _gy = ctx.get('_gy')
    _hf_d = ctx.get('_hf_d')
    _hf_dm = ctx.get('_hf_dm')
    _hi = ctx.get('_hi')
    _hinge = ctx.get('_hinge')
    _hp_g = ctx.get('_hp_g')
    _hp_p = ctx.get('_hp_p')
    _ihdil = ctx.get('_ihdil')
    _ihe = ctx.get('_ihe')
    _iher = ctx.get('_iher')
    _ihp_gt = ctx.get('_ihp_gt')
    _ihp_pred = ctx.get('_ihp_pred')
    _ihsig = ctx.get('_ihsig')
    _lap_g = ctx.get('_lap_g')
    _lap_p = ctx.get('_lap_p')
    _margin = ctx.get('_margin')
    _mid = ctx.get('_mid')
    _mid_d_px = ctx.get('_mid_d_px')
    _mid_dil = ctx.get('_mid_dil')
    _mid_e_px = ctx.get('_mid_e_px')
    _mid_ero = ctx.get('_mid_ero')
    _msig = ctx.get('_msig')
    _oe_px = ctx.get('_oe_px')
    _oepx = ctx.get('_oepx')
    _outside_edit = ctx.get('_outside_edit')
    _p01 = ctx.get('_p01')
    _p01i = ctx.get('_p01i')
    _p01m = ctx.get('_p01m')
    _pers_i = ctx.get('_pers_i')
    _pers_l = ctx.get('_pers_l')
    _person_img = ctx.get('_person_img')
    _person_img_bs = ctx.get('_person_img_bs')
    _person_img_i = ctx.get('_person_img_i')
    _person_lat = ctx.get('_person_lat')
    _person_lat_i = ctx.get('_person_lat_i')
    _pg_i = ctx.get('_pg_i')
    _pg_l = ctx.get('_pg_l')
    _ph = ctx.get('_ph')
    _ph_band = ctx.get('_ph_band')
    _ph_dil = ctx.get('_ph_dil')
    _ph_dil_px = ctx.get('_ph_dil_px')
    _ph_ero = ctx.get('_ph_ero')
    _ph_ero_px = ctx.get('_ph_ero_px')
    _ph_pix_err = ctx.get('_ph_pix_err')
    _ph_sg = ctx.get('_ph_sg')
    _ph_sig_gate = ctx.get('_ph_sig_gate')
    _plab = ctx.get('_plab')
    _plab_i = ctx.get('_plab_i')
    _plabm = ctx.get('_plabm')
    _pred01 = ctx.get('_pred01')
    _pred_lab = ctx.get('_pred_lab')
    _ratio = ctx.get('_ratio')
    _rhinge = ctx.get('_rhinge')
    _ring_dil = ctx.get('_ring_dil')
    _ring_tv = ctx.get('_ring_tv')
    _ring_x = ctx.get('_ring_x')
    _ring_y = ctx.get('_ring_y')
    _sig_cut = ctx.get('_sig_cut')
    _sig_w = ctx.get('_sig_w')
    _sw = ctx.get('_sw')
    _sw2 = ctx.get('_sw2')
    _sw_str = ctx.get('_sw_str')
    _sw_str2 = ctx.get('_sw_str2')
    _target_ratio = ctx.get('_target_ratio')
    _tvr = ctx.get('_tvr')
    agnostic = ctx.get('agnostic')
    batch = ctx.get('batch')
    bg_mean = ctx.get('bg_mean')
    bnd_diff = ctx.get('bnd_diff')
    bnd_v6 = ctx.get('bnd_v6')
    delta_b_v6 = ctx.get('delta_b_v6')
    delta_s_v6 = ctx.get('delta_s_v6')
    denom = ctx.get('denom')
    device = ctx.get('device')
    dist_from_bg = ctx.get('dist_from_bg')
    garment_prior = ctx.get('garment_prior')
    iid = ctx.get('iid')
    lambda_bg_shell_ab = ctx.get('lambda_bg_shell_ab')
    lambda_bg_shell_keep = ctx.get('lambda_bg_shell_keep')
    lambda_inside_ab = ctx.get('lambda_inside_ab')
    lambda_inside_hf = ctx.get('lambda_inside_hf')
    lambda_mid_ab = ctx.get('lambda_mid_ab')
    lambda_mid_hf = ctx.get('lambda_mid_hf')
    lambda_no_bg = ctx.get('lambda_no_bg')
    lambda_person_halo_keep = ctx.get('lambda_person_halo_keep')
    lambda_tv_ring = ctx.get('lambda_tv_ring')
    outside = ctx.get('outside')
    person = ctx.get('person')
    person_imgs = ctx.get('person_imgs')
    pred_f = ctx.get('pred_f')
    pred_img = ctx.get('pred_img')
    repair_band = ctx.get('repair_band')
    repair_img_mask_bg = ctx.get('repair_img_mask_bg')
    ring_v6_full = ctx.get('ring_v6_full')
    sigma = ctx.get('sigma')
    use_v6 = ctx.get('use_v6')
    vae_device = ctx.get('vae_device')
    vgg = ctx.get('vgg')
    weight_dtype = ctx.get('weight_dtype')
    weight_map_img = ctx.get('weight_map_img')
    wub_v6 = ctx.get('wub_v6')
    x0_pred = ctx.get('x0_pred')
    x_hat_0 = ctx.get('x_hat_0')
    x_repair_b_lat = ctx.get('x_repair_b_lat')
    x_repair_s_lat = ctx.get('x_repair_s_lat')
    # ── v25: garment chroma-ownership losses (Lab a,b + chroma hinge) ──
    # Positive pressure: inside the garment region the prediction must match
    # GT chroma. Attacks grey hue directly (grey = a,b magnitude -> 0).
    L_ab = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    L_chroma = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    if int(os.environ.get("USE_CHROMA_LOSS", "0")):
        import kornia
        _pred01 = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _gt01   = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        _pred_lab = kornia.color.rgb_to_lab(_pred01)   # L 0..100, a/b ~-128..127
        _gt_lab   = kornia.color.rgb_to_lab(_gt01)
        _Mg = F.interpolate(garment_prior.float(), size=(Hi, Wi), mode="nearest").to(vae_device)
        _Mg_b = (_Mg > 0.5).float()
        _area_g = _Mg_b.sum().clamp(min=1.0)
        # Loss 1: a,b reconstruction in garment region
        _ab_err = F.smooth_l1_loss(_pred_lab[:, 1:3], _gt_lab[:, 1:3], reduction="none").mean(1, keepdim=True)
        L_ab = ((_ab_err * _Mg_b).sum() / _area_g).to(L_flow.device, dtype=torch.float32)
        # Loss 2: chroma hinge — punish pred being LESS saturated than GT
        _C_pred = torch.sqrt(_pred_lab[:, 1:2] ** 2 + _pred_lab[:, 2:3] ** 2 + 1e-6)
        _C_gt   = torch.sqrt(_gt_lab[:, 1:2] ** 2 + _gt_lab[:, 2:3] ** 2 + 1e-6)
        _margin = float(os.environ.get("CHROMA_MARGIN", "2.0"))
        _hinge = F.relu(_C_gt - _C_pred - _margin)
        L_chroma = ((_hinge * _Mg_b).sum() / _area_g).to(L_flow.device, dtype=torch.float32)
    lambda_ab = float(os.environ.get("LAMBDA_AB", "0.05"))
    lambda_chroma = float(os.environ.get("LAMBDA_CHROMA", "0.03"))

    # ── v28: conditional chroma-RATIO hinge ──
    # Source-localization (D4) showed the latent MSE objective rewards grey.
    # Counter it with an image-space signal the objective lacks: penalise
    # C_pred/C_gt below a ratio, ONLY where GT chroma is high (skip black/
    # white garments), region-gated to the garment, gated to low/mid sigma.
    L_chroma_ratio = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    if int(os.environ.get("USE_CHROMA_RATIO_LOSS", "0")):
        import kornia
        _p01 = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _g01 = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        _plab = kornia.color.rgb_to_lab(_p01)
        _glab = kornia.color.rgb_to_lab(_g01)
        _Cp = torch.sqrt(_plab[:, 1:2] ** 2 + _plab[:, 2:3] ** 2 + 1e-6)
        _Cg = torch.sqrt(_glab[:, 1:2] ** 2 + _glab[:, 2:3] ** 2 + 1e-6)
        _Mgr = (F.interpolate(garment_prior.float(), size=(Hi, Wi), mode="nearest")
                .to(vae_device) > 0.5).float()
        # only pixels where GT chroma is genuinely high (skip black/white)
        _cg_thr = float(os.environ.get("CHROMA_GT_THRESH", "8.0"))
        _hi = (_Cg > _cg_thr).float()
        _ratio = _Cp / (_Cg + 1e-3)
        _target_ratio = float(os.environ.get("CHROMA_TARGET_RATIO", "0.9"))
        _rhinge = F.relu(_target_ratio - _ratio)            # >0 when pred too grey
        _gate = _Mgr * _hi
        _ga = _gate.sum().clamp(min=1.0)
        _L_cr = ((_rhinge * _gate).sum() / _ga).to(L_flow.device, dtype=torch.float32)
        # gate to low/mid sigma (chroma is being set there; skip noisy high sigma)
        _sig_cut = float(os.environ.get("CHROMA_SIGMA_CUTOFF", "0.6"))
        _sig_w = (sigma.float().mean() < _sig_cut).float().to(L_flow.device)
        L_chroma_ratio = _L_cr * _sig_w
        # v29: direct masked Lab a/b loss — points the HUE the right way
        # (the ratio hinge only enforces chroma magnitude). Same high-chroma
        # garment gate, same sigma gate.
        if int(os.environ.get("USE_AB_DIRECTION_LOSS", "0")):
            _ab_e = F.smooth_l1_loss(_plab[:, 1:3], _glab[:, 1:3], reduction="none").mean(1, keepdim=True)
            _L_abd = ((_ab_e * _gate).sum() / _ga).to(L_flow.device, dtype=torch.float32)
            L_ab_direction = _L_abd * _sig_w
        else:
            L_ab_direction = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    else:
        L_ab_direction = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_chroma_ratio = float(os.environ.get("LAMBDA_CHROMA_RATIO", "0.03"))
    lambda_ab_direction = float(os.environ.get("LAMBDA_AB_DIRECTION", "0.08"))

    # ── 5_23 edge-halo loss framework (P2/P4/P6/P8/P10) ──
    # Edge band = dilate(garment) - erode(garment) at image res. Three
    # optional terms, each gated by an env weight:
    #   EDGE_CHROMA_W : chroma-ratio hinge relu(0.9 - C_pred/C_gt) on the band
    #   EDGE_AB_W     : Lab a/b smooth-L1 on the band (hue direction)
    #   EDGE_GRAD_W   : |Laplacian(L_pred) - Laplacian(L_gt)| on the band
    #                   (preserve high-frequency contrast -> kill the washed
    #                    halo). EDGE_BAND_PX sets the band width; EDGE_FEATHER
    #                    > 0 softens the band with a Gaussian.
    L_edge = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    if int(os.environ.get("USE_EDGE_LOSS", "0")):
        import kornia
        _ep01 = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _eg01 = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        _eplab = kornia.color.rgb_to_lab(_ep01)
        _eglab = kornia.color.rgb_to_lab(_eg01)
        _egp = F.interpolate(garment_prior.float(), size=(Hi, Wi), mode="nearest").to(vae_device)
        _ebp = int(os.environ.get("EDGE_BAND_PX", "12"))
        _egd = F.max_pool2d(_egp, 2*_ebp+1, 1, _ebp).clamp(0, 1)
        _ege = (1.0 - F.max_pool2d(1.0 - _egp, 2*_ebp+1, 1, _ebp)).clamp(0, 1)
        _eband = (_egd - _ege).clamp(0, 1)
        _efeat = float(os.environ.get("EDGE_FEATHER", "0.0"))
        if _efeat > 0:
            from torchvision.transforms.functional import gaussian_blur as _gb_e
            _ek = int(2*round(2*_efeat)+1)
            _eband = _gb_e(_eband, kernel_size=[_ek, _ek], sigma=_efeat).clamp(0, 1)
        _earea = _eband.sum().clamp(min=1.0)
        _eterms = torch.zeros((), device=vae_device, dtype=torch.float32)
        _ecw = float(os.environ.get("EDGE_CHROMA_W", "0.0"))
        if _ecw > 0:
            _eCp = torch.sqrt(_eplab[:,1:2]**2 + _eplab[:,2:3]**2 + 1e-6)
            _eCg = torch.sqrt(_eglab[:,1:2]**2 + _eglab[:,2:3]**2 + 1e-6)
            _ehinge = F.relu(0.9 - _eCp/(_eCg+1e-3))
            _eterms = _eterms + _ecw * (_ehinge * _eband).sum() / _earea
        _eaw = float(os.environ.get("EDGE_AB_W", "0.0"))
        if _eaw > 0:
            _eab = F.smooth_l1_loss(_eplab[:,1:3], _eglab[:,1:3], reduction="none").mean(1, keepdim=True)
            _eterms = _eterms + _eaw * (_eab * _eband).sum() / _earea
        _egw = float(os.environ.get("EDGE_GRAD_W", "0.0"))
        if _egw > 0:
            _lap_p = kornia.filters.laplacian(_eplab[:,0:1], kernel_size=3)
            _lap_g = kornia.filters.laplacian(_eglab[:,0:1], kernel_size=3)
            _eterms = _eterms + _egw * ((_lap_p - _lap_g).abs() * _eband).sum() / _earea
        L_edge = _eterms.to(L_flow.device, dtype=torch.float32)

    # ── 5_24 Phase B v2: person-halo "keep" loss, OUTSIDE EDIT REGION only ──
    # Per instructions2: the keep band must exclude the dilated edit region
    # (otherwise we restrain the model where it SHOULD edit). D4 showed
    # 73-97% of person_halo is outside the edit region — train exactly there.
    #   M_person_halo  = dilate(person,15) - erode(person,3)
    #   M_outside_edit = 1 - dilate(M_full, PERSON_HALO_OUTSIDE_EDIT_PX)
    #   M_keep_halo    = M_person_halo * M_outside_edit
    L_person_halo_keep = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_person_halo_keep = float(os.environ.get("LAMBDA_PERSON_HALO_KEEP", "0.0"))
    if lambda_person_halo_keep > 0 and "parse_bg" in batch:
        _bg_lat = batch["parse_bg"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        _person_lat = (1.0 - _bg_lat).clamp(0, 1)
        _person_img = F.interpolate(_person_lat, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _ph_dil_px = int(os.environ.get("PERSON_HALO_DIL_PX", "15"))
        _ph_ero_px = int(os.environ.get("PERSON_HALO_ERO_PX", "3"))
        _ph_dil = F.max_pool2d(_person_img, 2*_ph_dil_px+1, 1, _ph_dil_px).clamp(0, 1)
        _ph_ero = (1.0 - F.max_pool2d(1.0 - _person_img, 2*_ph_ero_px+1, 1, _ph_ero_px)).clamp(0, 1)
        _ph_band = (_ph_dil - _ph_ero).clamp(0, 1)
        # subtract the edit region (dilated) — keep band is OUTSIDE the edit
        _oe_px = int(os.environ.get("PERSON_HALO_OUTSIDE_EDIT_PX", "5"))
        _edit_img = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest")
        _edit_dil = F.max_pool2d(_edit_img, 2*_oe_px+1, 1, _oe_px).clamp(0, 1)
        _outside_edit = (1.0 - _edit_dil).clamp(0, 1)
        _ph_band = (_ph_band * _outside_edit).clamp(0, 1).to(vae_device)
        _ph_sig_gate = float(os.environ.get("PERSON_HALO_SIGMA_GATE", "0.5"))
        _ph_sg = (sigma < _ph_sig_gate).float().view(-1, 1, 1, 1).to(vae_device)
        _ph_band = _ph_band * _ph_sg
        _ph_pix_err = (pred_img - person_imgs).abs().mean(dim=1, keepdim=True)
        L_person_halo_keep = (_ph_pix_err * _ph_band).sum() / (_ph_band.sum() + 1e-6)
        L_person_halo_keep = L_person_halo_keep.to(L_flow.device, dtype=torch.float32)

    # ── 5_24 Phase E: bg-shell "keep" L1 ──
    # bg_shell = dilate(person, 30) AND parse_bg (outside silhouette, near body)
    # Stops garment color/matte bleeding outside the silhouette into bg.
    L_bg_shell_keep = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_bg_shell_keep = float(os.environ.get("LAMBDA_BG_SHELL_KEEP", "0.0"))
    if lambda_bg_shell_keep > 0 and "parse_bg" in batch:
        _bg_lat = batch["parse_bg"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        _person_lat = (1.0 - _bg_lat).clamp(0, 1)
        _person_img_bs = F.interpolate(_person_lat, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _bg_img = F.interpolate(_bg_lat, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _bs_dil_px = int(os.environ.get("BG_SHELL_DIL_PX", "30"))
        _bs_dil = F.max_pool2d(_person_img_bs, 2*_bs_dil_px+1, 1, _bs_dil_px).clamp(0, 1)
        _bs_band = (_bs_dil * _bg_img).clamp(0, 1).to(vae_device)
        _bs_sig_gate = float(os.environ.get("BG_SHELL_SIGMA_GATE", "0.5"))
        _bs_sg = (sigma < _bs_sig_gate).float().view(-1, 1, 1, 1).to(vae_device)
        _bs_band = _bs_band * _bs_sg
        _bs_pix_err = (pred_img - person_imgs).abs().mean(dim=1, keepdim=True)
        L_bg_shell_keep = (_bs_pix_err * _bs_band).sum() / (_bs_band.sum() + 1e-6)
        L_bg_shell_keep = L_bg_shell_keep.to(L_flow.device, dtype=torch.float32)

    # ── 5_24 instructions5: inside_edit_halo (Lab a/b + highpass) ──
    # M_inside_edit_halo = M_person_halo * M_edit. Inside the edit region
    # but on the person silhouette. Use Lab a/b direction + highpass —
    # NOT pixel L1 (this region SHOULD edit; preserve would over-constrain).
    L_inside_edit_halo_ab = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    L_inside_edit_halo_hf = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_inside_ab = float(os.environ.get("LAMBDA_INSIDE_EDIT_AB", "0.0"))
    lambda_inside_hf = float(os.environ.get("LAMBDA_INSIDE_EDIT_HF", "0.0"))
    if (lambda_inside_ab > 0 or lambda_inside_hf > 0) and "parse_bg" in batch:
        import kornia as _kih
        _bg_l = batch["parse_bg"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        _pers_l = (1.0 - _bg_l).clamp(0, 1)
        _pers_i = F.interpolate(_pers_l, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _ihdil = int(os.environ.get("PERSON_HALO_DIL_PX", "15"))
        _iher  = int(os.environ.get("PERSON_HALO_ERO_PX", "3"))
        _ph_dil = F.max_pool2d(_pers_i, 2*_ihdil+1, 1, _ihdil).clamp(0, 1)
        _ph_ero = (1.0 - F.max_pool2d(1.0 - _pers_i, 2*_iher+1, 1, _iher)).clamp(0, 1)
        _ph = (_ph_dil - _ph_ero).clamp(0, 1)
        _oepx = int(os.environ.get("PERSON_HALO_OUTSIDE_EDIT_PX", "5"))
        _edit_i = F.interpolate(M_full.float(), size=(Hi, Wi), mode="nearest")
        _edit_d = F.max_pool2d(_edit_i, 2*_oepx+1, 1, _oepx).clamp(0, 1)
        _ihe = (_ph * _edit_d).clamp(0, 1).to(vae_device)
        _ihsig = (sigma < float(os.environ.get("INSIDE_EDIT_SIGMA_GATE", "0.5"))).float().view(-1, 1, 1, 1).to(vae_device)
        _ihe = _ihe * _ihsig
        # sample weight (T4): per-sample multiplier
        _sw_str = os.environ.get("SAMPLE_WEIGHT_00008", "1.0")
        if _sw_str != "1.0" and "image_id" in batch:
            _sw = torch.tensor([float(_sw_str) if iid == "00008_00" else 1.0 for iid in batch["image_id"]],
                               device=vae_device, dtype=torch.float32).view(-1, 1, 1, 1)
            _ihe = _ihe * _sw
        _p01 = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _g01 = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        if lambda_inside_ab > 0:
            _plab = _kih.color.rgb_to_lab(_p01); _glab = _kih.color.rgb_to_lab(_g01)
            _ab_d = F.smooth_l1_loss(_plab[:,1:3], _glab[:,1:3], reduction="none").mean(1, keepdim=True)
            L_inside_edit_halo_ab = ((_ab_d * _ihe).sum() / (_ihe.sum() + 1e-6)).to(L_flow.device, torch.float32)
        if lambda_inside_hf > 0:
            _ihp_pred = _p01 - F.avg_pool2d(_p01, 3, 1, 1)
            _ihp_gt   = _g01 - F.avg_pool2d(_g01, 3, 1, 1)
            _hf_d = (_ihp_pred - _ihp_gt).abs().mean(dim=1, keepdim=True)
            L_inside_edit_halo_hf = ((_hf_d * _ihe).sum() / (_ihe.sum() + 1e-6)).to(L_flow.device, torch.float32)

    # ── 5_24 instructions5: mid-band film loss (Lab a/b + highpass) ──
    # M_mid = dilate(M_garment, 10) - erode(M_garment, 12). The
    # layer-of-grain over the edit region.
    L_mid_ab = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    L_mid_hf = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_mid_ab = float(os.environ.get("LAMBDA_MID_AB", "0.0"))
    lambda_mid_hf = float(os.environ.get("LAMBDA_MID_HF", "0.0"))
    if (lambda_mid_ab > 0 or lambda_mid_hf > 0) and "parse_garment" in batch:
        import kornia as _kmid
        _pg_l = batch["parse_garment"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        _pg_i = F.interpolate(_pg_l, size=(Hi, Wi), mode="nearest").clamp(0, 1)
        _mid_d_px = int(os.environ.get("MID_DIL_PX", "10"))
        _mid_e_px = int(os.environ.get("MID_ERO_PX", "12"))
        _mid_dil  = F.max_pool2d(_pg_i, 2*_mid_d_px+1, 1, _mid_d_px).clamp(0, 1)
        _mid_ero  = (1.0 - F.max_pool2d(1.0 - _pg_i, 2*_mid_e_px+1, 1, _mid_e_px)).clamp(0, 1)
        _mid = (_mid_dil - _mid_ero).clamp(0, 1).to(vae_device)
        _msig = (sigma < float(os.environ.get("MID_SIGMA_GATE", "0.5"))).float().view(-1, 1, 1, 1).to(vae_device)
        _mid = _mid * _msig
        _sw_str2 = os.environ.get("SAMPLE_WEIGHT_00008", "1.0")
        if _sw_str2 != "1.0" and "image_id" in batch:
            _sw2 = torch.tensor([float(_sw_str2) if iid == "00008_00" else 1.0 for iid in batch["image_id"]],
                                device=vae_device, dtype=torch.float32).view(-1, 1, 1, 1)
            _mid = _mid * _sw2
        _p01m = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _g01m = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        if lambda_mid_ab > 0:
            _plabm = _kmid.color.rgb_to_lab(_p01m); _glabm = _kmid.color.rgb_to_lab(_g01m)
            _ab_dm = F.smooth_l1_loss(_plabm[:,1:3], _glabm[:,1:3], reduction="none").mean(1, keepdim=True)
            L_mid_ab = ((_ab_dm * _mid).sum() / (_mid.sum() + 1e-6)).to(L_flow.device, torch.float32)
        if lambda_mid_hf > 0:
            _hp_p = _p01m - F.avg_pool2d(_p01m, 3, 1, 1)
            _hp_g = _g01m - F.avg_pool2d(_g01m, 3, 1, 1)
            _hf_dm = (_hp_p - _hp_g).abs().mean(dim=1, keepdim=True)
            L_mid_hf = ((_hf_dm * _mid).sum() / (_mid.sum() + 1e-6)).to(L_flow.device, torch.float32)

    # ── 5_24 Phase I: bg-shell Lab a/b smooth-L1 ──
    # Pull pred a/b toward GT a/b on bg_shell. Not chroma magnitude — direction.
    L_bg_shell_ab = torch.zeros((), device=L_flow.device, dtype=torch.float32)
    lambda_bg_shell_ab = float(os.environ.get("LAMBDA_BG_SHELL_AB", "0.0"))
    if lambda_bg_shell_ab > 0 and "parse_bg" in batch:
        import kornia as _kab
        _bg_lat_i = batch["parse_bg"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        _person_lat_i = (1.0 - _bg_lat_i).clamp(0, 1)
        _person_img_i = F.interpolate(_person_lat_i, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _bg_img_i = F.interpolate(_bg_lat_i, size=(Hi, Wi), mode="bilinear", align_corners=False).clamp(0, 1)
        _bs_dil_px_i = int(os.environ.get("BG_SHELL_DIL_PX", "30"))
        _bs_dil_i = F.max_pool2d(_person_img_i, 2*_bs_dil_px_i+1, 1, _bs_dil_px_i).clamp(0, 1)
        _bs_band_i = (_bs_dil_i * _bg_img_i).clamp(0, 1).to(vae_device)
        _bs_sg_i = (sigma < float(os.environ.get("BG_SHELL_SIGMA_GATE", "0.5"))).float().view(-1, 1, 1, 1).to(vae_device)
        _bs_band_i = _bs_band_i * _bs_sg_i
        _p01i = ((pred_img + 1.0) / 2.0).clamp(0, 1).float()
        _g01i = ((person_imgs + 1.0) / 2.0).clamp(0, 1).float()
        _plab_i = _kab.color.rgb_to_lab(_p01i)
        _glab_i = _kab.color.rgb_to_lab(_g01i)
        _ab_diff = F.smooth_l1_loss(_plab_i[:,1:3], _glab_i[:,1:3], reduction="none").mean(1, keepdim=True)
        L_bg_shell_ab = (_ab_diff * _bs_band_i).sum() / (_bs_band_i.sum() + 1e-6)
        L_bg_shell_ab = L_bg_shell_ab.to(L_flow.device, dtype=torch.float32)

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
    # If PERCEPTUAL_REGION=garment, the masked-to-garment variant is computed
    # later (in the "5_29 new losses" block) to avoid running VGG twice.
    L_percep = torch.tensor(0.0, device=device)
    if int(os.environ.get("USE_PERCEPTUAL", "0")) and os.environ.get("PERCEPTUAL_REGION", "") != "garment":
        vgg = get_vgg_features(vae_device, weight_dtype)
        L_percep = perceptual_loss(pred_img, person_imgs, weight_map_img, vgg)
        L_percep = L_percep.to(L_flow.device, dtype=torch.float32)

    return {k: v for k, v in locals().items() if not k.startswith("__") and k != "ctx"}