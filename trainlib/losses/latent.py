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


"""Flow-matching loss, v6 specialized heads, latent recon/route/repair, and the VAE decode to pred_img."""

def latent_losses(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    B = ctx.get('B')
    C = ctx.get('C')
    C_p_ = ctx.get('C_p_')
    C_t = ctx.get('C_t')
    E_bdry = ctx.get('E_bdry')
    E_int = ctx.get('E_int')
    H = ctx.get('H')
    H2 = ctx.get('H2')
    L_b_keep = ctx.get('L_b_keep')
    L_b_loss = ctx.get('L_b_loss')
    L_flow = ctx.get('L_flow')
    L_s_keep = ctx.get('L_s_keep')
    L_s_loss = ctx.get('L_s_loss')
    L_tv_edge = ctx.get('L_tv_edge')
    M_b_v6 = ctx.get('M_b_v6')
    M_full = ctx.get('M_full')
    M_g_v6 = ctx.get('M_g_v6')
    M_garment_v6 = ctx.get('M_garment_v6')
    M_k_v6 = ctx.get('M_k_v6')
    M_model_v6 = ctx.get('M_model_v6')
    M_other_v6 = ctx.get('M_other_v6')
    M_repair_bg_v6 = ctx.get('M_repair_bg_v6')
    M_repair_skin_v6 = ctx.get('M_repair_skin_v6')
    M_s_v6 = ctx.get('M_s_v6')
    W = ctx.get('W')
    W2 = ctx.get('W2')
    Wmap_p = ctx.get('Wmap_p')
    _M_blur_train = ctx.get('_M_blur_train')
    _beta_for_l2 = ctx.get('_beta_for_l2')
    _blur = ctx.get('_blur')
    _bsq = ctx.get('_bsq')
    _c = ctx.get('_c')
    _dec_half = ctx.get('_dec_half')
    _dec_list = ctx.get('_dec_list')
    _den = ctx.get('_den')
    _di = ctx.get('_di')
    _dilate_mf = ctx.get('_dilate_mf')
    _eb_x = ctx.get('_eb_x')
    _eb_y = ctx.get('_eb_y')
    _edge_band = ctx.get('_edge_band')
    _gate_for_l2 = ctx.get('_gate_for_l2')
    _gate_mode = ctx.get('_gate_mode')
    _gp_d = ctx.get('_gp_d')
    _gp_e = ctx.get('_gp_e')
    _gt = ctx.get('_gt')
    _gx = ctx.get('_gx')
    _gy = ctx.get('_gy')
    _hi_th = ctx.get('_hi_th')
    _high_suppress = ctx.get('_high_suppress')
    _hs_amt = ctx.get('_hs_amt')
    _hs_th = ctx.get('_hs_th')
    _hs_w = ctx.get('_hs_w')
    _img_wbg = ctx.get('_img_wbg')
    _img_wbody = ctx.get('_img_wbody')
    _img_wrep = ctx.get('_img_wrep')
    _inM = ctx.get('_inM')
    _iv_dec = ctx.get('_iv_dec')
    _k_blur = ctx.get('_k_blur')
    _k_wm = ctx.get('_k_wm')
    _late_w = ctx.get('_late_w')
    _lo_th = ctx.get('_lo_th')
    _ls_power = ctx.get('_ls_power')
    _msg = ctx.get('_msg')
    _ng = ctx.get('_ng')
    _nm = ctx.get('_nm')
    _npM = ctx.get('_npM')
    _num = ctx.get('_num')
    _pack_mask = ctx.get('_pack_mask')
    _pbr = ctx.get('_pbr')
    _pr = ctx.get('_pr')
    _prM = ctx.get('_prM')
    _prec = ctx.get('_prec')
    _pred = ctx.get('_pred')
    _ps_v6 = ctx.get('_ps_v6')
    _rb = ctx.get('_rb')
    _rec = ctx.get('_rec')
    _reg_mse = ctx.get('_reg_mse')
    _s = ctx.get('_s')
    _std_w = ctx.get('_std_w')
    _th = ctx.get('_th')
    _ub_recon = ctx.get('_ub_recon')
    _v6_use_dilate_route = ctx.get('_v6_use_dilate_route')
    _vp_hi = ctx.get('_vp_hi')
    _vp_hi_p = ctx.get('_vp_hi_p')
    _vp_lat = ctx.get('_vp_lat')
    _vp_lo = ctx.get('_vp_lo')
    _vp_lo_p = ctx.get('_vp_lo_p')
    _vt_hi = ctx.get('_vt_hi')
    _vt_hi_p = ctx.get('_vt_hi_p')
    _vt_lo = ctx.get('_vt_lo')
    _vt_lo_p = ctx.get('_vt_lo_p')
    _w_sigma = ctx.get('_w_sigma')
    _wm_for_route = ctx.get('_wm_for_route')
    _wm_sig = ctx.get('_wm_sig')
    _x0f = ctx.get('_x0f')
    active = ctx.get('active')
    agnostic = ctx.get('agnostic')
    batch = ctx.get('batch')
    bdry_hi = ctx.get('bdry_hi')
    bdry_lo = ctx.get('bdry_lo')
    bdry_region = ctx.get('bdry_region')
    bg_repair = ctx.get('bg_repair')
    body_repair = ctx.get('body_repair')
    boundary_mask = ctx.get('boundary_mask')
    decoded = ctx.get('decoded')
    delta_b_lat = ctx.get('delta_b_lat')
    delta_b_v6 = ctx.get('delta_b_v6')
    delta_s_lat = ctx.get('delta_s_lat')
    delta_s_v6 = ctx.get('delta_s_v6')
    delta_target = ctx.get('delta_target')
    denom = ctx.get('denom')
    denorm = ctx.get('denorm')
    device = ctx.get('device')
    diff_l1 = ctx.get('diff_l1')
    dist_to_grey = ctx.get('dist_to_grey')
    energy = ctx.get('energy')
    f_bdry = ctx.get('f_bdry')
    f_int = ctx.get('f_int')
    frac_ratio = ctx.get('frac_ratio')
    g_hi_freq = ctx.get('g_hi_freq')
    g_lo_freq = ctx.get('g_lo_freq')
    gar_energy = ctx.get('gar_energy')
    gar_flat = ctx.get('gar_flat')
    gar_sorted = ctx.get('gar_sorted')
    garment_prior = ctx.get('garment_prior')
    gp_dilated = ctx.get('gp_dilated')
    gp_eroded = ctx.get('gp_eroded')
    grad_x = ctx.get('grad_x')
    grad_y = ctx.get('grad_y')
    grey_ref = ctx.get('grey_ref')
    hidden_C = ctx.get('hidden_C')
    hidden_full = ctx.get('hidden_full')
    id_late_w = ctx.get('id_late_w')
    identity_mask = ctx.get('identity_mask')
    idx_75 = ctx.get('idx_75')
    int_hi = ctx.get('int_hi')
    int_lo = ctx.get('int_lo')
    int_region = ctx.get('int_region')
    interior_mask = ctx.get('interior_mask')
    keep_for_b = ctx.get('keep_for_b')
    keep_for_s = ctx.get('keep_for_s')
    keep_mask = ctx.get('keep_mask')
    l1_b = ctx.get('l1_b')
    l1_s = ctx.get('l1_s')
    lambda_anti_grey = ctx.get('lambda_anti_grey')
    lambda_late_shell = ctx.get('lambda_late_shell')
    lambda_tv_edge = ctx.get('lambda_tv_edge')
    loss_weights = ctx.get('loss_weights')
    m = ctx.get('m')
    m_ag = ctx.get('m_ag')
    m_bg_p = ctx.get('m_bg_p')
    m_body_p = ctx.get('m_body_p')
    m_core_p = ctx.get('m_core_p')
    m_keep_p = ctx.get('m_keep_p')
    m_repair_p = ctx.get('m_repair_p')
    m_ub_p = ctx.get('m_ub_p')
    m_v = ctx.get('m_v')
    margin = ctx.get('margin')
    mask_p = ctx.get('mask_p')
    n_gar = ctx.get('n_gar')
    person = ctx.get('person')
    pred_C = ctx.get('pred_C')
    pred_img = ctx.get('pred_img')
    r_max = ctx.get('r_max')
    ratio = ctx.get('ratio')
    rb = ctx.get('rb')
    repair_band = ctx.get('repair_band')
    rho_max = ctx.get('rho_max')
    route_logits = ctx.get('route_logits')
    route_packed = ctx.get('route_packed')
    s = ctx.get('s')
    s_map = ctx.get('s_map')
    s_v = ctx.get('s_v')
    sigma = ctx.get('sigma')
    silh_early_w = ctx.get('silh_early_w')
    silhouette_mask = ctx.get('silhouette_mask')
    snr_w = ctx.get('snr_w')
    sq_err = ctx.get('sq_err')
    sq_err_hi = ctx.get('sq_err_hi')
    sq_err_lo = ctx.get('sq_err_lo')
    sq_err_std = ctx.get('sq_err_std')
    tau = ctx.get('tau')
    transformer = ctx.get('transformer')
    ub_exp = ctx.get('ub_exp')
    uncertain_band = ctx.get('uncertain_band')
    use_v6 = ctx.get('use_v6')
    v6_heads = ctx.get('v6_heads')
    v6_out = ctx.get('v6_out')
    v_abs = ctx.get('v_abs')
    v_pred_lat = ctx.get('v_pred_lat')
    v_target = ctx.get('v_target')
    v_ub = ctx.get('v_ub')
    vae = ctx.get('vae')
    vae_device = ctx.get('vae_device')
    vt_p = ctx.get('vt_p')
    w_bdry = ctx.get('w_bdry')
    w_bdry_hi = ctx.get('w_bdry_hi')
    w_bdry_img = ctx.get('w_bdry_img')
    w_bdry_lo = ctx.get('w_bdry_lo')
    w_bg_rep = ctx.get('w_bg_rep')
    w_body_rep = ctx.get('w_body_rep')
    w_early = ctx.get('w_early')
    w_garment = ctx.get('w_garment')
    w_int = ctx.get('w_int')
    w_int_hi = ctx.get('w_int_hi')
    w_int_img = ctx.get('w_int_img')
    w_int_lo = ctx.get('w_int_lo')
    w_keep = ctx.get('w_keep')
    w_keep_d = ctx.get('w_keep_d')
    w_keep_s = ctx.get('w_keep_s')
    w_rb = ctx.get('w_rb')
    w_repair = ctx.get('w_repair')
    w_rs = ctx.get('w_rs')
    w_uncertain = ctx.get('w_uncertain')
    w_warp_fp = ctx.get('w_warp_fp')
    warp_fp_region = ctx.get('warp_fp_region')
    wbg = ctx.get('wbg')
    wbr = ctx.get('wbr')
    wc = ctx.get('wc')
    weight_dtype = ctx.get('weight_dtype')
    weight_map = ctx.get('weight_map')
    wk = ctx.get('wk')
    wr = ctx.get('wr')
    wub = ctx.get('wub')
    x = ctx.get('x')
    x0_5d = ctx.get('x0_5d')
    x0_decoded_input = ctx.get('x0_decoded_input')
    x0_pred = ctx.get('x0_pred')
    y_route = ctx.get('y_route')
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
            w_warp_fp = float(os.environ.get("W_WARP_FP", "0.0"))    # boost flow-loss weight in warp's false-positive zone
            weight_map = (w_keep * keep_mask
                          + w_garment * garment_prior
                          + w_uncertain * uncertain_band
                          + w_repair * repair_band
                          + w_warp_fp * warp_fp_region)
        # Optionally smooth the weight_map boundaries to prevent model from
        # memorizing hard zone transitions as visible output edges.
        if int(os.environ.get("WEIGHT_MAP_SOFT", "0")):
            from torchvision.transforms.functional import gaussian_blur as _gb_wm
            _wm_sig = float(os.environ.get("WEIGHT_MAP_SOFT_SIG", "2.0"))
            _k_wm = int(2 * round(2 * _wm_sig) + 1)
            weight_map = _gb_wm(weight_map.float(), kernel_size=[_k_wm, _k_wm],
                                 sigma=_wm_sig).to(weight_map.dtype)
    Wmap_p = pack_latents(weight_map.expand(B, C, H, W), B, C, H, W).mean(dim=-1, keepdim=True)

    # USE_FREQ_FLOW=1: frequency-decomposed flow loss with sigma gates.
    # High σ → low-freq channel dominates (structure/regions); low σ → high-freq (texture).
    # Avg-pool 3x3 in latent space (k env-tunable). Region weighting still applies via Wmap_p.
    if int(os.environ.get("USE_FREQ_FLOW", "0")):
        _vp_lat = unpack_latents(pred_C, B, C, H, W)  # (B, 16, 128, 96)
        _k_blur = int(os.environ.get("FREQ_BLUR_K", "3"))
        def _blur(x):
            return F.avg_pool2d(x, kernel_size=_k_blur, stride=1, padding=_k_blur//2)
        _vp_lo = _blur(_vp_lat.float())
        _vt_lo = _blur(v_target.float())
        _vp_hi = _vp_lat.float() - _vp_lo
        _vt_hi = v_target.float() - _vt_lo
        # Pack each component back to token form
        _vp_lo_p = pack_latents(_vp_lo, B, C, H, W)
        _vt_lo_p = pack_latents(_vt_lo, B, C, H, W)
        _vp_hi_p = pack_latents(_vp_hi, B, C, H, W)
        _vt_hi_p = pack_latents(_vt_hi, B, C, H, W)
        sq_err_lo = ((_vp_lo_p - _vt_lo_p) ** 2).mean(dim=-1, keepdim=True)
        sq_err_hi = ((_vp_hi_p - _vt_hi_p) ** 2).mean(dim=-1, keepdim=True)
        # Sigma gates (per-sample). σ near 1 = noisy = low-freq target; σ near 0 = high-freq.
        # FREQ_GATE_MODE=complement (default): g_lo + g_hi = 1, no dead zone.
        #                =dual: separate thresholds (mid-σ underweighted, deprecated).
        _s = sigma.float().view(B, 1, 1)
        _w_sigma = float(os.environ.get("FREQ_SIGMA_WIDTH", "0.10"))
        _gate_mode = os.environ.get("FREQ_GATE_MODE", "complement")
        if _gate_mode == "complement":
            _th = float(os.environ.get("FREQ_THRESHOLD", "0.5"))
            g_lo_freq = torch.sigmoid((_s - _th) / _w_sigma)
            g_hi_freq = 1.0 - g_lo_freq
            # Optional sharp high-σ suppression: at σ above HIGH_SUPPRESS_TH,
            # push high-freq weight toward 0 (pure-noise regime should only see
            # low-freq teaching). After suppression, renormalize so gates still
            # sum to 1.
            _hs_amt = float(os.environ.get("HIGH_SUPPRESS_AMT", "0.0"))
            if _hs_amt > 0:
                _hs_th = float(os.environ.get("HIGH_SUPPRESS_TH", "0.65"))
                _hs_w  = float(os.environ.get("HIGH_SUPPRESS_WIDTH", "0.05"))
                _high_suppress = torch.sigmoid((_s - _hs_th) / _hs_w)
                g_hi_freq = g_hi_freq * (1.0 - _hs_amt * _high_suppress)
                g_lo_freq = 1.0 - g_hi_freq
        else:
            _hi_th = float(os.environ.get("FREQ_HIGH_SIGMA_TH", "0.65"))
            _lo_th = float(os.environ.get("FREQ_LOW_SIGMA_TH",  "0.35"))
            g_lo_freq = torch.sigmoid((_s - _hi_th) / _w_sigma)
            g_hi_freq = torch.sigmoid((_lo_th - _s) / _w_sigma)
        sq_err = g_lo_freq * sq_err_lo + g_hi_freq * sq_err_hi
        # Optional small "standard" sq_err mixed in (recovers the lost cross-term
        # 2*diff_low*diff_high that the bifurcation drops). Default 0.0 keeps
        # pure bifurcation; small (e.g. 0.1) acts as a gradient-stability anchor.
        _std_w = float(os.environ.get("FREQ_STD_WEIGHT", "0.0"))
        if _std_w > 0:
            sq_err_std = ((pred_C.float() - vt_p.float()) ** 2).mean(dim=-1, keepdim=True)
            sq_err = sq_err + _std_w * sq_err_std
    else:
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

    # ── L_warp_fp: kept as 0; the warp_fp boost now lives in weight_map (W_WARP_FP env). ──
    L_warp_fp = torch.tensor(0.0, device=device)
    _w_fp = 0.0

    # ── L_beta_l2: magnitude regularization on the garment_net's β output, weighted by
    # the spatial gate so we only penalize β within the region that actually contributes
    # to the transformer hidden state (off-gate β is zeroed by the hook anyway). The user
    # asked for this to combat over-injection on easy cases — large lambda → smaller β.
    L_beta_l2 = torch.tensor(0.0, device=device)
    if (int(os.environ.get("USE_GARMENT_NET", "0"))
            and os.environ.get("GARMENT_NET_MODE", "norm_residual") == "adaln"):
        _beta_for_l2 = state._GARMENT_RESIDUAL_HOLDER.get("beta")
        _gate_for_l2 = state._GARMENT_RESIDUAL_HOLDER.get("gate")
        if _beta_for_l2 is not None:
            if _gate_for_l2 is not None:
                _bsq = (_beta_for_l2.float() ** 2).mean(dim=-1, keepdim=True)  # (B, T, 1)
                L_beta_l2 = (_bsq * _gate_for_l2.float()).sum() / (_gate_for_l2.float().sum() + 1e-6)
            else:
                L_beta_l2 = (_beta_for_l2.float() ** 2).mean()

    # ── Specialized heads: produce δ (repair residual) + route logits from v_pred features ──
    # Each head has its own parameters → their gradients don't cross.
    # - RepairHead: L1 δ residual supervised in ring (trains repair_head only)
    # - RoutingHead: CE 4-class supervised globally (trains routing_head only)
    # - Main transformer v_pred: flow MSE supervised in M_g (trains main only)
    delta_s_v6 = torch.zeros_like(v_pred_lat)
    delta_b_v6 = torch.zeros_like(v_pred_lat)
    route_logits = None
    if use_v6 and "hidden" in state._HIDDEN_HOLDER:
        hidden_full = state._HIDDEN_HOLDER["hidden"]                                       # (B, N_total, 3072)
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
    _ub_recon = uncertain_band.float()
    if int(os.environ.get("BG_RING_ISOLATION", "0")) and "parse_bg" in batch:   # isolation: recon excludes bg side of the band
        _pbr = batch["parse_bg"].to(device, weight_dtype)
        if _pbr.dim() == 3: _pbr = _pbr.unsqueeze(1)
        if tuple(_pbr.shape[-2:]) != tuple(uncertain_band.shape[-2:]):
            _pbr = F.interpolate((_pbr > 0.5).float(), size=uncertain_band.shape[-2:], mode="nearest")
        _ub_recon = (_ub_recon * (1.0 - (_pbr > 0.5).float())).clamp(0, 1)
    L_recon_ub = ((diff_l1 * _ub_recon).sum() / (_ub_recon.sum() + 1e-6))

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
        # DILATE_M_FULL > 0 changes the v6 region target: route is computed
        # INSIDE M_model (= dilated M_full) only, with NO "keep" class. The
        # 3 active classes are garment / repair_skin / repair_bg. The 4th
        # class is treated as "outside M_model = ignored" via ignore_index.
        # See 5_28/instructions6.md.
        _v6_use_dilate_route = int(os.environ.get("V6_DILATE_ROUTE", "1" if _dilate_mf > 0 else "0"))
        if _v6_use_dilate_route:
            # M_model = M_full (already dilated upstream when DILATE_M_FULL>0)
            M_model_v6 = M_full.to(weight_dtype)
            # M_garment = warped-mask garment ∩ M_model (NOT M_g_v6 which uses dilation by V6_R_OUT)
            _wm_for_route = batch["warped_mask"].to(device, dtype=weight_dtype)
            if _wm_for_route.dim() == 3: _wm_for_route = _wm_for_route.unsqueeze(1)
            M_garment_v6 = ((_wm_for_route > 0.5).to(weight_dtype) * M_model_v6).clamp(0, 1)
            # M_repair_skin = parse_skin ∩ M_model ∩ ~M_garment (parse_skin = face+neck+arms in this codebase)
            if "parse_skin" in batch:
                _ps_v6 = batch["parse_skin"].to(device, dtype=weight_dtype)
                if _ps_v6.dim() == 3: _ps_v6 = _ps_v6.unsqueeze(1)
                M_repair_skin_v6 = ((_ps_v6 > 0.5).to(weight_dtype) * M_model_v6 * (1 - M_garment_v6)).clamp(0, 1)
            else:
                M_repair_skin_v6 = torch.zeros_like(M_garment_v6)
            # M_repair_bg = M_model ∩ ~M_garment ∩ ~M_repair_skin  (catches hair, neck, bg, anything else)
            M_repair_bg_v6 = (M_model_v6 * (1 - M_garment_v6) * (1 - M_repair_skin_v6)).clamp(0, 1)

            delta_target = (person - agnostic).detach()
            l1_s = (delta_s_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
            l1_b = (delta_b_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
            L_s_loss = (l1_s * M_repair_skin_v6.float()).sum() / (M_repair_skin_v6.float().sum() + 1e-6)
            L_b_loss = (l1_b * M_repair_bg_v6.float()).sum() / (M_repair_bg_v6.float().sum() + 1e-6)
            # Heads should output ZERO outside their region (still desirable). Outside
            # M_model is unsupervised because it gets pasted from agnostic anyway.
            keep_for_s = ((1.0 - M_repair_skin_v6) * M_model_v6).float()
            keep_for_b = ((1.0 - M_repair_bg_v6)   * M_model_v6).float()
            L_s_keep = (delta_s_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_s).sum() / (keep_for_s.sum() + 1e-6)
            L_b_keep = (delta_b_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_b).sum() / (keep_for_b.sum() + 1e-6)
            w_rs   = float(os.environ.get("W_V6_DELTA_S",     "0.5"))
            w_rb   = float(os.environ.get("W_V6_DELTA_B",     "0.5"))
            w_keep_d = float(os.environ.get("W_V6_DELTA_KEEP", "1.0"))
            L_repair_v6 = (w_rs * L_s_loss + w_rb * L_b_loss
                         + w_keep_d * (L_s_keep + L_b_keep)).to(L_flow.device, dtype=torch.float32)

            # Routing target: 3 active classes inside M_model, class 3 = ignored outside.
            y_route = torch.full((B, H, W), 3, device=device, dtype=torch.long)  # default: ignore
            y_route[(M_garment_v6[:, 0]     > 0.5)] = 0
            y_route[(M_repair_skin_v6[:, 0] > 0.5)] = 1
            y_route[(M_repair_bg_v6[:, 0]   > 0.5)] = 2
            # ignore_index=3 means CE skips outside-M_model pixels entirely.
            L_route_v6 = F.cross_entropy(route_logits.float(), y_route, ignore_index=3).to(L_flow.device, dtype=torch.float32)

            if int(os.environ.get("V6_ROUTE_ACC", "0")):
                with torch.no_grad():
                    _pred = route_logits.float().argmax(dim=1)                       # (B,H,W) over 4 classes
                    _msg = [f"[v6route] sig={float(sigma.float().mean()):.2f}"]
                    for _c, _nm in [(0, "gar"), (1, "skin"), (2, "BG")]:
                        _gt = (y_route == _c)            # GT pixels of this class (inside M_model)
                        _pr = (_pred == _c)
                        _ng = int(_gt.sum().item())
                        _rec = (float((_pred[_gt] == _c).float().mean()) if _ng > 0 else float("nan"))
                        # precision restricted to inside-M_model (ignore class-3 region)
                        _inM = (y_route != 3)
                        _prM = _pr & _inM
                        _npM = int(_prM.sum().item())
                        _prec = (float((_gt[_prM]).float().mean()) if _npM > 0 else float("nan"))
                        _msg.append(f"{_nm}:rec={_rec:.3f} prec={_prec:.3f} n={_ng}")
                    print("  ".join(_msg), flush=True)
        else:
            # LEGACY path: 4-class scheme used by the V15 / 10k baseline.
            # δ_s supervised in M_s; δ_b supervised in M_b ∪ M_other.
            delta_target = (person - agnostic).detach()
            l1_s = (delta_s_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
            l1_b = (delta_b_v6.float() - delta_target.float()).abs().mean(dim=1, keepdim=True)
            L_s_loss = (l1_s * M_s_v6.float()).sum() / (M_s_v6.float().sum() + 1e-6)
            L_b_loss = (l1_b * (M_b_v6 + M_other_v6).float()).sum() / ((M_b_v6 + M_other_v6).float().sum() + 1e-6)
            keep_for_s = (1.0 - M_s_v6).float()
            keep_for_b = (1.0 - (M_b_v6 + M_other_v6)).float()
            L_s_keep = (delta_s_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_s).sum() / (keep_for_s.sum() + 1e-6)
            L_b_keep = (delta_b_v6.float().abs().mean(dim=1, keepdim=True) * keep_for_b).sum() / (keep_for_b.sum() + 1e-6)
            w_rs   = float(os.environ.get("W_V6_DELTA_S",     "0.5"))
            w_rb   = float(os.environ.get("W_V6_DELTA_B",     "0.5"))
            w_keep_d = float(os.environ.get("W_V6_DELTA_KEEP", "1.0"))
            L_repair_v6 = (w_rs * L_s_loss + w_rb * L_b_loss
                         + w_keep_d * (L_s_keep + L_b_keep)).to(L_flow.device, dtype=torch.float32)

            y_route = torch.zeros((B, H, W), device=device, dtype=torch.long)
            y_route[(M_g_v6[:, 0] > 0.5)] = 0
            y_route[(M_s_v6[:, 0] > 0.5)] = 1
            y_route[(M_b_v6[:, 0] > 0.5)] = 2
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
    # Optional: apply inference-style soft composite at training time so train and
    # inference share the same pred_img base. Critical for output-space garment
    # net training (otherwise gnet learns to "compensate" for the raw x0_pred at
    # high sigma, then over-corrects at inference where the composite is applied).
    if int(os.environ.get("USE_TRAIN_COMPOSITE", "0")):
        from torchvision.transforms.functional import gaussian_blur as _gb_train_comp
        _M_blur_train = _gb_train_comp(M_full.float(), kernel_size=[7, 7], sigma=2.0).to(M_full.dtype)
        x0_decoded_input = (1 - _M_blur_train) * agnostic + _M_blur_train * x0_pred
    else:
        x0_decoded_input = x0_pred

    x0_5d = x0_decoded_input.to(vae_device, dtype=weight_dtype).unsqueeze(2)
    m_v = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
    s_v = torch.tensor(vae.config.latents_std ).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
    denorm = x0_5d * s_v + m_v
    # Decode one sample at a time to cap the VAE's forward memory transient.
    # v22: for the invariance-doubled batch, decode the 2nd half WITHOUT grad —
    # L_img then trains only the 1st fill variant (= v16's original img loss),
    # which keeps GPU-1 memory at single-batch level. L_flow and L_inv still
    # supervise the 2nd half (they don't route through the VAE).
    _iv_dec = int(os.environ.get("USE_INVARIANCE", "0")) and transformer.training and denorm.shape[0] % 2 == 0
    _dec_half = denorm.shape[0] // 2 if _iv_dec else denorm.shape[0]
    with torch.amp.autocast("cuda", dtype=weight_dtype):
        _dec_list = []
        for _di in range(denorm.shape[0]):
            if _di < _dec_half:
                _dec_list.append(vae.decode(denorm[_di:_di+1], return_dict=False)[0][:, :, 0])
            else:
                with torch.no_grad():
                    _dec_list.append(vae.decode(denorm[_di:_di+1], return_dict=False)[0][:, :, 0])
        decoded = torch.cat(_dec_list, 0)
    pred_img = decoded.clamp(-1, 1)
    Hi, Wi = pred_img.shape[2], pred_img.shape[3]
    return {k: v for k, v in locals().items() if not k.startswith("__") and k != "ctx"}