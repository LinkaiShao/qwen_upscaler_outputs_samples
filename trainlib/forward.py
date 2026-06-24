"""train_step — the forward pass + loss assembly."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
try:
    from multi_block_garment import (MultiBlockGarmentInjection, install_multi_block_hooks,
                                      build_spatial_mask_from_warped, per_block_norms_table)
except Exception:
    pass
from trainlib import state

from trainlib.models import (GarmentCrossAttn, GarmentLatentEnhancer, GarmentNet, GarmentNetAdaLN, GarmentNetCrossAttn, GarmentNetOutput, GarmentRepairGate, GarmentSlotEncoder, OOTDInjector, PatchDiscriminator, QwenAuxSlotEncoder, QwenControlNet, QwenGarmentEncoder, QwenGarmentNetAdaLN, QwenGarmentNetBlockCopyAdaLN, QwenGarmentNetCrossAttn, QwenLatentRefiner, QwenSlotEnricher, V6Heads, _make_qwen_block, perceptual_loss)
from trainlib.data import (pack_latents, unpack_latents, vae_decode_to_pil, precompute_rough_pils, precompute_prompt_embeds, load_pose_latents, VTONDataset, collate_fn)
from trainlib.builders import (get_vgg_features, _get_v6_heads, _v6_hidden_hook, _get_invalid_token, CrossAttnGarmentProcessor, QwenGarmentBranch, OOTDQwenAttnProcessor, _get_garment_branch, _get_qwen_slot_enricher, _get_qwen_aux_slot, _get_qwen_refiner, _get_garment_encoder, _get_garment_xattn, _get_controlnet, _make_controlnet_block_hook, _proj_out_pre_hook, _get_garment_net, _garment_inject_hook, _get_discriminator, _get_critic, HoleyAttnProcessor, AgnosticCtrlProcessor, build_repair_attn_mask, install_garment_gates)
from trainlib.constants import *


def train_step(transformer,
               pose_cache, prompt_cache,
               vae, person_image_cache, vae_device, img_loss_weight,
               loss_weights,
               batch, device, weight_dtype,
               sigma_beta_alpha=1.0, sigma_beta_beta=1.0,
               global_step=0, max_steps=400):
    # v22: invariance training — duplicate the batch so each sample runs twice
    # with two different agnostic inner-fills (shared noise/sigma). An invariance
    # loss then ties the two outputs together inside the edit region.
    if int(os.environ.get("USE_INVARIANCE", "0")) and transformer.training:
        batch = {k: (torch.cat([v, v], 0) if torch.is_tensor(v)
                     else (list(v) + list(v) if isinstance(v, (list, tuple)) else v))
                 for k, v in batch.items()}
    person   = batch["person_latent"].to(device, dtype=weight_dtype)           # (B, 16, 128, 96)
    agnostic = batch["agnostic_latent"].to(device, dtype=weight_dtype)

    # USE_AGNOSTIC_RANDOM_FILL=1: at TRAIN time, corrupt the agnostic latent so
    # the model learns it can't trust the in-mask content for color.
    # AGNOSTIC_CORRUPT_MODE:
    #   "binary"     (default): replace agnostic IN-MASK entirely with random fill
    #                Hard cut at M_edit.
    #   "soft_trust" : per-pixel A_trust ramp:
    #                core (eroded edit)  → A_trust=0   (full corruption)
    #                boundary band       → A_trust=AGNOSTIC_BOUNDARY_TRUST (~0.5)
    #                keep (outside dil.) → A_trust=1   (no corruption)
    #                output = agnostic * A_trust + corrupt * (1 - A_trust)
    if int(os.environ.get("USE_AGNOSTIC_RANDOM_FILL", "0")) and transformer.training:
        _M_edit_rf = batch["agnostic_mask_latent"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        # Per-element random fill. Modes: 0 white / 1 noise / 2 lowfreq /
        # 3 black / 4 per-channel random constant. For invariance (v22) the two
        # halves of the doubled batch are forced to get DIFFERENT fill modes.
        _Brf = agnostic.shape[0]
        _inv_rf = (int(os.environ.get("USE_INVARIANCE", "0")) and transformer.training
                   and _Brf % 2 == 0)
        _half_rf = _Brf // 2 if _inv_rf else _Brf
        def _mk_fill_rf(mode, like1):  # like1: (16, H, W)
            if mode == 0: return torch.ones_like(like1)
            if mode == 1: return torch.randn_like(like1)
            if mode == 3: return -torch.ones_like(like1)
            if mode == 4:
                return torch.randn(like1.shape[0], 1, 1, device=device,
                                   dtype=weight_dtype).expand_as(like1).contiguous()
            _gh, _gw = 8, 6
            _low = torch.randn(1, like1.shape[0], _gh, _gw, device=device, dtype=weight_dtype)
            return F.interpolate(_low.float(), size=like1.shape[-2:], mode="bilinear",
                                 align_corners=False).to(weight_dtype)[0]
        _NMODES_RF = 5
        # AGNOSTIC_FILL_MODE: pin the fill mode for every sample (e.g. "1" = high-freq
        # Gaussian noise). Default "" = random across all 5 modes. Pinning to 1 avoids the
        # flat-color modes (0 white / 3 black / 4 random-const) the model could copy.
        _fixed_mode_rf = os.environ.get("AGNOSTIC_FILL_MODE", "")
        _fill_rf = torch.empty_like(agnostic)
        _modes_rf = []
        for _b in range(_Brf):
            if _fixed_mode_rf != "":
                _md = int(_fixed_mode_rf)
            elif _inv_rf and _b >= _half_rf:
                _avoid = _modes_rf[_b - _half_rf]
                _cand = [m for m in range(_NMODES_RF) if m != _avoid]
                _md = _cand[int(torch.randint(len(_cand), ()).item())]
            else:
                _md = int(torch.randint(_NMODES_RF, ()).item())
            _modes_rf.append(_md)
            _fill_rf[_b] = _mk_fill_rf(_md, agnostic[_b])
        _mode_rf = os.environ.get("AGNOSTIC_CORRUPT_MODE", "binary")
        if _mode_rf == "soft_trust":
            _k_rf = int(os.environ.get("AGNOSTIC_TRUST_K", "3"))
            _erode_iters = int(os.environ.get("AGNOSTIC_TRUST_ERODE", "2"))
            _dilate_iters = int(os.environ.get("AGNOSTIC_TRUST_DILATE", "2"))
            _bnd_trust = float(os.environ.get("AGNOSTIC_BOUNDARY_TRUST", "0.5"))
            _pad = _k_rf // 2
            # erosion = 1 - max_pool(1 - mask)
            def _dil(x, iters):
                for _ in range(iters):
                    x = F.max_pool2d(x, kernel_size=_k_rf, stride=1, padding=_pad)
                return x.clamp(0, 1)
            _m_core_rf = 1.0 - _dil(1.0 - _M_edit_rf, _erode_iters)   # erosion
            _m_dil_rf  = _dil(_M_edit_rf, _dilate_iters)
            _m_bnd_rf  = (_m_dil_rf - _m_core_rf).clamp(0, 1)
            _m_keep_rf = (1.0 - _m_dil_rf).clamp(0, 1)
            _A_trust = (0.0 * _m_core_rf
                        + _bnd_trust * _m_bnd_rf
                        + 1.0 * _m_keep_rf).clamp(0, 1).to(weight_dtype)
            agnostic = agnostic * _A_trust + _fill_rf * (1.0 - _A_trust)
        else:
            # binary: hard cut at M_edit
            _M_rf = (_M_edit_rf > 0.5).to(weight_dtype)
            agnostic = agnostic * (1 - _M_rf) + _fill_rf * _M_rf

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
    # Optional outward dilation of M_full so the supervised inpaint region
    # extends past the agnostic-v3.2 wipe boundary. Covers the VAE-induced
    # grey fringe at pixel-resolution decoding. (instructions20, 2026-05-26)
    _dilate_mf = int(os.environ.get("DILATE_M_FULL", "0"))
    if _dilate_mf > 0:
        _kmf = 2 * _dilate_mf + 1
        M_full = F.max_pool2d(M_full, kernel_size=_kmf, stride=1, padding=_dilate_mf).clamp(0, 1)
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
    # warp_fp_region: warp claims "garment" but GT says no. The bleed source —
    # exposed via dedicated flow-loss weight (W_WARP_FP) so the model is
    # supervised hardest exactly where the bleed manifests.
    _wm_for_fp = batch["warped_mask"].to(device, dtype=weight_dtype)
    if _wm_for_fp.dim() == 3: _wm_for_fp = _wm_for_fp.unsqueeze(1)
    warp_fp_region = ((_wm_for_fp > 0.5).to(weight_dtype) * (1.0 - garment_prior)).clamp(0, 1)

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

    # Per-sample cached tensors. When pose_cache is empty (full-train mode), use a zero
    # placeholder; SLOT_ORDER excludes "pose" so it's never used downstream.
    if "pose" in SLOT_ORDER:
        pose = torch.stack([pose_cache[iid] for iid in image_ids]).to(device, dtype=weight_dtype)
    else:
        pose = torch.zeros(B, C, H, W, device=device, dtype=weight_dtype)

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
    # L_band: bias half the samples into the bg-defect formation window σ≈0.20-0.42
    # (steps ~14-17/20) so the band loss has signal exactly where the defect forms.
    if int(os.environ.get("USE_BAND_LOSS", "0")) and int(os.environ.get("BAND_SIGMA_BIAS", "1")) and transformer.training:
        _bb = (torch.rand(B, device=device) < 0.5)
        _blo = float(os.environ.get("BAND_SIGMA_LO", "0.20")); _bhi = float(os.environ.get("BAND_SIGMA_HI", "0.42"))
        _bu = torch.empty(B, device=device).uniform_(_blo, _bhi).to(weight_dtype)
        sigma = torch.where(_bb, _bu, sigma)
    _fix_sig = os.environ.get("FIXED_SIGMA", "")              # diagnostics: pin σ (e.g. 0.30)
    if _fix_sig:
        sigma = torch.full((B,), float(_fix_sig), device=device, dtype=weight_dtype)
    sigma_cap = float(os.environ.get("SIGMA_CAP", "1.0"))
    if sigma_cap < 1.0:
        sigma = sigma * sigma_cap
    s = sigma.view(B, 1, 1, 1)
    noise = torch.randn_like(person)
    # v22: the two halves of the invariance-doubled batch share noise & sigma,
    # so the only thing that differs between them is the agnostic inner fill.
    if int(os.environ.get("USE_INVARIANCE", "0")) and transformer.training and B % 2 == 0:
        _hinv = B // 2
        sigma = torch.cat([sigma[:_hinv], sigma[:_hinv]], 0)
        s     = sigma.view(B, 1, 1, 1)
        noise = torch.cat([noise[:_hinv], noise[:_hinv]], 0)

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

    # ── Region-map conditioning injection (USE_REGION_MAP=1) ──
    # Pack 4 region masks at latent res into a 16-ch latent feature via a
    # learnable 1x1 conv adapter and add (gated) to the noisy C_t. Tells the
    # model where to paint (M_core), where to repair (M_repair), what to keep
    # (M_keep), and where the boundary ring is (M_bnd) — separate channels.
    if int(os.environ.get("USE_REGION_MAP", "0")):
        import torch.nn as _rm_nn
        if "state._REGION_ADAPTER" not in globals() or state._REGION_ADAPTER is None:
            state._REGION_ADAPTER = _rm_nn.Conv2d(4, 16, kernel_size=1, bias=True).to(device, dtype=weight_dtype)
            _rm_nn.init.zeros_(state._REGION_ADAPTER.weight); _rm_nn.init.zeros_(state._REGION_ADAPTER.bias)
            state._REGION_ADAPTER.requires_grad_(True)
        if "state._REGION_GATE" not in globals() or state._REGION_GATE is None:
            state._REGION_GATE = _rm_nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
            state._REGION_GATE.requires_grad_(True)
        # Build region masks at latent res (B, 1, H, W)
        _M_full_lat_rm = M_full.float()                                       # (B, 1, H, W)
        if "parse_garment" in batch:
            _M_core_lat_rm = (batch["parse_garment"].to(device, dtype=weight_dtype).float() * _M_full_lat_rm).clamp(0, 1)
        elif "warped_mask" in batch:
            _M_core_lat_rm = (batch["warped_mask"].to(device, dtype=weight_dtype).float() * _M_full_lat_rm).clamp(0, 1)
        else:
            _M_core_lat_rm = _M_full_lat_rm
        _M_repair_lat_rm = (_M_full_lat_rm - _M_core_lat_rm).clamp(0, 1)
        _M_keep_lat_rm   = (1.0 - _M_full_lat_rm).clamp(0, 1)
        _rm_k = int(os.environ.get("REGION_BND_K", "1"))
        _M_full_dil = F.max_pool2d(_M_full_lat_rm, 2*_rm_k+1, 1, _rm_k).clamp(0, 1)
        _M_full_ero = 1.0 - F.max_pool2d(1.0 - _M_full_lat_rm, 2*_rm_k+1, 1, _rm_k).clamp(0, 1)
        _M_bnd_lat_rm = (_M_full_dil - _M_full_ero).clamp(0, 1)
        _region_map = torch.cat([_M_core_lat_rm, _M_repair_lat_rm,
                                  _M_bnd_lat_rm, _M_keep_lat_rm], dim=1).to(weight_dtype)
        _region_feat = state._REGION_ADAPTER(_region_map)                            # (B, 16, H, W)
        _rm_gate = torch.sigmoid(state._REGION_GATE).to(weight_dtype)
        C_t = C_t + _rm_gate * _region_feat

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
    # ── Garment-latent residual enhancement (paradigm #4-ish) ──
    # Add a learnable delta to garment_latent BEFORE packing into slot 3.
    # The frozen LoRA was trained with garment_latent in this slot, so it
    # naturally consumes the enhanced version. Zero-init last conv → starts
    # at delta=0. Trains via flow MSE backprop through the LoRA's frozen
    # conditioning path.
    if (int(os.environ.get("USE_GARMENT_NET", "0"))
            and os.environ.get("GARMENT_NET_MODE", "norm_residual") == "garment_latent_residual"
            and "garment_pixel" in batch):
        gp_imgs = batch["garment_pixel"].to(device, dtype=weight_dtype)              # (B, 3, 1024, 768)
        gnet = _get_garment_net(device, weight_dtype)                                 # GarmentLatentEnhancer
        gar_delta = gnet(gp_imgs)                                                     # (B, 16, 128, 96), init ~ 0
        garment = garment + gar_delta

    agn_p   = pack_latents(agnostic, B, C, H, W)
    pose_p  = pack_latents(pose,     B, C, H, W)
    rough_p = pack_latents(rough,    B, C, H, W)
    gar_p   = pack_latents(garment,  B, C, H, W)

    # v22: replace agnostic-slot tokens inside the edit core with a learned
    # [INVALID_AGNOSTIC] token. A_valid: keep=1, boundary≈0.7, core=0. The core
    # tokens become a non-image symbol — no grey/colour evidence for the model.
    if int(os.environ.get("USE_INVALID_TOKEN", "0")):
        _M_edit_iv = batch["agnostic_mask_latent"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        if _M_edit_iv.dim() == 3: _M_edit_iv = _M_edit_iv.unsqueeze(1)
        _M_edit_iv = _M_edit_iv[:, :1]
        _kiv = int(os.environ.get("INVALID_TOKEN_K", "3")); _piv = _kiv // 2
        _eriv  = int(os.environ.get("INVALID_TOKEN_ERODE", "2"))
        _diliv = int(os.environ.get("INVALID_TOKEN_DILATE", "2"))
        def _dil_iv(x, it):
            for _ in range(it):
                x = F.max_pool2d(x, kernel_size=_kiv, stride=1, padding=_piv)
            return x.clamp(0, 1)
        _core_iv = (1.0 - _dil_iv(1.0 - _M_edit_iv, _eriv)).clamp(0, 1)
        _dilm_iv = _dil_iv(_M_edit_iv, _diliv).clamp(0, 1)
        _bnd_iv  = (_dilm_iv - _core_iv).clamp(0, 1)
        _keep_iv = (1.0 - _dilm_iv).clamp(0, 1)
        _bnd_valid = float(os.environ.get("INVALID_TOKEN_BND_VALID", "0.7"))
        _A_valid_lat = (0.0 * _core_iv + _bnd_valid * _bnd_iv + 1.0 * _keep_iv).clamp(0, 1)
        _A_valid_tok = pack_latents(_A_valid_lat.expand(B, 16, H, W), B, 16, H, W).mean(dim=-1, keepdim=True)
        _e_inv = _get_invalid_token(device, weight_dtype, agn_p.shape[-1])
        agn_p = _A_valid_tok * agn_p + (1.0 - _A_valid_tok) * _e_inv

    # Factorial matrix "agnostic OFF": zero the whole agnostic slot — no keep
    # region, no core. The slot provides zero person/agnostic information.
    if int(os.environ.get("USE_ZERO_AGNOSTIC_SLOT", "0")):
        agn_p = torch.zeros_like(agn_p)

    # USE_REGION_MAP_AGNOSTIC=1: also stamp the region_feat onto the AGNOSTIC
    # slot tokens, so the agnostic slot itself carries explicit "this is the
    # invalid region" info via the same 4-channel mask projection used for C_t.
    # Forces the agnostic slot to be mask-aware rather than relying on the
    # latent's grey-fill statistics.
    if (int(os.environ.get("USE_REGION_MAP", "0"))
            and int(os.environ.get("USE_REGION_MAP_AGNOSTIC", "0"))
            and "state._REGION_ADAPTER" in globals() and state._REGION_ADAPTER is not None):
        # Re-build the region_feat (same recipe as in the C_t injection block).
        _M_full_lat_ag = M_full.float()
        if "parse_garment" in batch:
            _M_core_ag = (batch["parse_garment"].to(device, dtype=weight_dtype).float() * _M_full_lat_ag).clamp(0,1)
        elif "warped_mask" in batch:
            _M_core_ag = (batch["warped_mask"].to(device, dtype=weight_dtype).float() * _M_full_lat_ag).clamp(0,1)
        else:
            _M_core_ag = _M_full_lat_ag
        _M_repair_ag = (_M_full_lat_ag - _M_core_ag).clamp(0, 1)
        _M_keep_ag   = (1.0 - _M_full_lat_ag).clamp(0, 1)
        _rm_k_ag = int(os.environ.get("REGION_BND_K", "1"))
        _dil_ag = F.max_pool2d(_M_full_lat_ag, 2*_rm_k_ag+1, 1, _rm_k_ag).clamp(0, 1)
        _ero_ag = 1.0 - F.max_pool2d(1.0 - _M_full_lat_ag, 2*_rm_k_ag+1, 1, _rm_k_ag).clamp(0, 1)
        _M_bnd_ag = (_dil_ag - _ero_ag).clamp(0, 1)
        _rmap_ag = torch.cat([_M_core_ag, _M_repair_ag, _M_bnd_ag, _M_keep_ag], dim=1).to(weight_dtype)
        _region_feat_ag = state._REGION_ADAPTER(_rmap_ag)              # (B, 16, H, W)
        _rm_gate_ag = torch.sigmoid(state._REGION_GATE).to(weight_dtype)
        _rfeat_packed_ag = pack_latents(_rm_gate_ag * _region_feat_ag, B, C, H, W)
        agn_p = agn_p + _rfeat_packed_ag

    # ── v8 Qwen Slot Enricher (USE_QWEN_SLOT_ENRICH=1) ──
    # A separate Qwen-block encoder processes garment_latent and produces a
    # slot-dim residual. Added to gar_p BEFORE it enters the transformer.
    # Frozen base sees an enriched garment slot via normal joint attention.
    if int(os.environ.get("USE_QWEN_SLOT_ENRICH", "0")):
        _enricher = _get_qwen_slot_enricher(device, weight_dtype)
        _gar_lat = batch["garment_latent"].to(device, dtype=weight_dtype)
        _slot_residual = _enricher(_gar_lat)        # (B, 3072, 64) — zero-init at start
        gar_p = gar_p + _slot_residual
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
                _wsuf = os.environ.get("WARP_SUFFIX", "")
                _lp_alt = os.path.join(BASE, "my_vton_cache/latents", f"{iid}_warped_silhouette_latent{_wsuf}.pt")
                _lp = _lp_alt if _wsuf and os.path.exists(_lp_alt) else \
                      os.path.join(BASE, "my_vton_cache/latents", f"{iid}_warped_silhouette_latent.pt")
                _sil_lats.append(torch.load(_lp, weights_only=True))
            sil = torch.stack(_sil_lats).to(device, dtype=weight_dtype)                   # (B, 16, 128, 96)
            sil_p = pack_latents(sil, B, C, H, W)
        else:
            _wm = batch["warped_mask"].to(device, dtype=weight_dtype)                    # (B, 1, 128, 96)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            # WARP_PERTURB: random morphological perturbation of warped_mask at training
            # to teach the LoRA robustness to imperfect warpnet outputs.
            # Value N means random offset in [-N, N] latent px; positive=erode, negative=dilate.
            _perturb_max = int(os.environ.get("WARP_PERTURB", "0"))
            if _perturb_max > 0 and torch.is_grad_enabled():
                import random as _rand
                _p = _rand.randint(-_perturb_max, _perturb_max)
                if _p > 0:
                    _wm_bin0 = (_wm > 0.5).to(weight_dtype)
                    _wm = -F.max_pool2d(-_wm_bin0, 2*_p+1, 1, _p).clamp(0, 1)
                elif _p < 0:
                    _wm_bin0 = (_wm > 0.5).to(weight_dtype)
                    _wm = F.max_pool2d(_wm_bin0, -2*_p+1, 1, -_p).clamp(0, 1)
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

    # ── Extra-slot garment helper (paradigm #4) ──
    # Append a garment-only encoded slot to the conditioning sequence. The
    # frozen LoRA's joint attention will compute over all tokens including
    # this new one. Zero-init last conv so initial slot ≈ 0 (no early
    # disruption to the frozen base's predictions).
    n_extra_slots = 0
    if (int(os.environ.get("USE_GARMENT_NET", "0"))
            and os.environ.get("GARMENT_NET_MODE", "norm_residual") == "extra_slot"
            and "garment_pixel" in batch):
        gp_imgs = batch["garment_pixel"].to(device, dtype=weight_dtype)
        gnet = _get_garment_net(device, weight_dtype)
        slot_tensors["garment_aux"] = gnet(gp_imgs)                              # (B, 3072, 64)
        n_extra_slots = 1

    # v9: Qwen Aux Slot — Qwen-block encoder produces zero-init slot tokens APPENDED
    # to the conditioning sequence. Frozen transformer's joint attention queries them.
    if int(os.environ.get("USE_QWEN_AUX_SLOT", "0")):
        qaux = _get_qwen_aux_slot(device, weight_dtype)
        _qaux_lat = batch["garment_latent"].to(device, dtype=weight_dtype)
        slot_tensors["garment_aux"] = qaux(_qaux_lat)                            # (B, 3072, 64), zero-init
        n_extra_slots = 1

    slot_order_eff = list(SLOT_ORDER) + (["garment_aux"] if n_extra_slots else [])
    cond_seq = [slot_tensors[n] for n in slot_order_eff]
    hidden = torch.cat([C_p_] + cond_seq, dim=1)                                # (B, 3072 * NUM_SLOTS, 64)
    img_shapes = [[(1, H//2, W//2)] * (NUM_SLOTS + n_extra_slots)] * B

    ATTN_MASK_HOLDER["mask"] = None

    # ── v19/v20/v21: trust-map-weighted agnostic-slot control ──
    # A_trust is high (≈1) for useful agnostic regions (face/hair/keep) and low
    # (≈0.3) for the grey/edit core. We suppress attention to / value-weight of
    # the LOW-trust agnostic tokens only — useful agnostic context is preserved.
    state._AGN_CTRL.clear()
    if int(os.environ.get("USE_AGN_CTRL", "0")) and transformer.training:
        _M_edit_ac = batch["agnostic_mask_latent"].to(device, dtype=weight_dtype).float().clamp(0, 1)
        if _M_edit_ac.dim() == 3: _M_edit_ac = _M_edit_ac.unsqueeze(1)
        _M_edit_ac = _M_edit_ac[:, :1]                          # (B,1,H,W)
        _k_ac  = int(os.environ.get("AGN_TRUST_K", "3"))
        _er_ac = int(os.environ.get("AGN_ERODE", "2"))
        _di_ac = int(os.environ.get("AGN_DILATE", "2"))
        _pad_ac = _k_ac // 2
        def _dil_ac(x, it):
            for _ in range(it):
                x = F.max_pool2d(x, kernel_size=_k_ac, stride=1, padding=_pad_ac)
            return x
        _core_ac = (1.0 - _dil_ac(1.0 - _M_edit_ac, _er_ac)).clamp(0, 1)   # erode(M_edit)
        _dilm_ac = _dil_ac(_M_edit_ac, _di_ac).clamp(0, 1)                 # dilate(M_edit)
        _bnd_ac  = (_dilm_ac - _core_ac).clamp(0, 1)                       # boundary band
        _keep_ac = (1.0 - _dilm_ac).clamp(0, 1)                            # keep (outside)
        def _tok_ac(m):
            return pack_latents(m.expand(B, 16, H, W), B, 16, H, W).mean(dim=-1)   # (B, n_tok)
        _core_t = _tok_ac(_core_ac); _bnd_t = _tok_ac(_bnd_ac); _keep_t = _tok_ac(_keep_ac)
        _agn_seq_pos = (SLOT_ORDER.index("agnostic") + 1) if "agnostic" in SLOT_ORDER else 1
        _tok_per_slot = _core_t.shape[1]
        state._AGN_CTRL["agn_img_start"] = _agn_seq_pos * _tok_per_slot
        state._AGN_CTRL["n_agn"] = _tok_per_slot
        if int(os.environ.get("AGN_KEY_BIAS", "0")):
            _tc = float(os.environ.get("AGN_TRUST_CORE", "0.3"))
            _tb = float(os.environ.get("AGN_TRUST_BND",  "0.85"))
            _tk = float(os.environ.get("AGN_TRUST_KEEP", "1.0"))
            _alpha = float(os.environ.get("AGN_KEY_BIAS_ALPHA", "0.5"))
            _eps   = float(os.environ.get("AGN_TRUST_EPS", "1e-3"))
            _A_tok = (_tc * _core_t + _tb * _bnd_t + _tk * _keep_t).clamp(0, 1)
            state._AGN_CTRL["key_bias_tok"] = _alpha * torch.log(_A_tok + _eps)
        if int(os.environ.get("AGN_V_SCALE", "0")):
            _vc = float(os.environ.get("AGN_VSCALE_CORE", "0.5"))
            _vb = float(os.environ.get("AGN_VSCALE_BND",  "0.9"))
            _vk = float(os.environ.get("AGN_VSCALE_KEEP", "1.0"))
            state._AGN_CTRL["v_scale_tok"] = (_vc * _core_t + _vb * _bnd_t + _vk * _keep_t).clamp(0, 1)

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

    # ── Garment net hook-mode dispatch (norm_residual / adaln / cross_attn) ──
    _gnet_mode = os.environ.get("GARMENT_NET_MODE", "norm_residual")
    state._GARMENT_RESIDUAL_HOLDER.pop("residual", None)
    state._GARMENT_RESIDUAL_HOLDER.pop("gamma", None)
    state._GARMENT_RESIDUAL_HOLDER.pop("beta", None)
    state._GARMENT_RESIDUAL_HOLDER.pop("gate", None)
    state._CROSS_ATTN_HOLDER.pop("K_g", None)
    state._CROSS_ATTN_HOLDER.pop("V_g", None)
    state._CROSS_ATTN_HOLDER.pop("gate", None)
    if (int(os.environ.get("USE_GARMENT_NET", "0")) and "garment_pixel" in batch
            and _gnet_mode in ("norm_residual", "adaln", "cross_attn")):
        # Qwen-style encoder eats garment_latent (B,16,128,96); ConvNet eats garment_pixel.
        _gnet_encoder = os.environ.get("GARMENT_NET_ENCODER", "conv")
        if _gnet_encoder.startswith("qwen"):  # qwen | qwen_blockcopy — both eat garment_latent
            gp_imgs = batch["garment_latent"].to(device, dtype=weight_dtype)
        else:
            gp_imgs = batch["garment_pixel"].to(device, dtype=weight_dtype)              # (B, 3, 1024, 768)
        garment_net = _get_garment_net(device, weight_dtype, hidden_dim=transformer.inner_dim)
        if _gnet_mode == "norm_residual":
            state._GARMENT_RESIDUAL_HOLDER["residual"] = garment_net(gp_imgs)               # (B, 3072, 3072)
        elif _gnet_mode == "adaln":
            gamma, beta = garment_net(gp_imgs)
            state._GARMENT_RESIDUAL_HOLDER["gamma"] = gamma
            state._GARMENT_RESIDUAL_HOLDER["beta"]  = beta
        else:  # cross_attn
            K_g, V_g = garment_net(gp_imgs)                                           # (B, 192, inner_dim)
            state._CROSS_ATTN_HOLDER["K_g"] = K_g
            state._CROSS_ATTN_HOLDER["V_g"] = V_g
        # Spatial gate (shared across modes): packed mask, optionally dilated.
        # GARMENT_NET_GATE_SOURCE: "warped_mask" (default — inference-equivalent) |
        # "target_mask" (train-only — confines garment_net contribution to GT silhouette
        # so it can't inject features in the bleed area; pairs with the user's
        # off-silhouette no-bleed constraint). Inference always uses warped_mask
        # (no GT available); the train-time tighter gate teaches the garment_net to
        # produce sparse output off-silhouette, so the loose inference gate carries
        # near-zero signal in the bleed region.
        if int(os.environ.get("GARMENT_NET_GATE", "1")):
            _gate_src = os.environ.get("GARMENT_NET_GATE_SOURCE", "warped_mask")
            if _gate_src == "target_mask":
                _wm_g = garment_prior  # already binary target_mask, (B,1,H,W) on device/dtype
            else:
                _wm_g = batch["warped_mask"].to(device, dtype=weight_dtype)
                if _wm_g.dim() == 3: _wm_g = _wm_g.unsqueeze(1)
            if int(os.environ.get("GARMENT_NET_GATE_SOFT", "0")):
                _wm_g_b = _wm_g.clamp(0.0, 1.0).to(weight_dtype)   # continuous gate
            else:
                _wm_g_b = (_wm_g > 0.5).to(weight_dtype)            # binary gate
            _gate_dil = int(os.environ.get("GARMENT_NET_GATE_DILATE", "0"))
            if _gate_dil > 0:
                _wm_g_b = F.max_pool2d(_wm_g_b, 2*_gate_dil+1, 1, _gate_dil).clamp(0, 1)
            _wm_g_b = _wm_g_b.expand(B, C, H, W)
            _gate_packed = pack_latents(_wm_g_b, B, C, H, W).mean(dim=-1, keepdim=True)
            if _gnet_mode in ("norm_residual", "adaln"):
                state._GARMENT_RESIDUAL_HOLDER["gate"] = _gate_packed
            else:  # cross_attn — gate is per-image-token
                state._CROSS_ATTN_HOLDER["gate"] = _gate_packed

    # ── OOTD garment branch K,V injection (USE_GARMENT_OOTD=1) v5 ──
    # v5: depth-specific outputs (per inject block reads its own branch depth);
    # mixed mask source (target/warped/jittered) to avoid train/inference gate
    # mismatch; garment-condition dropout for robustness.
    state._GARMENT_BRANCH_HOLDER.clear()
    if int(os.environ.get("USE_GARMENT_OOTD", "0")):
        _gb = _get_garment_branch(device, weight_dtype)
        _gb_lat = batch["garment_latent"].to(device, dtype=weight_dtype)
        _gb_outs = _gb.precompute(_gb_lat)                                            # list of N post-block outputs
        # Depth-specific: zip sorted inject_indices with branch outputs (in order).
        _ootd_inject_keys = sorted(state._OOTD_INJECTORS.keys())
        if int(os.environ.get("GARMENT_OOTD_DEPTH_SPECIFIC", "1")):
            # Each main inject block reads its own branch-depth output
            for i, blk_i in enumerate(_ootd_inject_keys):
                depth_idx = min(i, len(_gb_outs) - 1)
                state._GARMENT_BRANCH_HOLDER[f"h_{blk_i}"] = _gb_outs[depth_idx]
        # Always set final_h fallback (used when depth-specific disabled)
        state._GARMENT_BRANCH_HOLDER["final_h"] = _gb_outs[-1]

        # Mixed mask source: GARMENT_OOTD_GATE_SOURCE=mixed (default in v5)
        # 50% target_mask, 25% warped_mask, 25% jittered warped_mask
        _gx_gate_src = os.environ.get("GARMENT_OOTD_GATE_SOURCE", "target_mask")
        if _gx_gate_src == "mixed":
            r = torch.rand(()).item()
            if r < 0.5:
                _src = "target_mask"
            elif r < 0.75:
                _src = "warped_mask"
            else:
                _src = "warped_jitter"
        else:
            _src = _gx_gate_src

        if _src == "target_mask":
            _ootd_mask = garment_prior
        elif _src == "warped_mask":
            _wm = batch["warped_mask"].to(device, dtype=weight_dtype)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            _ootd_mask = (_wm > 0.5).to(weight_dtype)
        else:  # warped_jitter — small dilate/erode of warped mask (latent px)
            _wm = batch["warped_mask"].to(device, dtype=weight_dtype)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            _wm_b = (_wm > 0.5).to(weight_dtype)
            _r = int(torch.randint(1, 4, ()).item())
            if torch.rand(()).item() < 0.5:
                _ootd_mask = F.max_pool2d(_wm_b, 2*_r+1, 1, _r).clamp(0, 1)        # dilate
            else:
                _ootd_mask = -F.max_pool2d(-_wm_b, 2*_r+1, 1, _r)                   # erode
                _ootd_mask = _ootd_mask.clamp(0, 1)

        _ootd_mask_e = _ootd_mask.expand(B, C, H, W)
        _ootd_p_g_tok = pack_latents(_ootd_mask_e, B, C, H, W).mean(dim=-1, keepdim=True)
        state._GARMENT_BRANCH_HOLDER["p_g_tok"] = _ootd_p_g_tok

        # Garment-condition dropout (per training step)
        _p_drop = float(os.environ.get("GARMENT_OOTD_DROPOUT", "0.0"))
        if _p_drop > 0 and torch.rand(()).item() < _p_drop:
            state._GARMENT_BRANCH_HOLDER["dropout"] = True
        # σ-conditional gating (GARMENT_OOTD_SIGMA_GATE=1): scale contribution by σ.
        # σ_w = clamp((σ - thr) / (1 - thr), 0, 1). Default thr=0.3 → linear ramp 0→1
        # over [0.3, 1.0], full contribution at high noise, zero at very low σ.
        if int(os.environ.get("GARMENT_OOTD_SIGMA_GATE", "0")):
            _thr = float(os.environ.get("GARMENT_OOTD_SIGMA_THR", "0.3"))
            _sigma_w = ((sigma.float() - _thr) / max(1.0 - _thr, 1e-6)).clamp(0.0, 1.0)
            state._GARMENT_BRANCH_HOLDER["sigma_w"] = _sigma_w.to(weight_dtype)

    # ── Garment cross-attn at proj_out (USE_GARMENT_XATTN=1) ──
    # Compute G = encoder(garment_latent) and p_g_tok (per-token garment mask).
    # The proj_out pre-hook will read these from the holder and inject
    # F_g = F + γ * p_g_tok * CrossAttn(F, G) before proj_out runs.
    state._GARMENT_XATTN_HOLDER.clear()
    if int(os.environ.get("USE_GARMENT_XATTN", "0")):
        _gx_enc = _get_garment_encoder(device, weight_dtype)
        _gx_lat = batch["garment_latent"].to(device, dtype=weight_dtype)
        _G = _gx_enc(_gx_lat)                                                          # (B, 3072, 3072)
        # Spatial gate (train: target_mask; infer in inference.py: warped_mask)
        _gx_gate_src = os.environ.get("GARMENT_XATTN_GATE_SOURCE", "target_mask")
        if _gx_gate_src == "target_mask":
            _gx_mask = garment_prior                                                   # (B, 1, H, W)
        else:
            _gx_mask = batch["warped_mask"].to(device, dtype=weight_dtype)
            if _gx_mask.dim() == 3: _gx_mask = _gx_mask.unsqueeze(1)
            _gx_mask = (_gx_mask > 0.5).to(weight_dtype)
        _gx_mask_e = _gx_mask.expand(B, C, H, W)
        _p_g_tok = pack_latents(_gx_mask_e, B, C, H, W).mean(dim=-1, keepdim=True)     # (B, 3072, 1)
        state._GARMENT_XATTN_HOLDER["G"]       = _G
        state._GARMENT_XATTN_HOLDER["p_g_tok"] = _p_g_tok
        state._GARMENT_XATTN_HOLDER["gamma"]   = float(os.environ.get("GARMENT_XATTN_GAMMA", "1.0"))
        state._GARMENT_XATTN_HOLDER["N_C"]     = C_p_.size(1)

    # ── 5_30: Multi-block garment injection (USE_MULTI_BLOCK_INJ=1) ──
    # Inject xattn + adaln-β at multiple transformer blocks. NO gate, NO zero-init.
    # Spatial mask from warped_mask (garment full, boundary medium, skin/bg zero).
    state._MULTI_GAR_HOLDER.clear()
    if int(os.environ.get("USE_MULTI_BLOCK_INJ", "0")) and state._MULTI_GAR_INJECTION is not None:
        # Reuse the garment_encoder G if already computed for xattn; otherwise compute now
        if "G" in state._GARMENT_XATTN_HOLDER:
            _G_multi = state._GARMENT_XATTN_HOLDER["G"]
        else:
            _gx_enc_mb = _get_garment_encoder(device, weight_dtype)
            _gx_lat_mb = batch["garment_latent"].to(device, dtype=weight_dtype)
            _G_multi = _gx_enc_mb(_gx_lat_mb)
        # Build spatial mask from warped_mask
        _wm_mb = batch.get("warped_mask")
        if _wm_mb is not None:
            _wm_mb = _wm_mb.to(device, dtype=weight_dtype)
            if _wm_mb.dim() == 3: _wm_mb = _wm_mb.unsqueeze(1)
            _wm_b_mb = (_wm_mb > 0.5).to(weight_dtype).expand(B, C, H, W)
            _g_tok = pack_latents(_wm_b_mb, B, C, H, W).mean(dim=-1, keepdim=True)
            # M_full (edit region): get from batch
            _M_ag_mb = batch["agnostic_mask_latent"].to(device, dtype=weight_dtype)
            if _M_ag_mb.dim() == 3: _M_ag_mb = _M_ag_mb.unsqueeze(1)
            _M_ag_mb = (_M_ag_mb > 0.5).to(weight_dtype).expand(B, C, H, W)
            _m_tok = pack_latents(_M_ag_mb, B, C, H, W).mean(dim=-1, keepdim=True)
            # garment=full, boundary=0.4, skin/bg=0 (defaults; tunable)
            _mb_gw  = float(os.environ.get("MULTI_BLOCK_GARMENT_W",  "1.0"))
            _mb_bw  = float(os.environ.get("MULTI_BLOCK_BOUNDARY_W", "0.4"))
            _mb_skw = float(os.environ.get("MULTI_BLOCK_SKIN_W",     "0.0"))
            _mb_bgw = float(os.environ.get("MULTI_BLOCK_BG_W",       "0.0"))
            _boundary_tok = (_m_tok * (1 - _g_tok)).clamp(0, 1)
            _spatial_mask = (_mb_gw * _g_tok + _mb_bw * _boundary_tok).clamp(0, 1)
        else:
            _spatial_mask = torch.ones(B, C_p_.size(1), 1, device=device, dtype=weight_dtype)
        state._MULTI_GAR_HOLDER["G"] = _G_multi
        state._MULTI_GAR_HOLDER["spatial_mask"] = _spatial_mask
        state._MULTI_GAR_HOLDER["N_C"] = C_p_.size(1)
        state._MULTI_GAR_HOLDER["xattn_spatial_w"] = float(os.environ.get("MULTI_BLOCK_XATTN_W", "1.0"))
        state._MULTI_GAR_HOLDER["adaln_spatial_w"] = float(os.environ.get("MULTI_BLOCK_ADALN_W", "1.0"))
        # 5_31 FULL chain: also stash raw garment_latent so the chain pre-hook can run it
        if int(os.environ.get("USE_MULTI_BLOCK_FULL", "0")):
            _gx_lat_full = batch["garment_latent"].to(device, dtype=weight_dtype)
            if _gx_lat_full.dim() == 3: _gx_lat_full = _gx_lat_full.unsqueeze(0)
            state._MULTI_GAR_HOLDER["garment_latent"] = _gx_lat_full

    # ── ControlNet branch (USE_CONTROLNET=1) ──
    # Run agnostic through ControlNet, store per-block residuals in holder.
    # Block hooks installed at setup time consume these residuals.
    state._CONTROLNET_HOLDER.clear()
    if int(os.environ.get("USE_CONTROLNET", "0")) and "agnostic_latent" in batch:
        cnet = _get_controlnet(device, weight_dtype)
        agn_lat_full = batch["agnostic_latent"].to(device, dtype=weight_dtype)
        if agn_lat_full.dim() == 3: agn_lat_full = agn_lat_full.unsqueeze(0)
        cn_residuals = cnet(agn_lat_full)  # list of N (B, 3072, 3072)
        # Map residuals to block indices via env var (default 0..N-1)
        cn_csv = os.environ.get("CONTROLNET_INJECT_BLOCKS", "")
        if cn_csv:
            cn_blocks = [int(s) for s in cn_csv.split(",") if s.strip()]
        else:
            cn_blocks = list(range(len(cn_residuals)))
        assert len(cn_blocks) == len(cn_residuals), \
            f"CONTROLNET_INJECT_BLOCKS must have {len(cn_residuals)} indices"
        for blk_i, res in zip(cn_blocks, cn_residuals):
            state._CONTROLNET_HOLDER[blk_i] = res
        state._CONTROLNET_HOLDER["N_C"] = C_p_.size(1)

    # ── Main forward (factored into a callable so SOAR can run it twice) ──
    def _fwd(C_lat_in, sig_in):
        _Cp = pack_latents(C_lat_in, B, C, H, W)
        _hid = torch.cat([_Cp] + cond_seq, dim=1)
        _o = transformer(hidden_states=_hid, timestep=sig_in, encoder_hidden_states=pe_batch,
                         encoder_hidden_states_mask=pm_batch, img_shapes=img_shapes,
                         txt_seq_lens=txt_seq_lens, return_dict=False)[0]
        return _o[:, :_Cp.size(1), :]

    # ── DPO (USE_DPO): Direct Preference Optimization vs frozen v01 "loser" latents. ──
    # y_win = GT person latent; y_lose = frozen v01 deployed-output latent (static, on disk).
    # Noise BOTH with the SAME noise & sigma, run two GRAD forwards, compute per-branch flow
    # MSE (L_win, L_lose), and minimize  -log sigmoid(beta*(L_lose - L_win))  — pushing the
    # model to reconstruct the winner and AWAY from the haloed loser. Backprop ONLY L_dpo.
    # Returns early (skips v6/band/SOAR/img machinery). Guarded to training so val (5 reserved,
    # which have no loser file) falls through to the normal flow path.
    if int(os.environ.get("USE_DPO", "0")) and transformer.training:
        _beta = float(os.environ.get("DPO_BETA", "0.1"))
        _ldir = os.environ.get("DPO_LOSER_CACHE_DIR",
                               "/home/link/Desktop/Code/fashion gen testing/my_vton_cache/dpo_loser_v01")
        _yl = []
        for _iid in image_ids:
            _t = torch.load(os.path.join(_ldir, f"{_iid}_deployed_v01_latent.pt"), weights_only=True)
            _yl.append(_t.unsqueeze(0) if _t.dim() == 3 else _t)
        y_lose = torch.cat(_yl, dim=0).to(device, dtype=weight_dtype)              # (B,16,128,96)
        # Same noise & sigma for both branches — only the data point differs (pure-noise init).
        C_win   = person + s * (noise - person)
        C_lose  = y_lose + s * (noise - y_lose)
        vt_win  = pack_latents(M_full * (noise - person), B, C, H, W)
        vt_lose = pack_latents(M_full * (noise - y_lose),  B, C, H, W)
        pred_win  = _fwd(C_win,  sigma)
        pred_lose = _fwd(C_lose, sigma)
        L_win  = ((pred_win.float()  - vt_win.float())  ** 2).mean(dim=(1, 2))     # (B,)
        L_lose = ((pred_lose.float() - vt_lose.float()) ** 2).mean(dim=(1, 2))     # (B,)
        # Margin CAP: clamp the preference gap. Once L_lose-L_win ≥ cap the loser is
        # "repelled enough" and gets no further push (clamp → zero grad on L_lose), so the
        # unbounded MSE margin can't run away and corrupt the shared weights. 0 = uncapped.
        _mcap = float(os.environ.get("DPO_MARGIN_CAP", "0"))
        _margin = (L_lose - L_win)
        if _mcap > 0:
            _margin = _margin.clamp(max=_mcap)
        L_dpo  = -F.logsigmoid(_beta * _margin).mean()                            # = -log σ(β·gap)
        # SFT anchor on the winner: reference-free DPO has no KL pin, so the model can
        # "reward-hack" the margin by inflating L_lose, which corrupts the SHARED weights
        # and degrades L_win too. Adding λ·L_win keeps GT reconstruction anchored while DPO
        # repels the loser. (DPO_SFT_WEIGHT=0 → pure spec formula.)
        _sftw = float(os.environ.get("DPO_SFT_WEIGHT", "1.0"))
        L_total = L_dpo + _sftw * L_win.mean()
        _gap = (L_lose - L_win).mean()
        return L_total, {"dpo": float(L_dpo.detach()), "Lwin": float(L_win.mean().detach()),
                         "Llose": float(L_lose.mean().detach()), "gap": float(_gap.detach()),
                         "sigma": float(sigma.float().mean().detach())}

    # ── SOAR (USE_SOAR): off-trajectory self-correction for exposure bias. ──
    # Instead of supervising on a forward-noised GT state C_t (which the model never sees at
    # inference), take ONE stop-grad Euler step σ1→σ2 with the current model to land on the
    # model's OWN off-trajectory state, re-noise it slightly, then let the standard flow
    # (+ img/recon/band) losses steer THAT state back to the clean x0. 2 forwards/step.
    _soar_p = float(os.environ.get("SOAR_PROB", "1.0"))
    _do_soar = int(os.environ.get("USE_SOAR", "0")) and (_soar_p >= 1.0 or float(torch.rand(1).item()) < _soar_p)
    if _do_soar:
        _dsig = float(os.environ.get("SOAR_DSIGMA", "0.05"))     # Euler step size in σ
        _rn   = float(os.environ.get("SOAR_RENOISE", "0.10"))    # re-noise scale (∝ σ_final)
        _smin = float(os.environ.get("SOAR_SIGMA_MIN", "0.05"))
        _ksteps = int(os.environ.get("SOAR_KSTEPS", "1"))        # k stop-grad Euler steps (k>1 = more drift)
        if int(os.environ.get("SOAR_FORCE_START", "0")):
            # AGGRESSIVE deployed-style start: re-init at a HIGH sigma with the INFERENCE init
            # (real context outside the hole, pure noise inside the agnostic), so the K-step
            # stop-grad rollout traverses the halo-forming sigma band and reproduces the from-
            # noise edge ring the model only exhibits at deployment (verified: a loss on the
            # true-noised state is blind to it — gem=0.019 vs deployed |d|=30).
            _ss0 = float(os.environ.get("SOAR_START_SIGMA", "0.95"))
            sigma = torch.full_like(sigma, _ss0)
            s = sigma.view(B, 1, 1, 1)
            _dep_noise = torch.randn_like(person)
            C_t = ((1.0 - M_full) * agnostic + M_full * _dep_noise).to(person.dtype)
        # PROTECT GARMENT (SOAR_PROTECT_GARMENT): anchor the garment region to the TRUE-noised GT
        # throughout the rollout so ONLY the background drifts/reproduces the halo. Without this the
        # from-noise rollout regenerates the garment from noise → smears fine detail (Adidas stripes).
        # Garment then gets NORMAL flow supervision (detail preserved); only the bg gets the SOAR state.
        # DIL=0 anchors just the garment silhouette so the 0-15px edge ring OUTSIDE it still drifts.
        _protect = int(os.environ.get("SOAR_PROTECT_GARMENT", "0"))
        if _protect:
            # Anchor the WHOLE PERSON in the hole (garment + skin + body), not just garment — the
            # SOAR also smears revealed skin (neck speckle), not only stripes. Person = M_full ∩ ¬parse_bg.
            # ONLY the revealed background (M_full ∩ parse_bg) drifts → reproduces the halo for the edge loss.
            _Mf = M_full if M_full.dim() == 4 else M_full.unsqueeze(1)
            if int(os.environ.get("SOAR_PROTECT_PERSON", "0")) and "parse_bg" in batch:
                _pb = batch["parse_bg"].to(person.device, person.dtype)
                if _pb.dim() == 3: _pb = _pb.unsqueeze(1)
                if tuple(_pb.shape[-2:]) != tuple(person.shape[-2:]):
                    _pb = F.interpolate(_pb, size=person.shape[-2:], mode="nearest")
                _gp = (_Mf.to(person.dtype) * (1.0 - (_pb > 0.5).to(person.dtype))).clamp(0, 1)
            else:
                _gp = garment_prior.to(person.dtype)
                if _gp.dim() == 3: _gp = _gp.unsqueeze(1)
            _prd = int(os.environ.get("SOAR_PROTECT_DIL", "0"))
            if _prd > 0: _gp = F.max_pool2d(_gp, 2*_prd+1, 1, _prd).clamp(0, 1)
            _gnoise = torch.randn_like(person)
            def _anchor_gar(_xx, _sv):
                return (1.0 - _gp) * _xx + _gp * (person + _sv.view(B, 1, 1, 1) * (_gnoise - person))
            C_t = _anchor_gar(C_t, sigma)
        with torch.no_grad():
            _x = C_t; _sig_cur = sigma                           # walk the model's OWN trajectory k steps
            for _ks in range(_ksteps):
                _v = unpack_latents(_fwd(_x, _sig_cur), B, C, H, W)
                _sig_next = (_sig_cur - _dsig).clamp(min=_smin)
                _x = _x - (_sig_cur.view(B, 1, 1, 1) - _sig_next.view(B, 1, 1, 1)) * _v
                if _protect: _x = _anchor_gar(_x, _sig_next)
                _sig_cur = _sig_next
            _sig2 = _sig_cur
            _s2 = _sig2.view(B, 1, 1, 1)
            _x_hat = _x + _rn * _s2 * torch.randn_like(_x)        # re-noise slightly (∝ σ_final, bounded v_target)
        # CRITICAL: the no_grad forward above poisoned the autocast weight-cache with grad-less
        # bf16 casts; clear it so the grad forward below re-casts the weights WITH grad (else the
        # LoRA receives zero gradient under autocast and the model never trains).
        torch.clear_autocast_cache()
        # supervise THIS off-trajectory state at σ2, steering back to the clean person latent
        C_t = _x_hat.detach(); sigma = _sig2; s = _s2
        v_target = M_full * (C_t - person) / _s2.clamp(min=1e-3)
        vt_p = pack_latents(v_target, B, C, H, W)
        C_p_ = pack_latents(C_t, B, C, H, W)
        pred_C = _fwd(C_t, sigma)                                 # grad forward at the off-traj state
    else:
        pred_C = _fwd(C_t, sigma)                                                  # (B, 3072, 64)

    # ── v11: Qwen Latent Refiner (USE_QWEN_REFINER=1) ──
    # Post-pred residual on garment region. Frozen base produces pred_C; refiner
    # sees pred_C as latent-space output, takes garment_latent + warped_mask, and
    # produces a refinement residual added back ONLY in garment region (warped_mask).
    if int(os.environ.get("USE_QWEN_REFINER", "0")):
        _refiner = _get_qwen_refiner(device, weight_dtype)
        # Unpack pred_C to latent-space (B, 16, 128, 96)
        _pred_lat = unpack_latents(pred_C, B, C, H, W)
        _gar_lat_full = batch["garment_latent"].to(device, dtype=weight_dtype)
        _wm_full = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_full.dim() == 3: _wm_full = _wm_full.unsqueeze(1)
        _wm_full_b = (_wm_full > 0.5).to(weight_dtype)
        _residual_packed = _refiner(_pred_lat, _gar_lat_full, _wm_full_b)         # (B, 3072, 64)
        # Gate residual by garment region mask (per-token packed)
        _wm_e = _wm_full_b.expand(B, C, H, W)
        _wm_packed = pack_latents(_wm_e, B, C, H, W).mean(dim=-1, keepdim=True)   # (B, 3072, 1)
        pred_C = pred_C + _residual_packed * _wm_packed.to(pred_C.dtype)

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

    # ── L_hf_garment: high-pass L1 on garment region — force texture matching ──
    # Direct attack on the "low-frequency garment fill" failure mode. Computes
    # high-pass = (img - gaussian_blur(img)), then L1 on the M_g (parse_garment ∩
    # agnostic_mask) region between pred and gt. Optimization for blur-vs-detail.
    L_hf_garment = torch.tensor(0.0, device=device)
    lambda_hf_g = float(os.environ.get("LAMBDA_HF_GARMENT", "0.0"))
    if lambda_hf_g > 0 and "parse_garment" in batch:
        from torchvision.transforms.functional import gaussian_blur as _gb_hf
        pg_lat = batch["parse_garment"].to(vae_device, weight_dtype)
        am_lat = batch["agnostic_mask_latent"].to(vae_device, weight_dtype)
        if pg_lat.dim() == 3: pg_lat = pg_lat.unsqueeze(1)
        if am_lat.dim() == 3: am_lat = am_lat.unsqueeze(1)
        M_g_lat = (pg_lat * am_lat).clamp(0, 1)
        M_g_pix = F.interpolate(M_g_lat.float(), size=(Hi, Wi), mode="nearest").to(vae_device, weight_dtype)
        k_hf = int(os.environ.get("HF_KERNEL", "7"))
        sig_hf = float(os.environ.get("HF_SIGMA", "3.0"))
        pred_blur = _gb_hf(pred_img.float(), [k_hf, k_hf], sig_hf)
        gt_blur   = _gb_hf(person_imgs.float(), [k_hf, k_hf], sig_hf)
        hp_pred = pred_img.float() - pred_blur
        hp_gt   = person_imgs.float() - gt_blur
        diff = (hp_pred - hp_gt).abs().mean(dim=1, keepdim=True)
        L_hf_garment = (diff * M_g_pix).sum() / (M_g_pix.sum() + 1e-6)
        L_hf_garment = L_hf_garment.to(L_flow.device, dtype=torch.float32)

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

    # ── L_critic_adv: garment-CONDITIONED patch critic (LAMBDA_CRITIC_ADV>0) ──
    # A plain realism critic accepts a realistic-but-WRONG detail (fake wrist logo). This
    # one scores each patch GIVEN the garment + person/edit context (cross-attn to garment
    # features), trained with wrong-garment + corruption negatives so it MUST use the
    # garment. σ-gated to visible detail (active σ<0.7, full strength σ<0.4).
    L_critic_adv = torch.tensor(0.0, device=device)
    _lam_crit = float(os.environ.get("LAMBDA_CRITIC_ADV", "0.0"))
    _crit_pretrain = int(os.environ.get("CRITIC_PRETRAIN", "0"))
    if _lam_crit > 0 or _crit_pretrain:
        from conditioned_critic import corrupt_garment_region, CORRUPTIONS, HARMLESS, _gauss_blur
        Cr, Cr_opt = _get_critic(vae_device, weight_dtype)
        wd = weight_dtype
        gar_ref = batch["garment_latent"].to(vae_device, wd)          # (B,16,128,96)
        def _pix(key, dil=0):
            t = batch[key].to(vae_device, wd)
            if t.dim() == 3: t = t.unsqueeze(1)
            t = (t > 0.5).float()
            if dil: t = F.max_pool2d(t, 2 * dil + 1, 1, dil).clamp(0, 1)
            return F.interpolate(t, size=(Hi, Wi), mode="nearest")
        m_model = _pix("agnostic_mask_latent", dil=int(os.environ.get("DILATE_M_FULL", "3")))
        m_gar   = (_pix("warped_mask") * _pix("agnostic_mask_latent")).clamp(0, 1)
        m_skin  = (_pix("parse_skin") * m_model).clamp(0, 1)
        m_bg    = (_pix("parse_bg") * m_model).clamp(0, 1)
        person  = 1.0 - _pix("parse_bg")                              # 1=person, 0=bg
        m_be    = ((F.max_pool2d(person, 5, 1, 2) + F.max_pool2d(-person, 5, 1, 2)) * m_model).clamp(0, 1)
        masks5  = torch.cat([m_model, m_gar, m_skin, m_bg, m_be], dim=1)
        agn_ctx = (person_imgs * (1.0 - m_model)).detach().to(wd)     # real kept pixels, edit removed
        gt_img  = person_imgs.detach().to(wd)
        pred_d  = pred_img.detach().to(wd)
        _ww = float(os.environ.get("CRITIC_WRONG_GAR_W", "1.0"))
        _cw = float(os.environ.get("CRITIC_CORRUPT_W", "1.0"))
        _rkw = float(os.environ.get("CRITIC_RANK_W", "0.5"))        # localized model-fake RANKING
        _rmargin = float(os.environ.get("CRITIC_RANK_MARGIN", "1.0"))
        _topk = float(os.environ.get("CRITIC_BAD_TOPK", "0.15"))    # top fraction of M_model = "bad"
        _zf = float(os.environ.get("CRITIC_BAD_Z", "1.5"))          # AND outlier floor (mean+z·std)
        _hpw = float(os.environ.get("CRITIC_HARMLESS_W", "1.0"))    # harmless-drape POSITIVE
        # ----- train the critic (SPATIALLY-MASKED hinge) -----
        # Each negative is "fake" ONLY inside its invalid region; the clean remainder of
        # M_model stays "real". A global label would tell the patch critic that the ~95%
        # of a locally-corrupted image that is identical to GT is fake — contradicting the
        # real example and collapsing the output to a constant. Patch critic ⇒ patch-masked
        # supervision (the corruption region, the garment for wrong-garment, etc.).
        def _hreal(L, R): return (F.relu(1.0 - L) * R).sum() / (R.sum() + 1e-6)
        def _hfake(L, R): return (F.relu(1.0 + L) * R).sum() / (R.sum() + 1e-6)
        def _rmean(L, R): return float((L.detach() * R).sum() / (R.sum() + 1e-6))
        with torch.enable_grad():
            Cr.train()
            d_real = Cr(gt_img, agn_ctx, masks5, gar_ref)
            _hw = d_real.shape[-2:]
            def _R(m): return (F.interpolate(m, size=_hw, mode="area") > 0.3).float()
            R_model = _R(m_model); R_gar = _R(m_gar)
            d_fake = Cr(pred_d, agn_ctx, masks5, gar_ref)
            d_loss = _hreal(d_real, R_model)
            # ----- LOCALIZED model-fake via PAIRED RANKING (not wholesale fake) -----
            # Wholesale "all of M_model is fake" is contradictory: v01≈GT, so it tells the
            # critic GT and pred are simultaneously real and fake on identical pixels, and it
            # over-fires on valid drape. Instead: build a DETACHED model-error mask H_bad of
            # the top structured/local pred-vs-GT disagreements (drape/shading suppressed),
            # and only ask D(GT) > D(pred) there. The critic never declares the whole pred
            # fake — it learns "in THIS local spot, GT is more valid than v01's output."
            if _rkw > 0:
                with torch.no_grad():
                    _pf = pred_d.float(); _gf = gt_img.float()
                    _err = (_pf - _gf).abs().mean(1, keepdim=True)               # raw L1
                    _err_lf = _gauss_blur(_err, k=31, s=12.0)                    # smooth drape/shading
                    _err_hf = (_err - _err_lf).clamp(min=0)                      # structured/local part
                    def _gm(x):
                        gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
                        gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
                        return (gx.abs() + gy.abs()).mean(1, keepdim=True)
                    _edge = (_gm(_pf) - _gm(_gf)).abs()                          # edge/structure mismatch
                    _Hraw = (_err_hf + 0.5 * _edge) * m_model                    # only in M_model
                    # H_bad = genuine OUTLIER structured errors within M_model. Require both
                    # top-k AND (mean + z·std): an artifact-free sample has low std → almost
                    # no H_bad → no false "rank GT>pred" signal where the model is actually
                    # correct. Only real, localized artifacts (logo/ring/cuff) get ranked.
                    H_bad = torch.zeros_like(_Hraw)
                    for _b in range(_Hraw.shape[0]):
                        _mb = m_model[_b, 0] > 0.5
                        if int(_mb.sum()) < 16: continue
                        _vals = _Hraw[_b, 0][_mb]
                        _thr_k = torch.quantile(_vals, 1.0 - _topk)
                        _thr_o = _vals.mean() + _zf * _vals.std()
                        _thr = torch.maximum(_thr_k, _thr_o)
                        H_bad[_b, 0] = ((_Hraw[_b, 0] > _thr) & _mb & (_Hraw[_b, 0] > 1e-4)).float()
                R_bad = _R(H_bad)
                # paired ranking: GT should beat pred by a margin, ONLY in H_bad
                L_rank = (F.relu(_rmargin - (d_real - d_fake)) * R_bad).sum() / (R_bad.sum() + 1e-6)
                d_loss = d_loss + _rkw * L_rank
            # harmless-variation POSITIVES (battery): drape (small+big), shading, fabric blur —
            # all VALID, labeled real. Teaches the critic to tolerate natural garment variation
            # so it doesn't fire on acceptable change. Labeled real over the WHOLE garment (R_gar
            # ∪ M_model) since that's exactly where it was over-firing.
            if _hpw > 0:
                _gtf = gt_img.float()
                for _hk, _hfn in HARMLESS.items():
                    _hi, _ = _hfn(_gtf, m_gar)
                    d_harm = Cr(_hi.to(wd), agn_ctx, masks5, gar_ref)
                    d_loss = d_loss + _hpw * (_hreal(d_harm, R_gar) + _hreal(d_harm, R_model))
            if len(state._GAR_BUFFER) >= 2 and _ww > 0:
                wrong = None
                for cand in reversed(state._GAR_BUFFER):
                    cd = cand.to(vae_device, wd)
                    if cd.shape == gar_ref.shape[1:] and not torch.equal(cd, gar_ref[0]):
                        wrong = cd.unsqueeze(0); break
                if wrong is not None:
                    d_wrong = Cr(gt_img, agn_ctx, masks5, wrong)
                    # GT image is real but incompatible with the WRONG garment → garment
                    # patches fake; skin/bg unaffected → stay real.
                    d_loss = d_loss + _ww * (_hfake(d_wrong, R_gar)
                                             + _hreal(d_wrong, (R_model * (1 - R_gar)).clamp(0, 1)))
            if _cw > 0:
                gtf = gt_img.float()
                # full corruption battery in pretrain; one rotating type otherwise
                if _crit_pretrain:
                    _ckinds = list(CORRUPTIONS.keys())
                else:
                    _ckinds = [list(CORRUPTIONS.keys())[global_step % len(CORRUPTIONS)]]
                _cmetrics = {}
                for _ck in _ckinds:
                    _ci, _creg = CORRUPTIONS[_ck](gtf, m_model, m_gar, m_skin, m_bg, m_be)
                    d_corr = Cr(_ci.to(wd), agn_ctx, masks5, gar_ref)
                    R_c = _R(_creg.to(wd)); R_clean = (R_model * (1 - R_c)).clamp(0, 1)
                    # fake INSIDE the corruption; the rest of M_model is still real GT.
                    d_loss = d_loss + _cw * (_hfake(d_corr, R_c) + _hreal(d_corr, R_clean))
                    _cmetrics[_ck] = _rmean(d_corr, R_c) if float(R_c.sum()) > 0 else float("nan")
            Cr_opt.zero_grad(); d_loss.backward(); Cr_opt.step()
        state._GAR_BUFFER.append(gar_ref[0].detach().to("cpu", torch.float32))
        state._LAST_CRIT = {"real": _rmean(d_real, R_model), "fake": _rmean(d_fake, R_model)}
        try: state._LAST_CRIT["rankd"] = _rmean(d_real - d_fake, R_bad)   # GT−pred margin in H_bad (→ margin)
        except Exception: pass
        try: state._LAST_CRIT["harm"] = _rmean(d_harm, R_gar)   # garment region (where it over-fired)
        except Exception: pass
        try: state._LAST_CRIT["wrong"] = _rmean(d_wrong, R_gar)
        except Exception: pass
        try: state._LAST_CRIT.update({f"c_{k}": v for k, v in _cmetrics.items()})
        except Exception: pass
        # ----- generator loss: pull pred toward garment-valid (σ-gated, masked to M_model) -----
        # Skipped in CRITIC_PRETRAIN (Phase 1: critic only, generator untouched).
        _sg = float(sigma.float().mean())
        crit_gate = 0.0 if (_sg >= 0.7 or _crit_pretrain) else (1.0 if _sg < 0.4 else (0.7 - _sg) / 0.3)
        if crit_gate > 0:
            Cr.eval()
            for p in Cr.parameters(): p.requires_grad_(False)
            d_g = Cr(pred_img.to(wd), agn_ctx, masks5, gar_ref)
            R_g = (F.interpolate(m_model, size=d_g.shape[-2:], mode="area") > 0.3).float()
            L_critic_adv = (crit_gate * (-(d_g * R_g).sum() / (R_g.sum() + 1e-6))).to(L_flow.device, dtype=torch.float32)
            for p in Cr.parameters(): p.requires_grad_(True)

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

    loss = (w_flow * L_flow
            + lambda_band * g_band * L_band
            + lambda_inv * L_inv
            + lambda_ab * L_ab
            + lambda_chroma * L_chroma
            + lambda_chroma_ratio * L_chroma_ratio
            + lambda_ab_direction * L_ab_direction
            + L_edge
            + img_loss_weight * L_img
            + lambda_recon * L_recon_ub
            + lambda_antisludge * L_antisludge
            + lambda_tv * L_tv
            + lambda_alloc * L_early_alloc
            + lambda_broad_ratio * L_early_broad
            + lambda_percep_eff * L_percep
            + lambda_anti_rough * L_anti_rough
            + lambda_anti_grey * L_anti_grey
            + lambda_adv * L_adv
            + _lam_crit * L_critic_adv
            + lambda_late_shell * L_late_shell
            + lambda_tv_edge * L_tv_edge
            + lambda_tv_ring * L_tv_ring
            + lambda_no_bg * L_no_bg_leak
            + float(os.environ.get("LAMBDA_V6_REPAIR", "1.0")) * L_repair_v6
            + float(os.environ.get("LAMBDA_V6_ROUTE",  "0.5")) * L_route_v6
            + _w_fp * L_warp_fp
            + float(os.environ.get("LAMBDA_BETA_L2", "0.0")) * L_beta_l2
            + lambda_hf_g_eff * L_hf_garment
            + lambda_person_halo_keep * L_person_halo_keep
            + lambda_bg_shell_keep * L_bg_shell_keep
            + lambda_bg_shell_ab * L_bg_shell_ab
            + lambda_inside_ab * L_inside_edit_halo_ab
            + lambda_inside_hf * L_inside_edit_halo_hf
            + lambda_mid_ab * L_mid_ab
            + lambda_mid_hf * L_mid_hf
            + _lambda_gcl1 * L_garment_crop_l1
            + _lambda_rs * L_repair_skin_img
            + _lambda_rb * L_repair_bg_img
            + _lambda_bg_chroma * L_bg_chroma
            + _lambda_bg_field * L_bg_field
            + _lam_bel1 * L_body_edge_l1
            + _lam_belg * L_body_edge_grad
            + _lambda_bnd * L_boundary_l1
            + _w_gem * L_garment_edge
            + _w_brs * L_bg_route_self
            + L_v6_boundary)

    return loss, {"_T": ({"flow": w_flow * L_flow, "band": lambda_band * g_band * L_band,
                           "recon": lambda_recon * L_recon_ub} if int(os.environ.get("RETURN_LOSS_TENSORS", "0")) else None),
                  "flow": L_flow.item(),
                  "band": L_band.item(), "bandg": g_band.item(), "bandhp": L_band_hp_diag.item(),
                  "inv": L_inv.item(),
                  "ab": L_ab.item(),
                  "chroma": L_chroma.item(),
                  "cratio": L_chroma_ratio.item(),
                  "abdir": L_ab_direction.item(),
                  "edge": L_edge.item(),
                  "gem": L_garment_edge.item(),
                  "brs": L_bg_route_self.item(),
                  "img": L_img.item(),
                  "recon": L_recon_ub.item(),
                  "anti": L_antisludge.item(),
                  "tv": L_tv.item(),
                  "alloc": L_early_alloc.item(),
                  "broad": L_early_broad.item(),
                  "percep": L_percep.item(),
                  "hf_g": L_hf_garment.item(),
                  "warp_fp": L_warp_fp.item(),
                  "beta_l2": L_beta_l2.item(),
                  "antirough": L_anti_rough.item(),
                  "antigrey": L_anti_grey.item(),
                  "adv": L_adv.item(),
                  "critic": L_critic_adv.item(),
                  "late": L_late_shell.item(),
                  "phk": L_person_halo_keep.item(),
                  "bsk": L_bg_shell_keep.item(),
                  "bsab": L_bg_shell_ab.item(),
                  "iheab": L_inside_edit_halo_ab.item(),
                  "ihehf": L_inside_edit_halo_hf.item(),
                  "midab": L_mid_ab.item(),
                  "midhf": L_mid_hf.item(),
                  # 5_30 new losses (instructions7)
                  "bnd": L_boundary_l1.item(),
                  "bgc": L_bg_chroma.item(),
                  "bgf": L_bg_field.item(),
                  "bel1": L_body_edge_l1.item(), "belg": L_body_edge_grad.item(),
                  "img_g": state._LIMG_PARTS["img_g"], "img_s": state._LIMG_PARTS["img_s"],
                  "img_b": state._LIMG_PARTS["img_b"], "img_other": state._LIMG_PARTS["img_other"],
                  "img_k": state._LIMG_PARTS["img_k"], "img_ub": state._LIMG_PARTS["img_ub"],
                  "gcl": L_garment_crop_l1.item(),
                  "rbi": L_repair_bg_img.item(),
                  "rsi": L_repair_skin_img.item(),
                  "tve": L_tv_edge.item(),
                  "rou": L_route_v6.item(),
                  "rep": L_repair_v6.item(),
                  "sigma": sigma.mean().item()}
