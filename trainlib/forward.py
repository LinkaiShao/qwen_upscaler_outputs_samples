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
from trainlib.conditioning.masks import build_v6_masks
from trainlib.rollouts.halo_eval import deployed_halo_eval

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
    _mv6 = build_v6_masks(repair_band, M_full, agnostic, batch, device, weight_dtype)
    use_v6 = _mv6["use_v6"]
    M_g_v6, M_s_v6, M_b_v6 = _mv6["M_g_v6"], _mv6["M_s_v6"], _mv6["M_b_v6"]
    M_other_v6, M_k_v6 = _mv6["M_other_v6"], _mv6["M_k_v6"]
    M_edit_v6, M_core_v6 = _mv6["M_edit_v6"], _mv6["M_core_v6"]
    agnostic = _mv6["agnostic"]

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

    if int(os.environ.get("PUNISH_FULL_LATENT", "0")):
        # exp1: supervise the ENTIRE latent (denoise the whole frame), not just the edit
        # hole. Outside the hole v_target = (noise-person) too, so the model learns the
        # full image — its weight is set by the keep term (W_FLOW_KEEP). Pairs with whole-
        # latent noise init (C_t already noises everything under USE_PURE_NOISE).
        v_target = (noise - person)

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

    # ── In-loop DEPLOYED-HALO EVAL (state._DEPLOY_HALO_EVAL) — run the FULL N-step from-noise
    #    rollout (the deployment condition that creates the halo), decode, and measure the
    #    bg ring-contrast (ring 0-8px vs surround 20-50px, luminance dL vs GT). Returns the
    #    metric and short-circuits (NO grad/training). run.py calls this ~hourly on the
    #    reserved val ids so we WATCH the deployed halo live. Self-consistent Rec.601-luminance
    #    proxy (not Lab) — valid for the TREND, not calibrated to the offline -1.35 number.
    if state._DEPLOY_HALO_EVAL:
        return None, deployed_halo_eval(
            _fwd, unpack_latents, person, agnostic, M_full, sigma, vae, vae_device,
            weight_dtype, B, C, H, W, image_ids, person_image_cache, batch, device)

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
    _grad_roll = int(os.environ.get("GRAD_ROLLOUT_STEPS", "0"))
    if _grad_roll > 0:
        # ── NO-STOP-GRAD differentiable rollout: roll K steps from the deployed-style init WITH
        #    gradient (no torch.no_grad), so the loss on the final deployed output backprops THROUGH
        #    all K steps — the model learns how its early-step predictions create the late-step halo
        #    (the thing SOAR/exp2's stop-grad rollout cannot). K small = "some of the full rollout",
        #    bounding activation memory. ──
        _gs0 = float(os.environ.get("GRAD_ROLLOUT_SIGMA", "0.95"))
        _gsmin = float(os.environ.get("GRAD_ROLLOUT_SIGMA_MIN", "0.10"))
        sigma = torch.full_like(sigma, _gs0); s = sigma.view(B, 1, 1, 1)
        C_t = ((1.0 - M_full) * agnostic + M_full * torch.randn_like(person)).to(person.dtype)
        _protect = int(os.environ.get("SOAR_PROTECT_GARMENT", "0"))
        if _protect:
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
            _gnoise = torch.randn_like(person)
            def _anchor_gar(_xx, _sv):
                return (1.0 - _gp) * _xx + _gp * (person + _sv.view(B, 1, 1, 1) * (_gnoise - person))
            C_t = _anchor_gar(C_t, sigma)
        _dsig = (_gs0 - _gsmin) / max(_grad_roll, 1)
        _x = C_t; _sig_cur = sigma
        for _gk in range(_grad_roll):                       # DIFFERENTIABLE rollout — NO torch.no_grad
            _v = unpack_latents(_fwd(_x, _sig_cur), B, C, H, W)
            _sig_next = (_sig_cur - _dsig).clamp(min=_gsmin)
            _x = _x - (_sig_cur.view(B, 1, 1, 1) - _sig_next.view(B, 1, 1, 1)) * _v
            if _protect: _x = _anchor_gar(_x, _sig_next)
            _sig_cur = _sig_next
        sigma = _sig_cur; s = sigma.view(B, 1, 1, 1)
        C_t = _x                                            # NOT detached -> grad flows through the rollout
        v_target = M_full * (C_t - person) / s.clamp(min=1e-3)
        vt_p = pack_latents(v_target, B, C, H, W)
        C_p_ = pack_latents(C_t, B, C, H, W)
        pred_C = _fwd(C_t, sigma)
    elif _do_soar:
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

    from trainlib.losses.latent import latent_losses
    from trainlib.losses.region import region_image_losses
    from trainlib.losses.edge_halo import edge_halo_losses
    from trainlib.losses.garment_adv import garment_adv_losses
    from trainlib.losses.background import background_losses
    from trainlib.losses.total import total_loss_and_metrics
    _ctx = dict(locals())
    _ctx.update(latent_losses(_ctx))
    _ctx.update(region_image_losses(_ctx))
    _ctx.update(edge_halo_losses(_ctx))
    _ctx.update(garment_adv_losses(_ctx))
    _ctx.update(background_losses(_ctx))
    return total_loss_and_metrics(_ctx)