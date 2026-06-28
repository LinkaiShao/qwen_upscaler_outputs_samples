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


"""Garment high-frequency texture, PatchGAN adversarial, conditioned critic, anti-rough-HF."""

def garment_adv_losses(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    Cr = ctx.get('Cr')
    Cr_opt = ctx.get('Cr_opt')
    D = ctx.get('D')
    D_opt = ctx.get('D_opt')
    H_bad = ctx.get('H_bad')
    Hi = ctx.get('Hi')
    L = ctx.get('L')
    L_flow = ctx.get('L_flow')
    L_hf_garment = ctx.get('L_hf_garment')
    L_rank = ctx.get('L_rank')
    M_full = ctx.get('M_full')
    M_g_lat = ctx.get('M_g_lat')
    M_g_pix = ctx.get('M_g_pix')
    R = ctx.get('R')
    R_bad = ctx.get('R_bad')
    R_c = ctx.get('R_c')
    R_clean = ctx.get('R_clean')
    R_g = ctx.get('R_g')
    R_gar = ctx.get('R_gar')
    R_model = ctx.get('R_model')
    Wi = ctx.get('Wi')
    _Hraw = ctx.get('_Hraw')
    _R = ctx.get('_R')
    _b = ctx.get('_b')
    _ci = ctx.get('_ci')
    _ck = ctx.get('_ck')
    _ckinds = ctx.get('_ckinds')
    _cmetrics = ctx.get('_cmetrics')
    _creg = ctx.get('_creg')
    _crit_pretrain = ctx.get('_crit_pretrain')
    _cw = ctx.get('_cw')
    _edge = ctx.get('_edge')
    _err = ctx.get('_err')
    _err_hf = ctx.get('_err_hf')
    _err_lf = ctx.get('_err_lf')
    _gf = ctx.get('_gf')
    _gm = ctx.get('_gm')
    _gtf = ctx.get('_gtf')
    _hfake = ctx.get('_hfake')
    _hfn = ctx.get('_hfn')
    _hi = ctx.get('_hi')
    _hpw = ctx.get('_hpw')
    _hreal = ctx.get('_hreal')
    _hw = ctx.get('_hw')
    _lam_crit = ctx.get('_lam_crit')
    _mb = ctx.get('_mb')
    _pf = ctx.get('_pf')
    _pix = ctx.get('_pix')
    _rkw = ctx.get('_rkw')
    _rmargin = ctx.get('_rmargin')
    _rmean = ctx.get('_rmean')
    _sg = ctx.get('_sg')
    _thr = ctx.get('_thr')
    _thr_k = ctx.get('_thr_k')
    _thr_o = ctx.get('_thr_o')
    _topk = ctx.get('_topk')
    _vals = ctx.get('_vals')
    _ww = ctx.get('_ww')
    _zf = ctx.get('_zf')
    agn_ctx = ctx.get('agn_ctx')
    am_lat = ctx.get('am_lat')
    batch = ctx.get('batch')
    cand = ctx.get('cand')
    cd = ctx.get('cd')
    crit_gate = ctx.get('crit_gate')
    d_corr = ctx.get('d_corr')
    d_fake = ctx.get('d_fake')
    d_fake_g = ctx.get('d_fake_g')
    d_g = ctx.get('d_g')
    d_harm = ctx.get('d_harm')
    d_loss = ctx.get('d_loss')
    d_real = ctx.get('d_real')
    d_wrong = ctx.get('d_wrong')
    device = ctx.get('device')
    diff = ctx.get('diff')
    dil = ctx.get('dil')
    fake_for_g = ctx.get('fake_for_g')
    fake_patch = ctx.get('fake_patch')
    gar_ref = ctx.get('gar_ref')
    global_step = ctx.get('global_step')
    gt_blur = ctx.get('gt_blur')
    gt_img = ctx.get('gt_img')
    gtf = ctx.get('gtf')
    gx = ctx.get('gx')
    gy = ctx.get('gy')
    hp_gt = ctx.get('hp_gt')
    hp_pred = ctx.get('hp_pred')
    k = ctx.get('k')
    k_hf = ctx.get('k_hf')
    k_hp = ctx.get('k_hp')
    key = ctx.get('key')
    lambda_adv = ctx.get('lambda_adv')
    lambda_anti_rough = ctx.get('lambda_anti_rough')
    lambda_hf_g = ctx.get('lambda_hf_g')
    m = ctx.get('m')
    m_be = ctx.get('m_be')
    m_bg = ctx.get('m_bg')
    m_gar = ctx.get('m_gar')
    m_model = ctx.get('m_model')
    m_skin = ctx.get('m_skin')
    m_v = ctx.get('m_v')
    mask_edit_img = ctx.get('mask_edit_img')
    masks5 = ctx.get('masks5')
    p = ctx.get('p')
    person = ctx.get('person')
    person_imgs = ctx.get('person_imgs')
    pg_lat = ctx.get('pg_lat')
    pred_blur = ctx.get('pred_blur')
    pred_d = ctx.get('pred_d')
    pred_f = ctx.get('pred_f')
    pred_hp = ctx.get('pred_hp')
    pred_img = ctx.get('pred_img')
    prod = ctx.get('prod')
    real_patch = ctx.get('real_patch')
    repair_band = ctx.get('repair_band')
    repair_img_mask = ctx.get('repair_img_mask')
    rough = ctx.get('rough')
    rough_5d = ctx.get('rough_5d')
    rough_dec = ctx.get('rough_dec')
    rough_denorm = ctx.get('rough_denorm')
    rough_f = ctx.get('rough_f')
    rough_hp = ctx.get('rough_hp')
    rough_img = ctx.get('rough_img')
    s_v = ctx.get('s_v')
    sig_hf = ctx.get('sig_hf')
    sig_hp = ctx.get('sig_hp')
    sigma = ctx.get('sigma')
    t = ctx.get('t')
    v = ctx.get('v')
    vae = ctx.get('vae')
    vae_device = ctx.get('vae_device')
    wd = ctx.get('wd')
    weight_dtype = ctx.get('weight_dtype')
    wrong = ctx.get('wrong')
    x = ctx.get('x')
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

    return {k: v for k, v in locals().items() if not k.startswith("__") and k != "ctx"}