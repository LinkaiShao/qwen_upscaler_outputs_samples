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


"""Final weighted loss sum + the metrics dict returned to the trainer."""

def total_loss_and_metrics(ctx):
    """Behavior-preserving slice of train_step's loss section."""
    L_ab = ctx.get('L_ab')
    L_ab_direction = ctx.get('L_ab_direction')
    L_adv = ctx.get('L_adv')
    L_anti_grey = ctx.get('L_anti_grey')
    L_anti_rough = ctx.get('L_anti_rough')
    L_antisludge = ctx.get('L_antisludge')
    L_band = ctx.get('L_band')
    L_band_hp_diag = ctx.get('L_band_hp_diag')
    L_beta_l2 = ctx.get('L_beta_l2')
    L_bg_chroma = ctx.get('L_bg_chroma')
    L_bg_field = ctx.get('L_bg_field')
    L_bg_route_self = ctx.get('L_bg_route_self')
    L_bg_shell_ab = ctx.get('L_bg_shell_ab')
    L_bg_shell_keep = ctx.get('L_bg_shell_keep')
    L_body_edge_grad = ctx.get('L_body_edge_grad')
    L_body_edge_l1 = ctx.get('L_body_edge_l1')
    L_boundary_l1 = ctx.get('L_boundary_l1')
    L_chroma = ctx.get('L_chroma')
    L_chroma_ratio = ctx.get('L_chroma_ratio')
    L_critic_adv = ctx.get('L_critic_adv')
    L_early_alloc = ctx.get('L_early_alloc')
    L_early_broad = ctx.get('L_early_broad')
    L_edge = ctx.get('L_edge')
    L_flow = ctx.get('L_flow')
    L_garment_crop_l1 = ctx.get('L_garment_crop_l1')
    L_garment_edge = ctx.get('L_garment_edge')
    L_harmony = ctx.get('L_harmony')
    L_hf_garment = ctx.get('L_hf_garment')
    L_img = ctx.get('L_img')
    L_inside_edit_halo_ab = ctx.get('L_inside_edit_halo_ab')
    L_inside_edit_halo_hf = ctx.get('L_inside_edit_halo_hf')
    L_inv = ctx.get('L_inv')
    L_late_shell = ctx.get('L_late_shell')
    L_mid_ab = ctx.get('L_mid_ab')
    L_mid_hf = ctx.get('L_mid_hf')
    L_no_bg_leak = ctx.get('L_no_bg_leak')
    L_percep = ctx.get('L_percep')
    L_person_halo_keep = ctx.get('L_person_halo_keep')
    L_recon_ub = ctx.get('L_recon_ub')
    L_repair_bg_img = ctx.get('L_repair_bg_img')
    L_repair_skin_img = ctx.get('L_repair_skin_img')
    L_repair_v6 = ctx.get('L_repair_v6')
    L_route_v6 = ctx.get('L_route_v6')
    L_tv = ctx.get('L_tv')
    L_tv_edge = ctx.get('L_tv_edge')
    L_tv_ring = ctx.get('L_tv_ring')
    L_v6_boundary = ctx.get('L_v6_boundary')
    L_warp_fp = ctx.get('L_warp_fp')
    _lam_bel1 = ctx.get('_lam_bel1')
    _lam_belg = ctx.get('_lam_belg')
    _lam_crit = ctx.get('_lam_crit')
    _lambda_bg_chroma = ctx.get('_lambda_bg_chroma')
    _lambda_bg_field = ctx.get('_lambda_bg_field')
    _lambda_bnd = ctx.get('_lambda_bnd')
    _lambda_gcl1 = ctx.get('_lambda_gcl1')
    _lambda_rb = ctx.get('_lambda_rb')
    _lambda_rs = ctx.get('_lambda_rs')
    _w_brs = ctx.get('_w_brs')
    _w_fp = ctx.get('_w_fp')
    _w_gem = ctx.get('_w_gem')
    g_band = ctx.get('g_band')
    img_loss_weight = ctx.get('img_loss_weight')
    lambda_ab = ctx.get('lambda_ab')
    lambda_ab_direction = ctx.get('lambda_ab_direction')
    lambda_adv = ctx.get('lambda_adv')
    lambda_alloc = ctx.get('lambda_alloc')
    lambda_anti_grey = ctx.get('lambda_anti_grey')
    lambda_anti_rough = ctx.get('lambda_anti_rough')
    lambda_antisludge = ctx.get('lambda_antisludge')
    lambda_band = ctx.get('lambda_band')
    lambda_bg_shell_ab = ctx.get('lambda_bg_shell_ab')
    lambda_bg_shell_keep = ctx.get('lambda_bg_shell_keep')
    lambda_broad_ratio = ctx.get('lambda_broad_ratio')
    lambda_chroma = ctx.get('lambda_chroma')
    lambda_chroma_ratio = ctx.get('lambda_chroma_ratio')
    lambda_hf_g_eff = ctx.get('lambda_hf_g_eff')
    lambda_inside_ab = ctx.get('lambda_inside_ab')
    lambda_inside_hf = ctx.get('lambda_inside_hf')
    lambda_inv = ctx.get('lambda_inv')
    lambda_late_shell = ctx.get('lambda_late_shell')
    lambda_mid_ab = ctx.get('lambda_mid_ab')
    lambda_mid_hf = ctx.get('lambda_mid_hf')
    lambda_no_bg = ctx.get('lambda_no_bg')
    lambda_percep_eff = ctx.get('lambda_percep_eff')
    lambda_person_halo_keep = ctx.get('lambda_person_halo_keep')
    lambda_recon = ctx.get('lambda_recon')
    lambda_tv = ctx.get('lambda_tv')
    lambda_tv_edge = ctx.get('lambda_tv_edge')
    lambda_tv_ring = ctx.get('lambda_tv_ring')
    loss = ctx.get('loss')
    sigma = ctx.get('sigma')
    w_flow = ctx.get('w_flow')
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
            + float(os.environ.get("W_HARMONY", "0.0")) * L_harmony
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
                  "harm": L_harmony.item(),
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
