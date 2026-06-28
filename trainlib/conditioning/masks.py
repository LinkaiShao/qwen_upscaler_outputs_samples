"""v6 region-routing masks: partition the edit region into garment / skin / bg / other /
keep, plus edit/core masks. Behavior-preserving extraction of the block formerly inline in
forward.train_step (the `# v6 routing classes` section)."""
import os
import torch
import torch.nn.functional as F


def build_v6_masks(repair_band, M_full, agnostic, batch, device, weight_dtype):
    """Build the v6 routing-class masks.

    Returns a dict with use_v6 (bool), the region masks (M_g/M_s/M_b/M_other/M_k_v6),
    M_edit_v6, M_core_v6, and the (possibly core-zeroed) agnostic latent.

      M_edit = dilate(warped, r_out) [or M_full if V6_PARTITION_FULL_HOLE]  (edit support)
      M_core = erode(warped,  r_in)                                          (confident garment core)
      M_g    = warped ∩ M_edit                                               (new garment)
      ring   = M_edit − M_g; M_s = ring ∩ parse_skin; M_b = ring ∩ parse_bg ∩ ¬skin
      M_other= ring − M_s − M_b ; M_k = 1 − M_edit                           (keep untouched)
    """
    use_v6 = int(os.environ.get("USE_V6", "0")) and "parse_garment" in batch
    M_g_v6 = torch.zeros_like(repair_band)
    M_s_v6 = torch.zeros_like(repair_band)
    M_b_v6 = torch.zeros_like(repair_band)
    M_other_v6 = torch.zeros_like(repair_band)
    M_k_v6 = torch.ones_like(repair_band)
    M_edit_v6 = torch.zeros_like(repair_band)
    M_core_v6 = torch.zeros_like(repair_band)
    if use_v6:
        r_out = int(os.environ.get("V6_R_OUT", "7"))     # latent px
        r_in  = int(os.environ.get("V6_R_IN",  "2"))
        _wm_v6 = batch["warped_mask"].to(device, dtype=weight_dtype)
        if _wm_v6.dim() == 3: _wm_v6 = _wm_v6.unsqueeze(1)
        _wm_bin_v6 = (_wm_v6 > 0.5).to(weight_dtype)
        if int(os.environ.get("V6_PARTITION_FULL_HOLE", "0")):
            # regions partition the ENTIRE generated hole (M_full), not just warped+r_out.
            # -> g+s+b+other == M_full exactly (100% coverage); bg-in-gaps gets the bg task.
            M_edit_v6 = (M_full if M_full.dim() == 4 else M_full.unsqueeze(1)).clamp(0, 1)
        else:
            M_edit_v6 = F.max_pool2d(_wm_bin_v6, 2*r_out+1, 1, r_out).clamp(0, 1)
        _M_core_t = -F.max_pool2d(-_wm_bin_v6, 2*r_in+1, 1, r_in)
        M_core_v6 = (_M_core_t > 0.5).to(weight_dtype)

        _ps = batch["parse_skin"].to(device, dtype=weight_dtype)
        _pb = batch["parse_bg"].to(device, dtype=weight_dtype)
        if _ps.dim() == 3: _ps = _ps.unsqueeze(1)
        if _pb.dim() == 3: _pb = _pb.unsqueeze(1)
        _ps_b = (_ps > 0.5).to(weight_dtype)
        _pb_b = (_pb > 0.5).to(weight_dtype)

        # SAFE routing: warped defines garment class (inference-available). Parse only
        # subdivides the ring (M_edit − warped) into skin/bg; disagreements fall to M_other
        # (keep-like), avoiding oracle-parse geometry inside the garment core.
        M_g_v6 = (_wm_bin_v6 * M_edit_v6).clamp(0, 1)                          # garment within edit
        ring_v6 = (M_edit_v6 - M_g_v6).clamp(0, 1)                              # edit minus garment
        M_s_v6 = (ring_v6 * _ps_b).clamp(0, 1)                                  # skin only in ring
        M_b_v6 = (ring_v6 * _pb_b * (1.0 - M_s_v6)).clamp(0, 1)                 # bg only in ring
        M_other_v6 = (ring_v6 - M_s_v6 - M_b_v6).clamp(0, 1)                    # hair/pants/unknown
        M_k_v6 = (1.0 - M_edit_v6).clamp(0, 1)

        # Kill garment-core torso template: zero agnostic inside M_core (eroded warped_mask,
        # inference-available) so train and inference do the same input surgery.
        if int(os.environ.get("V6_ZERO_G_CORE", "1")):
            agnostic = agnostic * (1.0 - M_core_v6)

    return {
        "use_v6": use_v6, "M_g_v6": M_g_v6, "M_s_v6": M_s_v6, "M_b_v6": M_b_v6,
        "M_other_v6": M_other_v6, "M_k_v6": M_k_v6, "M_edit_v6": M_edit_v6,
        "M_core_v6": M_core_v6, "agnostic": agnostic,
    }
