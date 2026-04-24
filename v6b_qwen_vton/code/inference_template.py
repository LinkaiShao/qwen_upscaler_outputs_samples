"""Inference adapter — exp401 (5-pos: exp395 + rough spatial slot, pose-mode aware)."""

import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel, QwenDoubleStreamAttnProcessor2_0,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

# exp415: custom attention processor that reads attention_mask from module global
_INFER_MASK_HOLDER = {"mask": None}

class CoreRingAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None):
        if attention_mask is None:
            attention_mask = _INFER_MASK_HOLDER.get("mask")
        return super().__call__(attn, hidden_states, encoder_hidden_states,
                                encoder_hidden_states_mask, attention_mask, image_rotary_emb)

# Set by load_model based on config. 0=none, 1=block C-core→ring, 2=block both directions
ATTN_MASK_MODE = 0
PIN_RING_ROUGH = 0   # Set by train.py at save time
USE_SIGMA_SCHED = 0  # Set by train.py at save time
SIGMA_SCHED_LO = 0.8 # Set by train.py at save time
SIGMA_SCHED_HI = 1.2 # Set by train.py at save time
USE_PURE_NOISE = 0   # Set by train.py at save time
USE_ROUGH_BLUR_FIXED = 0  # Set by train.py at save time
ROUGH_BLUR_FIXED_SIG = 4.0  # Set by train.py at save time
USE_ROUGH_MASKED = 0  # Set by train.py at save time
USE_AGNOSTIC_MEAN_FILL = 0  # Set by train.py at save time
USE_AGNOSTIC_ROUGH_FILL = 0  # Set by train.py at save time
USE_AGNOSTIC_INPAINT = 0  # Set by train.py at save time
AGNOSTIC_INPAINT_SOFT_SIG = 0.0  # Set by train.py at save time
AGNOSTIC_ZERO_REPAIR = 0  # Set by train.py at save time
SILHOUETTE_SCALE = 1.0  # Set by train.py at save time
SILHOUETTE_SOFT = 0  # Set by train.py at save time
USE_VAE_SILHOUETTE = 0  # Set by train.py at save time
SILHOUETTE_SOFT_SIG = 2.0  # Set by train.py at save time
ROUGH_MASK_SOFT = 0  # Set by train.py at save time
ROUGH_MASK_SOFT_SIG = 3.0  # Set by train.py at save time
USE_REPAIR_ATTN_MASK = 0  # Set by train.py at save time
USE_BG_HINT = 0  # Set by train.py at save time
BG_HINT_SCALE = 1.0  # Set by train.py at save time
V6_ZERO_G_CORE = 0  # Set by train.py at save time
V6_R_IN = 2  # Set by train.py at save time
V6_R_OUT = 7  # Set by train.py at save time


WRONG_POSE_MAP = {
    "00006_00": "00008_00",
    "00008_00": "00013_00",
    "00013_00": "00017_00",
    "00017_00": "00034_00",
    "00034_00": "00006_00",
}

# Training slot order for this run (indices into [agnostic, pose, rough, garment]).
# Patched by train.py at save time to match what the LoRA was trained with.
DEFAULT_SLOT_ORDER = [0, 1, 2, 3]


def _pack(lat, B, C, H, W):
    return lat.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

def _unpack(lat, B, C, H, W):
    return lat.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


# ─────────────────────────── load_model ───────────────────────────

def load_model(run_dir, device, config):
    td = torch.device(device); wd = torch.bfloat16

    t = QwenImageTransformer2DModel.from_pretrained(
        config.pretrained_model, subfolder="transformer", torch_dtype=wd)
    t = get_peft_model(
        t, LoraConfig(r=config.rank, lora_alpha=config.alpha,
                      init_lora_weights=config.init_lora_weights,
                      target_modules=config.lora_targets, lora_dropout=0.0),
        adapter_name="tryon")

    tryon_path = os.path.join(run_dir, "tryon_lora.safetensors")
    if os.path.exists(tryon_path):
        set_peft_model_state_dict(t, load_file(tryon_path), adapter_name="tryon")

    # exp415: install custom attention processor if this run trained with mask mode
    if ATTN_MASK_MODE or USE_REPAIR_ATTN_MASK:
        n = 0
        for mod in t.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = CoreRingAttnProcessor()
                n += 1
        print(f"CoreRingAttnProcessor installed on {n} blocks (mode={ATTN_MASK_MODE}, repair_mask={USE_REPAIR_ATTN_MASK})")

    t.to(td).eval()

    prompt_cache_raw = torch.load(os.path.join(run_dir, "prompt_cache.pt"), weights_only=True)
    pose_cache_raw   = torch.load(os.path.join(run_dir, "pose_latent_cache.pt"), weights_only=True)
    prompt_cache = {k: (v[0].to(td, dtype=wd), v[1].to(td, dtype=torch.long))
                    for k, v in prompt_cache_raw.items()}
    pose_cache   = {k: v.to(td, dtype=wd) for k, v in pose_cache_raw.items()}

    # Load v6 specialized heads if present
    repair_head = None
    routing_head = None
    rh_path = os.path.join(run_dir, "repair_head.pt")
    if os.path.exists(rh_path):
        import torch.nn as nn
        repair_head = nn.Conv2d(16, 16, 1, bias=True).to(td, dtype=wd)
        sd = torch.load(rh_path, weights_only=True)
        # state_dict saved from module with key "conv.weight"/"conv.bias"
        repair_head.load_state_dict({"weight": sd["conv.weight"], "bias": sd["conv.bias"]})
        repair_head.eval()
        print(f"loaded repair_head from {rh_path}")
    rt_path = os.path.join(run_dir, "routing_head.pt")
    if os.path.exists(rt_path):
        import torch.nn as nn
        routing_head = nn.Conv2d(16, 4, 1, bias=True).to(td, dtype=wd)
        sd = torch.load(rt_path, weights_only=True)
        routing_head.load_state_dict({"weight": sd["conv.weight"], "bias": sd["conv.bias"]})
        routing_head.eval()
        print(f"loaded routing_head from {rt_path}")

    return {
        "transformer":   t,
        "prompt_cache":  prompt_cache,
        "pose_cache":    pose_cache,
        "repair_head":   repair_head,
        "routing_head":  routing_head,
        "config":        config,
        "device":        td,
        "weight_dtype":  wd,
    }


# ─────────────────────────── predict_sample ───────────────────────────

def predict_sample(model, batch, device, seed, settings):
    t             = model["transformer"]
    prompt_cache  = model["prompt_cache"]
    pose_cache    = model["pose_cache"]
    repair_head   = model.get("repair_head")
    routing_head  = model.get("routing_head")
    td            = model["device"]
    wd            = model["weight_dtype"]
    config        = model["config"]

    al = batch["agnostic_latent"].to(td, dtype=wd)
    gl = batch["garment_latent"].to(td, dtype=wd)
    rl = batch["rough_latent"].to(td, dtype=wd)
    am = batch["agnostic_mask_latent"].to(td, dtype=torch.float32)
    image_ids = batch.get("image_id", batch.get("image_ids"))
    if isinstance(image_ids, str): image_ids = [image_ids]

    B, C, H, W = al.shape
    M = (am > 0.5).to(dtype=wd)

    # Mean-fill agnostic: mirror training-time substitution of grey mask with
    # the per-sample mean color of unmasked agnostic region.
    if USE_AGNOSTIC_MEAN_FILL:
        _M = M
        _unmasked_sum = (al * (1 - _M)).sum(dim=(-2, -1), keepdim=True)
        _unmasked_area = (1 - _M).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        _agn_mean = _unmasked_sum / _unmasked_area
        al = al * (1 - _M) + _agn_mean * _M

    # Rough-fill repair zone: agnostic's repair zone (M_full minus garment silhouette)
    # gets filled with rough's body estimate so the model sees a coherent input.
    if USE_AGNOSTIC_ROUGH_FILL:
        wm_rf = batch.get("warped_mask")
        if wm_rf is not None:
            wm_rf = wm_rf.to(td, dtype=wd)
            if wm_rf.dim() == 3: wm_rf = wm_rf.unsqueeze(1)
            _gar_bin_rf = (wm_rf > 0.5).to(wd)
            _repair_rf = (M - _gar_bin_rf).clamp(0, 1)
            al = al * (1 - _repair_rf) + rl * _repair_rf

    if USE_AGNOSTIC_INPAINT:
        from torchvision.transforms.functional import gaussian_blur as _gb
        _M_paste = M
        if AGNOSTIC_INPAINT_SOFT_SIG > 0:
            _k_sp = int(2 * round(2 * AGNOSTIC_INPAINT_SOFT_SIG) + 1)
            _M_paste = _gb(M.float(), kernel_size=[_k_sp, _k_sp],
                            sigma=AGNOSTIC_INPAINT_SOFT_SIG).to(M.dtype).clamp(0, 1)
        _keep = (1 - _M_paste)
        _result = al.clone()
        for _ in range(20):
            _blurred = _gb(_result.float(), kernel_size=[7, 7], sigma=2.0).to(_result.dtype)
            _result = _result * _keep + _blurred * _M_paste
        al = _result

    # v6: zero agnostic inside the confident garment core (M_core = erode(warped, r_in)).
    # Mirrors train-time V6_ZERO_G_CORE without needing target parse (uses eroded warped
    # as a safe proxy for M_g; parse could refine but not available at inference for
    # arbitrary samples). Removes torso template from input → model must synthesize.
    if V6_ZERO_G_CORE:
        wm_v6 = batch.get("warped_mask")
        if wm_v6 is not None:
            wm_v6 = wm_v6.to(td, dtype=wd)
            if wm_v6.dim() == 3: wm_v6 = wm_v6.unsqueeze(1)
            wm_bin_v6 = (wm_v6 > 0.5).to(wd)
            _r_in_v6 = V6_R_IN
            _core_v6 = -F.max_pool2d(-wm_bin_v6, 2*_r_in_v6+1, 1, _r_in_v6)
            _core_v6 = (_core_v6 > 0.5).to(wd)
            al = al * (1.0 - _core_v6)

    # exp552+: zero the agnostic in the repair band (M_full - warped_mask). Mirror
    # of train-time AGNOSTIC_ZERO_REPAIR. Uses warped_mask (inference-available).
    if AGNOSTIC_ZERO_REPAIR:
        wm_zr = batch.get("warped_mask")
        if wm_zr is not None:
            wm_zr = wm_zr.to(td, dtype=wd)
            if wm_zr.dim() == 3: wm_zr = wm_zr.unsqueeze(1)
            _wm_zr_bin = (wm_zr > 0.5).to(wd)
            _M_ag_bin = M.to(wd)
            _repair_proxy = (_M_ag_bin - _wm_zr_bin).clamp(0, 1)
            al = al * (1 - _repair_proxy)

    # exp419: no neutralization — feed raw agnostic and rough as conditioning.
    # The model learns routing from the soft-mask loss structure.

    pose_mode = settings.get("pose_mode", "normal")
    pose_list = []
    for iid in image_ids:
        if pose_mode == "no_pose":
            pose_list.append(torch.zeros_like(pose_cache[iid]))
        elif pose_mode == "wrong_pose":
            wrong_id = WRONG_POSE_MAP.get(iid, iid)
            pose_list.append(pose_cache.get(wrong_id, pose_cache[iid]))
        else:
            pose_list.append(pose_cache[iid])
    pose = torch.stack(pose_list).to(td, dtype=wd)

    # ── Input ablation: zero out a named slot ──
    zero_slot = settings.get("zero_slot", None)
    if zero_slot == "agnostic": al   = torch.zeros_like(al)
    if zero_slot == "pose":     pose = torch.zeros_like(pose)
    if zero_slot == "rough":    rl   = torch.zeros_like(rl)
    if zero_slot == "garment":  gl   = torch.zeros_like(gl)

    # ── Garment-identity override: replace garment latent with arbitrary tensor ──
    if "garment_latent_override" in settings:
        gl = settings["garment_latent_override"].to(td, dtype=wd)

    # ── Slot order override: reorder the spatial-slot concatenation ──
    # settings["slot_order"] is a list of ints indexing into [agnostic, pose, rough, garment]
    # at positions 1..4 after C (position 0). Default = [0, 1, 2, 3].

    # Mirror training-time fixed rough blur, if configured
    if USE_ROUGH_BLUR_FIXED:
        from torchvision.transforms.functional import gaussian_blur
        bk = int(2 * round(2 * ROUGH_BLUR_FIXED_SIG) + 1)
        rl = gaussian_blur(rl.float(), kernel_size=[bk, bk], sigma=ROUGH_BLUR_FIXED_SIG).to(rl.dtype)

    if USE_ROUGH_MASKED:
        wm_r = batch.get("warped_mask")
        if wm_r is not None:
            wm_r = wm_r.to(td, dtype=wd)
            if wm_r.dim() == 3: wm_r = wm_r.unsqueeze(1)
            if ROUGH_MASK_SOFT:
                from torchvision.transforms.functional import gaussian_blur as _gbrm
                _k_r = int(2 * round(2 * ROUGH_MASK_SOFT_SIG) + 1)
                _wm_bin_r = (wm_r > 0.5).to(wd)
                _wm_soft_r = _gbrm(_wm_bin_r.float(), kernel_size=[_k_r, _k_r], sigma=ROUGH_MASK_SOFT_SIG).to(wd)
                rl = rl * _wm_soft_r
            else:
                rl = rl * (wm_r > 0.5).to(wd)

    agn_p   = _pack(al,   B, C, H, W)
    pose_p  = _pack(pose, B, C, H, W)
    rough_p = _pack(rl,   B, C, H, W)
    gar_p   = _pack(gl,   B, C, H, W)
    # Silhouette slot: use warped_mask as explicit garment location signal
    sil_p = None
    if 4 in settings.get("slot_order", DEFAULT_SLOT_ORDER):
        if USE_VAE_SILHOUETTE:
            # Pre-computed VAE-encoded silhouette latent (in-distribution).
            vl = batch.get("warped_silhouette_latent")
            if vl is not None:
                vl = vl.to(td, dtype=wd)
                sil_p = _pack(vl, B, C, H, W)
        if sil_p is None:
            wm = batch.get("warped_mask")
            if wm is not None:
                wm = wm.to(td, dtype=wd)
                if wm.dim() == 3: wm = wm.unsqueeze(1)
                if SILHOUETTE_SOFT:
                    from torchvision.transforms.functional import gaussian_blur as _gb_sil
                    _k_s = int(2 * round(2 * SILHOUETTE_SOFT_SIG) + 1)
                    wm_bin = (wm > 0.5).to(wd)
                    wm_b = _gb_sil(wm_bin.float(), kernel_size=[_k_s, _k_s], sigma=SILHOUETTE_SOFT_SIG).to(wd)
                else:
                    wm_b = (wm > 0.5).to(wd)
                if USE_BG_HINT:
                    # ch 0 = garment silhouette, ch 1 = bg-hint (agnostic ∩ ¬body ∩ ¬warped)
                    _sil = torch.zeros((B, C, H, W), device=td, dtype=wd)
                    _sil[:, 0:1] = wm_b * SILHOUETTE_SCALE
                    dp = batch.get("densepose")
                    if dp is not None:
                        dp = dp.to(td, dtype=wd)
                        body_img = (dp.sum(dim=1, keepdim=True) > 0.02).to(wd)
                        body_lat = F.interpolate(body_img, size=(H, W), mode="area")
                        body_lat = (body_lat > 0.5).to(wd)
                        wm_bin_lat = (wm > 0.5).to(wd)
                        M_ag_bin = (M > 0.5).to(wd)
                        _bg = (M_ag_bin * (1.0 - body_lat) * (1.0 - wm_bin_lat)).clamp(0, 1)
                        _sil[:, 1:2] = _bg * BG_HINT_SCALE
                    sil_p = _pack(_sil, B, C, H, W)
                else:
                    wm_b = (wm_b * SILHOUETTE_SCALE).expand(B, C, H, W)
                    sil_p = _pack(wm_b, B, C, H, W)
    # Body-rough slot: rough * (1 - warped_mask) for explicit body/background context
    br_p = None
    if 5 in settings.get("slot_order", DEFAULT_SLOT_ORDER):
        wm = batch.get("warped_mask")
        if wm is not None:
            wm = wm.to(td, dtype=wd)
            if wm.dim() == 3: wm = wm.unsqueeze(1)
            wm_bin = (wm > 0.5).to(wd)
            br_p = _pack(rl * (1 - wm_bin), B, C, H, W)
    # img_shapes_base is filled after slot_order is resolved (variable length)

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
    pe = torch.cat(pe_pad, dim=0).to(td, dtype=wd)
    pm = torch.cat(pm_pad, dim=0).to(td, dtype=torch.long)
    txt_seq_lens = pm.sum(dim=1).tolist()

    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")
    isl = (H//2) * (W//2)
    sl  = (sch.config.max_shift - sch.config.base_shift) / (sch.config.max_image_seq_len - sch.config.base_image_seq_len)
    mu  = isl * sl + (sch.config.base_shift - sl * sch.config.base_image_seq_len)
    sch.set_timesteps(settings.get("num_inference_steps", 50), mu=mu)

    g = torch.Generator(device=td).manual_seed(seed)
    noise = torch.randn(al.shape, device=td, dtype=wd, generator=g)

    C_lat = noise if USE_PURE_NOISE else (1 - M) * al + M * noise

    slot_order   = settings.get("slot_order", DEFAULT_SLOT_ORDER)
    img_shapes_base = [(1, H//2, W//2)] * (1 + len(slot_order))

    cfg_scale = settings.get("cfg_scale", 1.0)

    # Build attention mask once (depends on M and warped_mask, both static across steps)
    if USE_REPAIR_ATTN_MASK:
        # M is agnostic_mask_latent at latent res, shape (B, 1, H, W)
        # wm is warped_mask_128 at latent res, also (B, 1, H, W)
        wm_for_mask = batch.get("warped_mask")
        if wm_for_mask is not None:
            wm_for_mask = wm_for_mask.to(td, dtype=wd)
            if wm_for_mask.dim() == 3: wm_for_mask = wm_for_mask.unsqueeze(1)
            # agnostic slot index in seq: position 1 (right after C) for the standard SLOT_ORDER
            agn_slot_pos = 1 if 0 in slot_order else 1
            wm_bin = (wm_for_mask > 0.5).to(M.dtype)
            repair_band = (M - wm_bin).clamp(0, 1)
            keep_mask = (1.0 - M).clamp(0, 1)
            repair_tok = _pack(repair_band.expand(B, C, H, W), B, C, H, W).mean(dim=-1)
            keep_tok   = _pack(keep_mask.expand(B, C, H, W),   B, C, H, W).mean(dim=-1)
            repair_pos = (repair_tok > 0.5)
            keep_pos   = (keep_tok > 0.5)
            txt_len = pe.shape[1]
            img_tok = (H // 2) * (W // 2)
            num_slots = 1 + len(slot_order)
            total_seq = txt_len + num_slots * img_tok
            inf_mask = torch.zeros(B, 1, total_seq, total_seq, device=td, dtype=wd)
            for b in range(B):
                q_idx = txt_len + torch.where(repair_pos[b])[0]
                k_idx = txt_len + agn_slot_pos * img_tok + torch.where(keep_pos[b])[0]
                if q_idx.numel() > 0 and k_idx.numel() > 0:
                    inf_mask[b, 0, q_idx.unsqueeze(1), k_idx.unsqueeze(0)] = -1e4
            _INFER_MASK_HOLDER["mask"] = inf_mask

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=wd):
        for ts in sch.timesteps:
            C_p    = _pack(C_lat, B, C, H, W)
            sig    = (ts / 1000).to(device=td, dtype=wd).expand(B)

            # Sigma-scheduled conditioning scales (must match training)
            if USE_SIGMA_SCHED:
                s_val = (ts / 1000).to(device=td, dtype=wd)
                _span = SIGMA_SCHED_HI - SIGMA_SCHED_LO
                struct_scale = (SIGMA_SCHED_LO + _span * s_val).to(agn_p.dtype)
                detail_scale = (SIGMA_SCHED_HI - _span * s_val).to(gar_p.dtype)
                agn_p_s   = agn_p   * struct_scale
                pose_p_s  = pose_p  * struct_scale
                rough_p_s = rough_p * detail_scale
                gar_p_s   = gar_p   * detail_scale
            else:
                agn_p_s, pose_p_s, rough_p_s, gar_p_s = agn_p, pose_p, rough_p, gar_p

            slot_tensors = [agn_p_s, pose_p_s, rough_p_s, gar_p_s, sil_p, br_p]
            cond_seq = [slot_tensors[i] for i in slot_order]
            if cfg_scale != 1.0:
                uncond_tensors = [agn_p_s, pose_p_s, rough_p_s, torch.zeros_like(gar_p_s), sil_p, br_p]
                uncond_seq = [uncond_tensors[i] for i in slot_order]

            hidden = torch.cat([C_p] + cond_seq, dim=1)
            out = t(
                hidden_states              = hidden,
                timestep                   = sig,
                encoder_hidden_states      = pe,
                encoder_hidden_states_mask = pm,
                img_shapes                 = [img_shapes_base] * B,
                txt_seq_lens               = txt_seq_lens,
                return_dict                = False,
            )[0]
            v_cond = out[:, :C_p.size(1), :]

            if cfg_scale != 1.0:
                hidden_u = torch.cat([C_p] + uncond_seq, dim=1)
                out_u = t(
                    hidden_states              = hidden_u,
                    timestep                   = sig,
                    encoder_hidden_states      = pe,
                    encoder_hidden_states_mask = pm,
                    img_shapes                 = [img_shapes_base] * B,
                    txt_seq_lens               = txt_seq_lens,
                    return_dict                = False,
                )[0]
                v_uncond = out_u[:, :C_p.size(1), :]
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            C_p    = sch.step(v_pred, ts, C_p, return_dict=False)[0]
            C_lat  = _unpack(C_p, B, C, H, W)
            # Keep last step's unpacked v_pred for v6 repair head
            v_pred_lat_last = _unpack(v_pred, B, C, H, W)

    # ── v6 composition path: use specialized heads to compose final latent ──
    if repair_head is not None:
        wm_v6c = batch.get("warped_mask")
        if wm_v6c is None:
            raise RuntimeError("v6 inference requires warped_mask in batch")
        wm_v6c = wm_v6c.to(td, dtype=wd)
        if wm_v6c.dim() == 3: wm_v6c = wm_v6c.unsqueeze(1)
        wm_bin_c = (wm_v6c > 0.5).to(wd)
        # M_edit (dilate), M_g = warped, ring = M_edit - M_g, M_k = 1 - M_edit
        M_edit_c = F.max_pool2d(wm_bin_c, 2*V6_R_OUT+1, 1, V6_R_OUT).clamp(0, 1)
        M_g_c    = wm_bin_c
        ring_c   = (M_edit_c - M_g_c).clamp(0, 1)
        M_k_c    = (1.0 - M_edit_c).clamp(0, 1)
        # δ from repair head on last v_pred
        with torch.no_grad():
            delta_c = repair_head(v_pred_lat_last.to(wd))
        x_repair_c = al + delta_c                                              # source + residual
        # Compose: garment from denoised latent, ring from source+δ, keep from source
        final = M_g_c * C_lat + ring_c * x_repair_c + M_k_c * al
        return {"pred_latents": final}

    # Legacy soft composite path (non-v6)
    from torchvision.transforms.functional import gaussian_blur as _gb_final
    _comp_sig = float(os.environ.get("COMPOSITE_SIGMA", "2.0"))
    _comp_k = int(2 * round(2 * _comp_sig) + 1) if _comp_sig >= 2.0 else 7
    _M_blur = _gb_final(M.float(), kernel_size=[_comp_k, _comp_k], sigma=_comp_sig).to(M.dtype)
    _M_soft = _M_blur
    final = (1 - _M_soft) * al + _M_soft * C_lat
    return {"pred_latents": final}
