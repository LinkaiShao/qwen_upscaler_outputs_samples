"""GAN rollout branch (spec §2,5): differentiable inference-trajectory rollout + GAN loss.

ONE denoising path — reuses the run's real `predict_sample` with `grad_start_step=12`, so
the rollout reproduces the EXACT inference conditioning + scheduler the critic was trained
on. Prefix steps 0..11 run detached; tail 12..19 differentiable; decode + grey-hole paste
match the deployed pred.png exactly; then the paired-gap GAN loss (trainlib.gan_loss).

Faithfulness self-test (run directly): with grad_start=20 (all no-grad) the rollout's
pred_final must equal the panel pred.png produced by make_pred for the same sid/seed.
"""
import os, sys, importlib.util
import numpy as np
import torch
from PIL import Image

_VTON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # vtonautoresearch/
_BASE = os.path.dirname(_VTON)                                        # fashion gen testing/
if _VTON not in sys.path:
    sys.path.insert(0, _VTON)
from editmask import grey_hole                                        # noqa: E402
from trainlib.gan_loss import gan_critic_loss                        # noqa: E402

_INF = {}


def get_inference(run_final):
    """Import (once) the run's real inference.py — the single shared denoising path."""
    key = os.path.abspath(run_final)
    if key not in _INF:
        spec = importlib.util.spec_from_file_location("rollout_inference", f"{run_final}/inference.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("inference", m)
        spec.loader.exec_module(m)
        _INF[key] = m
    return _INF[key]


def _agnostic_keep(sid, H, W, device, dtype):
    """Real agnostic-v3.2 keep image (in [-1,1]) + grey-hole edit mask, matching make_pred."""
    for sp in ("test", "train"):
        p = f"{_BASE}/VITON-HD-dataset/{sp}/agnostic-v3.2/{sid}.jpg"
        if os.path.exists(p):
            a = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), dtype=np.float32)
            keep = torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)[None].to(device, dtype)
            em = torch.from_numpy(grey_hole(a.astype(np.uint8)).astype(np.float32))[None, None].to(device, dtype)
            return keep, em
    return None, None


def _cloth_img(sid, device, dtype):
    """Cloth image in [-1,1] from VITON-HD cloth/{sid}.jpg — the critic's own conditioning."""
    for sp in ("test", "train"):
        p = f"{_BASE}/VITON-HD-dataset/{sp}/cloth/{sid}.jpg"
        if os.path.exists(p):
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(a).permute(2, 0, 1)[None].to(device, dtype)
    return None


def _decode(vae, lat, _vm, _vs, vae_device, dtype):
    lat5 = lat.to(vae_device, dtype).unsqueeze(2)
    return vae.decode(lat5 * _vs + _vm, return_dict=False)[0][:, :, 0].clamp(-1, 1)   # (B,3,H,W)


def rollout_pred(transformer, batch, vae, _vm, _vs, vae_device, prompt_cache, pose_cache,
                 weight_dtype, device, config, run_final, seed, grad_start=12):
    """Run the shared inference path (grad tail from `grad_start`), decode + grey-hole paste.
    Returns (pred_final (B,3,H,W) grad in tail, keep, edit_mask, sid) or (None,...) if no keep."""
    inf = get_inference(run_final)
    # prompt_cache is a lazy loader (loads {sid}_prompt_embeds.pt on __getitem__);
    # predict_sample indexes it directly, so no pre-load needed.
    model = {"transformer": transformer, "prompt_cache": prompt_cache, "pose_cache": pose_cache,
             "weight_dtype": weight_dtype, "device": device, "config": config}
    _prev = os.environ.get("RAW_FULL_PRED")
    os.environ["RAW_FULL_PRED"] = "1"                 # raw full-frame gen; we paste ourselves
    try:
        out = inf.predict_sample(model, batch, device, seed,
                                 {"num_inference_steps": 20, "grad_start_step": grad_start, "pose_mode": "normal"})
    finally:
        if _prev is None:
            os.environ.pop("RAW_FULL_PRED", None)
        else:
            os.environ["RAW_FULL_PRED"] = _prev
    C_lat = out["pred_latents"]
    pred_img = _decode(vae, C_lat, _vm, _vs, vae_device, weight_dtype)
    B, _, H, W = pred_img.shape
    sid = batch.get("image_id", batch.get("image_ids"))
    if isinstance(sid, (list, tuple)):
        sid = sid[0]
    keep, em = _agnostic_keep(sid, H, W, vae_device, weight_dtype)
    if keep is None:
        return None, None, None, sid
    pred_final = em * pred_img + (1.0 - em) * keep    # hard grey-hole paste (differentiable in tail)
    return pred_final, keep, em, sid


def rollout_gan_loss(transformer, batch, vae, _vm, _vs, vae_device, critic,
                     prompt_cache, pose_cache, weight_dtype, device, config, run_final,
                     person_image_cache, seed, grad_start=12):
    """Full rollout + paired-gap GAN loss. Returns (L_gan, pred_final, diag)."""
    pred_final, keep, em, sid = rollout_pred(
        transformer, batch, vae, _vm, _vs, vae_device, prompt_cache, pose_cache,
        weight_dtype, device, config, run_final, seed, grad_start=grad_start)
    if pred_final is None:
        return None, None, {}
    H, W = pred_final.shape[-2:]
    with torch.no_grad():
        gp = person_image_cache[sid].to(vae_device, weight_dtype)
        if gp.dim() == 3:
            gp = gp.unsqueeze(0)
        gt_final = em * gp + (1.0 - em) * keep        # GT garment in the SAME hole/keep
    cloth = _cloth_img(sid, vae_device, weight_dtype)
    if cloth is None:
        return None, None, {}
    L_gan, diag = gan_critic_loss(critic, pred_final, gt_final, cloth, em, em)
    return L_gan, pred_final, diag
