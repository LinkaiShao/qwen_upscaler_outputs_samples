"""In-loop deployed-halo eval: a full N-step from-noise rollout (the deployment condition
that creates the halo), decoded, then the bg ring-contrast (ring 0-8px vs surround 20-50px,
seeded → deterministic). Returns the metric for train_step's early-return. Behavior-preserving
extraction from forward.train_step."""
import os
import torch
import torch.nn.functional as F


def deployed_halo_eval(fwd, unpack_latents, person, agnostic, M_full, sigma, vae, vae_device,
                       weight_dtype, B, C, H, W, image_ids, person_image_cache, batch, device):
    with torch.no_grad():
        _Mf = M_full if M_full.dim() == 4 else M_full.unsqueeze(1)
        _ns = int(os.environ.get("DEPLOY_EVAL_STEPS", "20"))
        _s0 = float(os.environ.get("DEPLOY_EVAL_SIGMA", "0.98"))
        _sigs = torch.linspace(_s0, 0.02, _ns + 1, device=device)
        # SEEDED noise -> every eval deterministic & comparable across steps, so the [HALO]
        # trajectory is a reliable improving/converged/diverged signal (not noise).
        _eg = torch.Generator(device=person.device).manual_seed(int(os.environ.get("DEPLOY_EVAL_SEED", "1234")))
        _enoise = torch.randn(person.shape, generator=_eg, device=person.device, dtype=person.dtype)
        _x = ((1.0 - _Mf) * agnostic + _Mf * _enoise).to(person.dtype)
        for _i in range(_ns):
            _sc = torch.full((B,), float(_sigs[_i]), device=device, dtype=sigma.dtype)
            _x = _x - (float(_sigs[_i]) - float(_sigs[_i + 1])) * unpack_latents(fwd(_x, _sc), B, C, H, W)
        _x5 = _x.to(vae_device, dtype=weight_dtype).unsqueeze(2)
        _mv = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
        _sv = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(vae_device, weight_dtype)
        _pred = torch.cat([vae.decode((_x5[_d:_d+1] * _sv + _mv), return_dict=False)[0][:, :, 0]
                           for _d in range(_x5.shape[0])], 0).clamp(-1, 1).float()
        _Hi, _Wi = _pred.shape[2], _pred.shape[3]
        _gt = torch.stack([person_image_cache[iid] for iid in image_ids]).to(vae_device).float()
        if tuple(_gt.shape[2:]) != (_Hi, _Wi):
            _gt = F.interpolate(_gt, size=(_Hi, _Wi), mode="bilinear", align_corners=False)
        _pb = batch["parse_bg"].to(vae_device, weight_dtype)
        if _pb.dim() == 3: _pb = _pb.unsqueeze(1)
        _pb = F.interpolate((_pb > 0.5).float(), size=(_Hi, _Wi), mode="nearest")
        _hole = F.interpolate(_Mf.float().to(vae_device), size=(_Hi, _Wi), mode="nearest")
        _person_m = (1.0 - _pb)                                       # ¬bg = person
        _inner = (_pb * _hole)                                        # generated bg in the hole
        def _dil(m, px): return F.max_pool2d(m, 2 * px + 1, 1, px).clamp(0, 1)
        _ring = ((_dil(_person_m, 8) - _person_m).clamp(0, 1) * _inner)              # bg within 8px of person
        _surr = ((_dil(_person_m, 50) - _dil(_person_m, 20)).clamp(0, 1) * _inner)   # bg 20-50px out
        def _lum(x):
            x01 = (x + 1.0) / 2.0
            return (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]) * 100.0
        _dL = _lum(_pred) - _lum(_gt)
        _rv = float((_dL * _ring).sum() / (_ring.sum() + 1e-6))
        _su = float((_dL * _surr).sum() / (_surr.sum() + 1e-6))
    return {"deploy_halo": _rv - _su, "halo_ring": _rv, "halo_surr": _su}
