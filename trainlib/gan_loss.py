"""GAN loss core for the rollout branch — paired GT→pred critic gap (sections 3,4,6).

Uses the frozen GarmentDiscriminator (discriminator_v01) the same way it was trained:
  D(img, cloth, edit_mask) -> (patch_logits (B,1,h,w), global (B,1)); higher = more real.

The generator is pushed to close the gap to its OWN GT's critic score (paired), with a
tolerant soft margin so patches stop being pushed once near GT — safer than -D(fake).mean(),
which invites unbounded critic exploitation. Pooling: mean over the generated region + the
worst 25% of patches (hands-free hard-example focus). lambda_gan is set from gradient
magnitude so the critic contributes a fixed fraction of the generator gradient.
"""
import os
import sys
import torch
import torch.nn.functional as F

_VTON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # vtonautoresearch/
if _VTON not in sys.path:
    sys.path.insert(0, _VTON)


def load_gan_critic(device, ckpt=None):
    """Load the frozen GarmentDiscriminator (discriminator_v01/D_best.pt)."""
    from discriminator import GarmentDiscriminator
    ckpt = ckpt or os.environ.get("GAN_CRITIC_PT",
                                  f"{_VTON}/runs/discriminator_v01/D_best.pt")
    D = GarmentDiscriminator().to(device).eval()
    D.load_state_dict(torch.load(ckpt, map_location=device))
    for p in D.parameters():
        p.requires_grad_(False)
    return D


def _top_fraction_mean(x, frac):
    """Mean of the largest `frac` fraction of a 1-D tensor (worst-patch pressure)."""
    if x.numel() == 0:
        return x.new_zeros(())
    k = max(1, int(round(x.numel() * frac)))
    return torch.topk(x, k, largest=True).values.mean()


def gan_critic_loss(critic, pred_img, gt_img, cloth_img, edit_mask, M_model,
                    soft_thr=0.5, soft_scale=0.5, hard_frac=0.25):
    """Paired soft-margin GAN loss over the generated region.

    pred_img / gt_img: (B,3,H,W) in [-1,1]; cloth_img: (B,3,*,*) in [-1,1];
    edit_mask / M_model: (B,1,H,W) in [0,1] (M_model = generated region).
    Returns (L_gan, diag dict). Only pred_img carries gradient.
    """
    _cd = next(critic.parameters()).dtype                  # critic runs in its own dtype (fp32)
    pred_img = pred_img.to(_cd); gt_img = gt_img.to(_cd); cloth_img = cloth_img.to(_cd); edit_mask = edit_mask.to(_cd)
    d_gt, _ = critic(gt_img, cloth_img, edit_mask)          # (B,1,h,w) — no grad needed
    d_fake, _ = critic(pred_img, cloth_img, edit_mask)      # (B,1,h,w) — grad to generator
    gap = d_gt.detach() - d_fake                            # >0 means pred scores worse than GT
    gan_map = F.softplus((gap - soft_thr) / soft_scale) * soft_scale   # tolerant: ~0 once near GT
    # operate only inside the generated region, at the critic-map resolution
    h, w = gan_map.shape[-2:]
    M_crit = (F.interpolate(M_model.float(), size=(h, w), mode="area") > 0.5)
    valid = gan_map[M_crit]
    if valid.numel() == 0:
        z = gan_map.sum() * 0.0
        return z, {"d_gt": float(d_gt.mean()), "d_fake": float(d_fake.mean()), "gap": 0.0, "npatch": 0}
    L_gan_mean = valid.mean()
    L_gan_hard = _top_fraction_mean(valid, hard_frac)
    L_gan = 0.5 * L_gan_mean + 0.5 * L_gan_hard
    diag = {"d_gt": float(d_gt.mean()), "d_fake": float(d_fake.mean().detach()),
            "gap": float(gap[M_crit].mean().detach()), "npatch": int(M_crit.sum())}
    return L_gan, diag


@torch.no_grad()
def grad_norm_of(loss, params, retain=True):
    """L2 norm of d(loss)/d(params) over trainable params (for lambda calibration)."""
    ps = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, ps, retain_graph=retain, allow_unused=True)
    sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return float(torch.sqrt(sq + 1e-12))


def calibrate_lambda_gan(g_base, g_gan, target_ratio=0.10):
    """lambda such that lambda*g_gan ≈ target_ratio*g_base (critic = ~10% of gen gradient)."""
    if g_gan <= 1e-9:
        return 0.0
    return target_ratio * g_base / g_gan
