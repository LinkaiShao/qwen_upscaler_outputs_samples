"""
Region-aware bg post-correction (NO training).
For generated-background pixels: replace low-frequency color with a smooth field
interpolated from the REAL visible bg (the agnostic has this outside the person),
keep Qwen's high-frequency, feather edges. Garment/skin untouched.

    pred_bg          = generated pixels classified as bg
    target_bg_field  = smooth field from real visible bg (nearest-fill + blur)
    pred_low         = blur(pred_bg);  pred_high = pred_bg - pred_low
    fixed_bg         = target_bg_field + pred_high * strength
    final            = pred (garment/skin) ; fixed_bg (generated bg) ; feathered
"""
import sys, numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
BASE="/home/link/Desktop/Code/fashion gen testing"
STRENGTH=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
FIELD_SIGMA=float(sys.argv[2]) if len(sys.argv)>2 else 25.0
def lum(x): return 0.299*x[...,0]+0.587*x[...,1]+0.114*x[...,2]
def cloud(gen,gt,pv):
    bg=~(pv>0); d=distance_transform_edt(bg); band=bg&(d>=8)&(d<=50)
    dL=lum(gen.astype(np.float32))-lum(gt.astype(np.float32))
    return float(abs(gaussian_filter(dL,15)[band]).mean())

def correct(gen, gt, pv):
    H,W=gen.shape[:2]
    bg = (pv==0)                                   # parse background
    fg = ~bg
    # "real visible bg" = the bg the deployment actually has: real pixels OUTSIDE the
    # person silhouette, away from the near-silhouette darkened band. Use gt there
    # (== agnostic bg at deploy). Exclude a dilation ring around the person so the
    # field is built only from clean far bg, then interpolate inward.
    dist_fg = distance_transform_edt(bg)           # dist from person
    visible = bg & (dist_fg > 12)                  # clean far bg (agnostic-real at deploy)
    # build target field: nearest-fill the visible real bg over the whole frame, blur
    src = gt.astype(np.float32).copy()
    idx = distance_transform_edt(~visible, return_distances=False, return_indices=True)
    filled = src[tuple(idx)]                        # nearest visible-bg color everywhere
    field = np.stack([gaussian_filter(filled[...,c], FIELD_SIGMA) for c in range(3)], -1)
    # correct generated bg: low from field, high from gen
    g = gen.astype(np.float32)
    g_low = np.stack([gaussian_filter(g[...,c], FIELD_SIGMA) for c in range(3)], -1)
    g_high = g - g_low
    fixed_bg = field + g_high*STRENGTH
    # feathered bg mask (soft, so garment/skin edges untouched)
    m = gaussian_filter(bg.astype(np.float32), 2.0)[...,None]
    out = g*(1-m) + fixed_bg*m
    return out.clip(0,255).astype(np.uint8)

srcdir=sys.argv[3] if len(sys.argv)>3 else "/tmp/rawdep_soar"
print(f"strength={STRENGTH} field_sigma={FIELD_SIGMA} src={srcdir}")
before=[]; after=[]
for s5 in ["00006","00008","00013","00017","00034"]:
    s=f"{s5}_00"
    im=Image.open(f"{srcdir}/{s}_raw_dep_gt.png").convert("RGB"); W3=im.size[0]//3
    gen=np.asarray(im.crop((0,0,W3,im.size[1])))
    gt=Image.open(f"{BASE}/VITON-HD-dataset/test/image/{s}.jpg").convert("RGB")
    W,Hh=gt.size; gen=np.asarray(Image.fromarray(gen).resize((W,Hh))); gt=np.asarray(gt)
    pv=np.asarray(Image.open(f"{BASE}/VITON-HD-dataset/test/image-parse-v3/{s}.png").resize((W,Hh),Image.NEAREST))
    if pv.ndim==3: pv=pv[...,0]
    c0=cloud(gen,gt,pv); fixed=correct(gen,gt,pv); c1=cloud(fixed,gt,pv)
    before.append(c0); after.append(c1)
    Image.fromarray(np.concatenate([gen,fixed,gt],1)).save(f"/tmp/postcorr_{s5}.png")
    print(f"  {s5}: CLOUD {c0:.2f} -> {c1:.2f}")
print(f"  MEAN CLOUD {np.mean(before):.2f} -> {np.mean(after):.2f}  (soar 8.19, if5 3.4, floor ~0)")
