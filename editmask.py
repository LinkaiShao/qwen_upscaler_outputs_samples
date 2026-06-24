"""Shared edit-region definition: the grey hole in agnostic-v3.2 + small full-res dilation.

Single source of truth so make_pred.py, the producer, and the discriminator trainer all
agree on what "the edit region" is. Lightweight (numpy + scipy only) so dataloader workers
can import it without dragging in the model.
"""
import numpy as np
from scipy import ndimage


def grey_hole(img, dil_px=8):
    """img: HxWx3 uint8 agnostic-v3.2. Returns HxW bool edit region.
    grey fill = [128,128,128], low chroma -> largest blob only -> fill holes -> dilate px.
    NOT agnostic_mask_latent: that coarse mask carries stray single-cell specks that
    dilation blows into face/crotch blobs."""
    d = np.abs(img.astype(int) - 128).max(2)
    chroma = img.astype(int).max(2) - img.astype(int).min(2)
    g = (d < 22) & (chroma < 14)
    g = ndimage.binary_closing(g, iterations=2)
    lab, n = ndimage.label(g)
    if n:
        g = (lab == max(range(1, n + 1), key=lambda i: (lab == i).sum()))
    g = ndimage.binary_fill_holes(g)
    return ndimage.binary_dilation(g, iterations=dil_px) if dil_px > 0 else g
