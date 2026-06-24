"""Module-level config/constants (computed at import)."""
import os, sys
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
try:
    from multi_block_garment import (MultiBlockGarmentInjection, install_multi_block_hooks,
                                      build_spatial_mask_from_warped, per_block_norms_table)
except Exception:
    pass
from trainlib import state

from trainlib.models import (GarmentCrossAttn, GarmentLatentEnhancer, GarmentNet, GarmentNetAdaLN, GarmentNetCrossAttn, GarmentNetOutput, GarmentRepairGate, GarmentSlotEncoder, OOTDInjector, PatchDiscriminator, QwenAuxSlotEncoder, QwenControlNet, QwenGarmentEncoder, QwenGarmentNetAdaLN, QwenGarmentNetBlockCopyAdaLN, QwenGarmentNetCrossAttn, QwenLatentRefiner, QwenSlotEnricher, V6Heads, _make_qwen_block, perceptual_loss)
from trainlib.data import (pack_latents, unpack_latents, vae_decode_to_pil, precompute_rough_pils, precompute_prompt_embeds, load_pose_latents, VTONDataset, collate_fn)

BASE = "/home/link/Desktop/Code/fashion gen testing"
LOCAL_CACHE = f"{BASE}/vtonautoresearch/local_cache"
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "3600"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",   "400"))
TRAIN_IDS = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
VAL_IDS = []
if int(os.environ.get("USE_FULL_TRAIN", "0")):
    from my_vton.v1.precompute import list_image_ids as _list_image_ids
    BASE = "/home/link/Desktop/Code/fashion gen testing"
    _ALL_TRAIN = _list_image_ids(f"{BASE}/VITON-HD-dataset", "train")
    _RESERVED_VTON_IDS = {"00006_00", "00008_00", "00013_00", "00017_00", "00034_00"}
    _ALL_TRAIN = [i for i in _ALL_TRAIN if i not in _RESERVED_VTON_IDS]
    TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "0")) or len(_ALL_TRAIN)
    VAL_LIMIT   = int(os.environ.get("VAL_LIMIT",   "0"))
    import random as _rng
    _rng_seed = _rng.Random(0)
    _ALL_TRAIN_SHUF = list(_ALL_TRAIN); _rng_seed.shuffle(_ALL_TRAIN_SHUF)
    TRAIN_IDS = _ALL_TRAIN_SHUF[:TRAIN_LIMIT]
    _HOLDOUT  = _ALL_TRAIN_SHUF[TRAIN_LIMIT:]
    # OVERFIT_SIDS: comma-separated list of reserved SIDs to use as training set
    # (overrides the reserved filter for one-sample overfit probes).
    _override_sids = os.environ.get("OVERFIT_SIDS", "").strip()
    if _override_sids:
        TRAIN_IDS = [s.strip() for s in _override_sids.split(",") if s.strip()]
        _HOLDOUT = []
        print(f"[OVERFIT_SIDS] training restricted to: {TRAIN_IDS}")
    if int(os.environ.get("VAL_USE_RESERVED", "0")):
        # Use the 5 reserved test IDs for validation (5_28/instructions7).
        # These are guaranteed not in TRAIN_IDS (already filtered above).
        VAL_IDS = sorted(_RESERVED_VTON_IDS)
    elif VAL_LIMIT > 0:
        VAL_IDS = _HOLDOUT[:VAL_LIMIT]
    print(f"[USE_FULL_TRAIN] TRAIN_IDS={len(TRAIN_IDS)} HOLDOUT={len(_HOLDOUT)} VAL_IDS={len(VAL_IDS)}")
FIXED_PROMPT = os.environ.get(
    "FIXED_PROMPT",
    "A realistic photo of the same person wearing the target garment. "
    "Only modify the upper-body region indicated by the white masked area in the input image. "
    "Outside that masked region (face, hair, arms, hands, pants, background) must remain identical to the input."
)
import trainlib.data as _datamod
_datamod.TRAIN_IDS = TRAIN_IDS
_datamod.FIXED_PROMPT = FIXED_PROMPT
ALL_SLOT_NAMES = ["agnostic", "pose", "rough", "garment", "silhouette", "body_rough"]
SLOT_ORDER = os.environ.get("SLOT_ORDER", "agnostic,pose,rough,garment").split(",")
assert set(SLOT_ORDER).issubset(set(ALL_SLOT_NAMES)), f"SLOT_ORDER must be a subset of {ALL_SLOT_NAMES}, got {SLOT_ORDER}"
SLOT_INDEX = {n: i for i, n in enumerate(ALL_SLOT_NAMES)}
SLOT_ORDER_IDX = [SLOT_INDEX[n] for n in SLOT_ORDER]
NUM_SLOTS = len(SLOT_ORDER) + 1   # +1 for C position 0
ATTN_MASK_HOLDER = {"mask": None}
INFERENCE_TEMPLATE = os.environ.get("INFERENCE_TEMPLATE_PATH",
                                    os.path.join(BASE, "vtonautoresearch", "inference_template.py"))
if __name__ == "__main__":
    main()
