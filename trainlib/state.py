"""Shared mutable runtime state (the model↔loss/orchestration bus).

These were module-level globals in the old monolithic train.py. They live here so the
split modules (builders, forward, run) can all read/assign the SAME objects. Access them
as ``state.X`` — never ``from state import X`` for the reassigned singletons (that would
copy the binding and go stale when a factory reassigns it).
"""
import os
from collections import deque as _deque

# ── singletons (reassigned by factories / main) ──
_VGG_MODEL = None
_V6_HEADS = None
_INVALID_TOKEN = None
_DEPLOY_HALO_EVAL = False  # when True, train_step does a 20-step from-noise rollout + ring-contrast metric and returns early (no training)
_REGION_ADAPTER = None
_REGION_GATE = None
_GARMENT_NET = None
_GARMENT_ENCODER = None
_GARMENT_XATTN = None
_CONTROLNET = None
_GARMENT_BRANCH = None
_QWEN_SLOT_ENRICHER = None
_QWEN_AUX_SLOT = None
_QWEN_REFINER = None
_DISC = None
_DISC_OPT = None
_CRITIC = None
_CRITIC_OPT = None
_MULTI_GAR_INJECTION = None
_MULTI_GAR_HOOKS = []
GARMENT_GATES = []        # filled by main() when USE_GARMENT_GATE=1

# ── holders (dicts mutated in place — the per-forward comm channels) ──
_HIDDEN_HOLDER = {}
_CROSS_ATTN_HOLDER = {}
_GARMENT_RESIDUAL_HOLDER = {}
_GARMENT_XATTN_HOLDER = {}
_CONTROLNET_HOLDER = {}
_GARMENT_BRANCH_HOLDER = {}
_OOTD_INJECTORS = {}
_AGN_CTRL = {}
_MULTI_GAR_HOLDER = {}
_LIMG_PARTS = {"img_g": 0.0, "img_s": 0.0, "img_b": 0.0, "img_other": 0.0, "img_k": 0.0, "img_ub": 0.0}
_GAR_BUFFER = _deque(maxlen=int(os.environ.get("CRITIC_GAR_BUFFER", "64")))
_LAST_CRIT = {}           # last critic scores (real/fake/wrong/per-corruption) for logging
