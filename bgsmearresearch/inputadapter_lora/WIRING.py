# WIRING.py — VERBATIM trainlib hooks for the garment ADAPTER (READ-ONLY snapshot; not executed).
# Live path = guarded hooks in trainlib importing root garment_adapter.py. Read this to see the whole footprint.

# ==== [1] trainlib/state.py singletons ====
_GARMENT_ADAPTER = None        # tonight's 6 runs: unified garment adapter (input/token/slot/adaln/controlnet)
_GARMENT_ADAPTER_BYPASS = False  # eval branch-off / base-solo: True -> adapter hooks return run37 unchanged
_ADAPTER_HOLDER = {}           # per-step: adapter_in / adapter_M / adapter_feat / N_C
_ADAPTER_HOOKS = []
_GAR_SCHED = None              # per-step schedule {"starve": bool, "branch_on": bool}

# ==== [2] trainlib/forward.py : per-step 50/30/10/10 SCHEDULE + asymmetric STARVATION + adapter holder ====
    # ── tonight's-6-runs COMMON PROTOCOL: per-step schedule + garment-adapter holder ──
    #    Schedule (random): 50% forced-handoff (starve std garment, adapter ON) / 30% both ON /
    #    10% base-solo (adapter OFF, std garment present) / 10% both OFF. Adapter always gets the
    #    CLEAN aligned garment; branch_on gates its contribution (base-solo == run37).
    if int(os.environ.get("USE_GARMENT_ADAPTER", "0")) and getattr(state, "_GARMENT_ADAPTER", None) is not None:
        import garment_adapter as _ga
        if int(os.environ.get("GARMENT_SCHEDULE", "0")) and torch.is_grad_enabled() and not getattr(state, "_DEPLOY_HALO_EVAL", False):
            _rs = float(torch.rand(()))
            if _rs < 0.50:   state._GAR_SCHED = {"starve": True,  "branch_on": True}
            elif _rs < 0.80: state._GAR_SCHED = {"starve": False, "branch_on": True}
            elif _rs < 0.90: state._GAR_SCHED = {"starve": False, "branch_on": False}
            else:            state._GAR_SCHED = {"starve": True,  "branch_on": False}
        else:
            state._GAR_SCHED = {"starve": bool(os.environ.get("GARMENT_SLOT_CORRUPT", "")), "branch_on": True}
        state._ADAPTER_HOLDER.clear()
        _ain = _ga.build_adapter_input(batch, device, weight_dtype, B, H, W)
        state._ADAPTER_HOLDER["adapter_in"] = _ain
        state._ADAPTER_HOLDER["adapter_M"] = _ga.mask_tok(batch, device, weight_dtype, B, C, H, W)
        state._ADAPTER_HOLDER["N_C"] = C_p_.size(1)
        _amode = os.environ.get("GARMENT_ADAPTER_MODE", "input_hidden")
        if _amode in ("spatial_adaln", "controlnet"):
            state._ADAPTER_HOLDER["adapter_feat"] = state._GARMENT_ADAPTER._encode(_ain.to(torch.float32))
        # latent_token / detail_slot injections are applied inside _fwd / slot-build (see below).

    # ── runs 69/71: corrupt the MAIN conditioning garment info so the base can't SHORTCUT to
    #    the clean garment latent — forcing reliance on the gnet branch (whose garment input
    #    stays CLEAN: OOTD/pose gnets read batch["garment_latent"] directly, NOT `garment`).
    #    Only the garment (+rough) MAIN slots are touched; bg/skin agnostic context untouched.
    #    GARMENT_SLOT_CORRUPT=zero|blur ; GARMENT_SLOT_CORRUPT_P = per-step prob (default 1.0).
    #    Decided ONCE per step (before _fwd) so pred_C/base/wrong stay consistent; train-only.
    #    Adapter runs: gated by the schedule's "starve" flag (only forced-handoff/both-off starve). ──
    _gsc = os.environ.get("GARMENT_SLOT_CORRUPT", "")
    _sched_starve = (getattr(state, "_GAR_SCHED", None) is None) or state._GAR_SCHED.get("starve", True)
    if _gsc and _sched_starve and torch.is_grad_enabled() and not getattr(state, "_DEPLOY_HALO_EVAL", False) \
            and float(torch.rand(())) < float(os.environ.get("GARMENT_SLOT_CORRUPT_P", "1.0")):

# ==== [3] trainlib/forward.py : _fwd latent_token injection (Run3) ====
        # Run3 latent_token: add M * zero_init_64d_adapter(aligned garment) to the packed C-slot (64-d) before the transformer.
        if (int(os.environ.get("USE_GARMENT_ADAPTER", "0")) and os.environ.get("GARMENT_ADAPTER_MODE", "") == "latent_token"
                and getattr(state, "_GARMENT_ADAPTER", None) is not None):
            import garment_adapter as _ga2
            if _ga2._active() and state._ADAPTER_HOLDER.get("adapter_in") is not None:
                _res64 = state._GARMENT_ADAPTER.project(state._ADAPTER_HOLDER["adapter_in"].to(torch.float32))
                _Cp = _Cp + (state._ADAPTER_HOLDER["adapter_M"] * _res64).to(_Cp.dtype)

# ==== [4] trainlib/forward.py : detail_slot slot replacement (Run4) ====
    # Run4 detail_slot: REPLACE the standard garment conditioning slot with a learned detail slot
    # (from the CLEAN aligned garment). When base-solo/bypass, keep the standard garment slot.
    if (int(os.environ.get("USE_GARMENT_ADAPTER", "0")) and os.environ.get("GARMENT_ADAPTER_MODE", "") == "detail_slot"
            and getattr(state, "_GARMENT_ADAPTER", None) is not None):
        import garment_adapter as _ga3
        if _ga3._active() and state._ADAPTER_HOLDER.get("adapter_in") is not None:
            slot_tensors["garment"] = state._GARMENT_ADAPTER.project(state._ADAPTER_HOLDER["adapter_in"].to(torch.float32)).to(gar_p.dtype)

# ==== [5] trainlib/run.py : build the unified adapter + install hooks (guarded USE_GARMENT_ADAPTER) ====
    # ── tonight's 6 runs: UNIFIED garment adapter (input/token/slot/adaln/controlnet), zero-init ──
    if int(os.environ.get("USE_GARMENT_ADAPTER", "0")):
        from garment_adapter import (GarmentAdapter, install_input_hidden_hook,
                                      install_spatial_adaln_hooks, install_controlnet_hooks)
        _amode = os.environ.get("GARMENT_ADAPTER_MODE", "input_hidden")
        _a_blocks = [int(x) for x in os.environ.get("GARMENT_ADAPTER_BLOCKS", "").split(",") if x.strip()] or None
        state._GARMENT_ADAPTER = GarmentAdapter(mode=_amode, cn_blocks=_a_blocks,
            n_gnet_blocks=int(os.environ.get("GARMENT_ADAPTER_GNET_BLOCKS", "2"))).to(device, torch.float32)
        _a_init = os.environ.get("GARMENT_ADAPTER_INIT_PATH", "")
        if _a_init and os.path.exists(_a_init):
            state._GARMENT_ADAPTER.load_state_dict(torch.load(_a_init, map_location="cpu", weights_only=True), strict=False)
            log.info(f"garment_adapter: warm-started from {_a_init}")
        if _amode == "input_hidden":
            state._ADAPTER_HOOKS = install_input_hidden_hook(transformer, state._GARMENT_ADAPTER, state._ADAPTER_HOLDER)
        elif _amode == "spatial_adaln":
            state._ADAPTER_HOOKS = install_spatial_adaln_hooks(transformer, state._GARMENT_ADAPTER, state._ADAPTER_HOLDER, _a_blocks or [12, 20, 28, 36])
        elif _amode == "controlnet":
            state._ADAPTER_HOOKS = install_controlnet_hooks(transformer, state._GARMENT_ADAPTER, state._ADAPTER_HOLDER, _a_blocks or [8, 16, 24, 32])
        _a_lr = float(os.environ.get("GARMENT_ADAPTER_LR", "1e-4"))
        _a_tr = [p_ for p_ in state._GARMENT_ADAPTER.parameters() if p_.requires_grad]
        param_groups.append({"params": _a_tr, "lr": _a_lr})
        log.info(f"garment_adapter: mode={_amode} (zero-init) | {sum(p_.numel() for p_ in _a_tr):,} params @ {_a_lr:.1e} | hooks={len(state._ADAPTER_HOOKS)}")

# ==== [6] trainlib/run.py : save garment_adapter.pt ====
    # tonight's 6 runs: garment adapter ckpt (final)
    if int(os.environ.get("USE_GARMENT_ADAPTER", "0")) and getattr(state, "_GARMENT_ADAPTER", None) is not None:
        torch.save({k: v.cpu() for k, v in state._GARMENT_ADAPTER.state_dict().items()},
                   os.path.join(final_path, "garment_adapter.pt"))
        log.info("Saved garment_adapter.pt")

# ==== [7] the injection HOOKS live in model.py (garment_adapter.py): install_input_hidden_hook /
#          install_spatial_adaln_hooks / install_controlnet_hooks + _active() (branch-off/base-solo gate). ====
