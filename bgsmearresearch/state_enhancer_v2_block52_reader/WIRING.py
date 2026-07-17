"""WIRING.py — verbatim copy of the NEW/CHANGED code blocks for
state_enhancer_v2_block52_reader. NOT executed; reference snapshot of exactly what was
wired into the main trainlib, each block labeled with its source file + function/site.

All blocks are env-gated so existing runs are byte-identical when the new flags
(STARVE_HIGH_SIGMA, ENH_SCHED_V2, USE_ENH_QUARANTINE) are unset.

New flags:
  ENH_SCHED_V2=1          -> 60/25/10/5 schedule (else original 50/30/10/10)
  STARVE_HIGH_SIGMA=1     -> move schedule decision before sigma + resample starve-step sigma
  STARVE_SIGMA_LO=0.5     -> starve sigma ~ U[LO, 1.0] (mid-to-high)
  USE_ENH_QUARANTINE=1    -> install pair-mask attn quarantine on ENH_QUARANTINE_BLOCKS
  ENH_QUARANTINE_BLOCKS="53,54,55,56,57,58,59"
  GARMENT_ADAPTER_BLOCKS="52"  (state_enhancer injection block)
  EARLY_BLOCK_CUTOFF=51 LR_EARLY_MULT=0.0  (freeze 0-51 LoRA, train 52-59 at args.lr)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/state.py — module-level holders (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
_GAR_SCHED_SET = False         # ENH v2 (A): True once _GAR_SCHED is decided THIS forward (moved before sigma when STARVE_HIGH_SIGMA); reset each forward
_ENH_QUARANTINE_HOLDER = {}    # ENH v2 (B): {"bias": (B,1,S,S)} additive attn bias blocking bg/skin C-queries from garment C-keys (built once/forward)


# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/forward.py — module-level helper (NEW), above _debug_finite
# ═══════════════════════════════════════════════════════════════════════════════
def _decide_gar_sched():
    """ENH v2 (A): sample the per-step garment schedule {starve, branch_on}.
    ENH_SCHED_V2=1 -> 60/25/10/5 (branch-only / both / base-solo / both-off).
    Unset -> the original 50/30/10/10 (byte-identical draw to the old inline decision)."""
    _rs = float(torch.rand(()))
    if int(os.environ.get("ENH_SCHED_V2", "0")):
        if   _rs < 0.60: return {"starve": True,  "branch_on": True}    # branch-only (forced handoff)
        elif _rs < 0.85: return {"starve": False, "branch_on": True}    # both on
        elif _rs < 0.95: return {"starve": False, "branch_on": False}   # base solo (== run37)
        else:            return {"starve": True,  "branch_on": False}   # both off
    if   _rs < 0.50: return {"starve": True,  "branch_on": True}
    elif _rs < 0.80: return {"starve": False, "branch_on": True}
    elif _rs < 0.90: return {"starve": False, "branch_on": False}
    return {"starve": True, "branch_on": False}


# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/forward.py — train_step, immediately BEFORE the sigma sample (~L284) (NEW/CHANGED)
# ═══════════════════════════════════════════════════════════════════════════════
    # ── ENH v2 (A): MOVE the garment-schedule decision to BEFORE sigma sampling so a starve
    #    step can be noised at mid/high sigma (which corrupts GT-garment detail in C_t and stops
    #    the main net solving without the enhancer). Gated by STARVE_HIGH_SIGMA so that with the
    #    flag UNSET the decision stays at its original site (~L470) => byte-identical RNG. The
    #    holder build at ~L470 must stay put (it needs C_p_), so ONLY the decision moves up. ──
    state._GAR_SCHED_SET = False
    if int(os.environ.get("STARVE_HIGH_SIGMA", "0")) and int(os.environ.get("USE_GARMENT_ADAPTER", "0")) \
            and int(os.environ.get("GARMENT_SCHEDULE", "0")) and torch.is_grad_enabled() \
            and not getattr(state, "_DEPLOY_HALO_EVAL", False):
        state._GAR_SCHED = _decide_gar_sched()
        state._GAR_SCHED_SET = True

    sigma = torch.distributions.Beta(sigma_beta_alpha, sigma_beta_beta).sample((B,)).to(device=device, dtype=weight_dtype)
    # ── ENH v2 (A): on a starve step (schedule decided above), resample sigma ~U[LO,1.0]
    #    (LO=STARVE_SIGMA_LO, default 0.5 = mid-to-high). The high-sigma-noised C_t IS the
    #    "corrupt garment detail in C_t" — same effect, cleaner. Per-sample (B). ──
    if getattr(state, "_GAR_SCHED_SET", False) and state._GAR_SCHED.get("starve") \
            and int(os.environ.get("STARVE_HIGH_SIGMA", "0")) and transformer.training and torch.is_grad_enabled():
        _slo = float(os.environ.get("STARVE_SIGMA_LO", "0.5"))
        sigma = torch.empty(B, device=device).uniform_(_slo, 1.0).to(weight_dtype)
        if not globals().get("_STARVE_SIG_LOGGED"):
            print(f"[enh_v2] STARVE_HIGH_SIGMA: starve steps sigma~U[{_slo},1.0] "
                  f"(this step mean={float(sigma.mean()):.3f})", flush=True)
            globals()["_STARVE_SIG_LOGGED"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/forward.py — train_step, the schedule-decision at the adapter-holder build (~L470) (CHANGED)
#   (only the if-branch that decides _GAR_SCHED; the holder build below it is UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════
        if int(os.environ.get("GARMENT_SCHEDULE", "0")) and torch.is_grad_enabled() and not getattr(state, "_DEPLOY_HALO_EVAL", False):
            # ENH v2 (A): if the decision was already made BEFORE sigma (STARVE_HIGH_SIGMA path),
            # reuse it — do NOT re-roll. Otherwise decide here exactly as before (byte-identical).
            if not getattr(state, "_GAR_SCHED_SET", False):
                state._GAR_SCHED = _decide_gar_sched()
                state._GAR_SCHED_SET = True
        else:
            state._GAR_SCHED = {"starve": bool(os.environ.get("GARMENT_SLOT_CORRUPT", "")), "branch_on": True}


# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/forward.py — train_step, right AFTER `hidden = torch.cat([C_p_] + cond_seq, ...)` (~L740) (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
    # ── ENH v2 (B): build the attention-QUARANTINE bias ONCE per forward and share it across
    #    the quarantine blocks via the holder. bg/skin QUERY tokens among the first N_C get
    #    -1e4 on GARMENT KEY tokens among the first N_C, so injected garment features cannot
    #    smear into bg/skin downstream of the block-52 injection. Guarded; default off. ──
    if int(os.environ.get("USE_ENH_QUARANTINE", "0")):
        state._ENH_QUARANTINE_HOLDER.pop("bias", None)                          # never reuse a stale-shape bias
        _qM = state._ADAPTER_HOLDER.get("adapter_M")                            # (B, N_C, 1) warped-garment gate
        if _qM is not None:
            import enh_quarantine as _eq
            state._ENH_QUARANTINE_HOLDER["bias"] = _eq.build_quarantine_bias(
                _qM, _qM.shape[1], pe_batch.shape[1], hidden.size(1), device, weight_dtype)
            if not globals().get("_ENH_Q_LOGGED"):
                _tot = pe_batch.shape[1] + hidden.size(1)
                print(f"[enh_v2] quarantine bias: N_C={_qM.shape[1]} seq_txt={pe_batch.shape[1]} "
                      f"total_seq={_tot} shape={tuple(state._ENH_QUARANTINE_HOLDER['bias'].shape)}", flush=True)
                globals()["_ENH_Q_LOGGED"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# trainlib/run.py — main(), right AFTER the garment_adapter install/log line (~L682) (NEW)
#   (per-block LR is UNCHANGED existing code: EARLY_BLOCK_CUTOFF=51 LR_EARLY_MULT=0.0 ->
#    3-tier log "0..51 @ 0.00e+00 | 52..59 @ <lr> | 60..59 @ ...")
# ═══════════════════════════════════════════════════════════════════════════════
    # ── ENH v2 (B): attention QUARANTINE — install EnhQuarantineProcessor on the downstream
    #    blocks so bg/skin C-queries cannot attend garment C-keys (no smear from the block-52
    #    injection). Guarded by USE_ENH_QUARANTINE; default off => all other runs untouched. ──
    if int(os.environ.get("USE_ENH_QUARANTINE", "0")):
        from enh_quarantine import install_quarantine_processors
        _q_blocks = [int(x) for x in os.environ.get("ENH_QUARANTINE_BLOCKS", "53,54,55,56,57,58,59").split(",") if x.strip()]
        _nq = install_quarantine_processors(transformer, _q_blocks)
        log.info(f"enh_quarantine: installed EnhQuarantineProcessor on blocks {_q_blocks} "
                 f"({_nq} attn processors) | bg/skin C-queries blocked from garment C-keys")


# ═══════════════════════════════════════════════════════════════════════════════
# enh_quarantine.py (root, NEW FILE) — see the sibling enh_quarantine.py in this folder for
# the full build_quarantine_bias() + EnhQuarantineProcessor + install_quarantine_processors().
# The processor only applies the bias when garment_adapter._active() is True, so base-solo /
# branch-off / eval passes stay byte-identical to run37.
# ═══════════════════════════════════════════════════════════════════════════════
