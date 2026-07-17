# WIRING.py — BRANCH-CREDIT (shared by all 12 BC runs) + BC01 specifics. READ-ONLY snapshot.
#
# THE FIX for enhancer inertness (RunA/v2 were inert because delta=0 is a free optimum under flow-MSE):
# credit the adapter in the TRAINING LOSS only when the ON pass beats a matched bypassed OFF pass in
# the garment region. delta=0 -> gar_loss_ON==gar_loss_OFF -> relu(0+margin)=margin penalty, so the
# optimizer must make ON < OFF to reduce loss. bg/skin protected by keep_loss(ON matches OFF outside garment).
#
# ── trainlib/forward.py (~line 1325, right after pred_C = _fwd(C_t, sigma) [ON pass]) ──
#   if USE_BRANCH_CREDIT and grad and adapter present and adapter_M present:
#       _Mtok = state._ADAPTER_HOLDER["adapter_M"]              # (B,N_C,1) garment gate
#       state._GARMENT_ADAPTER_BYPASS = True                    # OFF pass: adapter off
#       with torch.no_grad(): pred_C_off = _fwd(C_t, sigma)     # SAME C_t/sigma/starvation, no grad
#       state._GARMENT_ADAPTER_BYPASS = prev
#       _Mg = (_Mtok>0.5); gar_on=MSE(pred_C[:,:N_C],vt)*Mg; gar_off=MSE(pred_C_off[:,:N_C],vt)*Mg   (garment tokens)
#       L_branch_credit = W_BRANCH_CREDIT * relu(gar_on - gar_off.detach() + BC_MARGIN)
#       L_keep_bc       = W_KEEP_BGSKIN   * MSE(pred_C - pred_C_off.detach()) over NON-garment C tokens (bg/skin match OFF)
#       L_delta_bc      = W_DELTA_REG     * state._ENH_DELTA_SQMEAN   (enhancer delta magnitude, from the hook)
#   -> stored in locals()/_ctx; summed in total.py.
#   ALSO forward.py ~462: state._ENH_DELTA_SQMEAN=None reset each forward.
#
# ── trainlib/forward.py _decide_gar_sched() (~line 26) ──
#   if USE_BRANCH_CREDIT: return {"starve": rand < BC_STARVE_FRAC(0.60), "branch_on": True}
#   (branch_on ALWAYS True so the ON pass always has the adapter; credit-vs-OFF answers "is it helping".)
#
# ── garment_adapter.py install_state_enhancer_hooks (delta capture) ──
#   delta = adapter.xattn(H,G,M); state._ENH_DELTA_SQMEAN = (prev or 0) + (delta**2).mean()   (grad-carrying)
#
# ── trainlib/losses/total.py (~line 205) ──
#   loss = loss + _optbc('L_branch_credit') + _optbc('L_keep_bc') + _optbc('L_delta_bc')   (None-safe)
#   metrics: bc_on, bc_off, bc_cred, bc_keep, bc_delta   (watch bc_on < bc_off = enhancer helping)
#
# ── BC01 knobs (run.sh / run_branch_credit.sh) ──
#   BC_MODE=state_enhancer  ENH_BLOCK=52  READER_CUTOFF=51 (freeze 0-51 LoRA, train 52-59)
#   USE_ENH_QUARANTINE=0 (BC01 no quarantine)  STARVE_HIGH_SIGMA=0 (BC01 normal sigma; BC12 adds 0.65-1.0)
#   W_BRANCH_CREDIT=1.0 W_KEEP_BGSKIN=1.0 W_DELTA_REG=0.01 BC_MARGIN=0.02 BC_STARVE_FRAC=0.60
#   1h 5-ID overfit (OVERFIT_SIDS, USE_FULL_TRAIN=1, TIME_BUDGET=3600). NEVER mask run37 LoRA; NEVER full LoRA.
