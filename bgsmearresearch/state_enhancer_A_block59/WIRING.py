# WIRING.py — VERBATIM copy of the trainlib hook blocks the state-aware enhancer injects.
# READ-ONLY snapshot. The live executed path is root garment_adapter.py + these guarded blocks
# in trainlib/. This file exists so one folder fully documents the run without opening trainlib.
#
# THESIS: not "garment net predicts garment" but "garment net EDITS what run37 is already drawing".
#   Delta = CrossAttn(Q = run37 hidden after block i, K = V = garment_net(warped_rgb+garment_latent+warped_mask))
#   H_i'  = H_i + M_garment * ZeroOut(Delta)
# The edited tokens are the FIRST N_C tokens of the image stream = the noised TARGET latent run37
# denoises (velocity is read from _o[:, :N_C] at forward.py:1086), so the edit flows straight into
# proj_out -> velocity. Injected at a LATE block (59 RunA / 55 RunB) so there is little/no downstream
# self-attention to spread garment content into bg/skin. out_proj is zero-init => step0 == run37;
# M_garment (warped-mask gate) => delta is EXACTLY 0 on non-garment tokens (bg/skin) by construction.

# ─────────────────────────────────────────────────────────────────────────────
# garment_adapter.py  (root, imported live) — model.py in this folder is the full copy.
#   __init__  mode == "state_enhancer":  Qwen garment encoder (patch/pos/temb/dummy_text/blocks)
#             + cross-attn q_proj/k_proj/v_proj (bias-free) + out_proj (ZERO-INIT).
#   xattn(H, G, M):  q=q_proj(H), k=k_proj(G), v=v_proj(G); a=SDPA(q,k,v); return M * out_proj(a).
#   install_state_enhancer_hooks(transformer, adapter, holder, tgt_blocks):
#       forward_hook on blocks[bk]; H = img[:, :N_C]; delta = adapter.xattn(H, G, M);
#       img_new = cat([H + delta, img[:, N_C:]]).   (see model.py for the exact bodies)

# ─────────────────────────────────────────────────────────────────────────────
# trainlib/forward.py  — adapter-setup block (~line 462): builds the holder each forward.
#   state._ADAPTER_HOLDER["adapter_in"] = build_adapter_input(batch,...)          # [warped_rgb, garment_latent, warped_mask] = 20ch
#   state._ADAPTER_HOLDER["adapter_M"]  = mask_tok(batch,...)                      # (B, N_C, 1) warped-garment gate
#   state._ADAPTER_HOLDER["N_C"]        = C_p_.size(1)                             # first N_C tokens = noised target latent
#   if _amode in ("spatial_adaln","controlnet","state_enhancer"):                 # <-- state_enhancer added here
#       state._ADAPTER_HOLDER["adapter_feat"] = state._GARMENT_ADAPTER._encode(_ain)   # G computed ONCE per forward (reused across rollout steps)
#
#   Schedule (50/30/10/10) at ~line 454, gates _active():
#     _rs<0.50 -> starve standard garment + branch_on   (forced handoff: enhancer is the only garment path)
#     _rs<0.80 -> both active
#     _rs<0.90 -> base solo (branch off)
#     else     -> starve + branch off
#   Velocity extracted at forward.py:1086:  _v = _o[:, :_Cp.size(1), :]           # the tokens the enhancer edits

# ─────────────────────────────────────────────────────────────────────────────
# trainlib/run.py  — build + install (~line 658):
#   from garment_adapter import (..., install_state_enhancer_hooks)
#   state._GARMENT_ADAPTER = GarmentAdapter(mode="state_enhancer", cn_blocks=_a_blocks, n_gnet_blocks=2)
#   elif _amode == "state_enhancer":
#       state._ADAPTER_HOOKS = install_state_enhancer_hooks(transformer, state._GARMENT_ADAPTER,
#                                                           state._ADAPTER_HOLDER, _a_blocks or [59])
#   param_groups.append({"params": enhancer trainable, "lr": GARMENT_ADAPTER_LR})   # co-trained with LoRA (FREEZE_LORA=0)
#   save (~line 1855): torch.save(state._GARMENT_ADAPTER.state_dict(), final_path/"garment_adapter.pt")
