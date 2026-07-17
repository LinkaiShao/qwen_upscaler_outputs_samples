# WIRING.py — VERBATIM COPY of the trainlib hook code this run injects (READ-ONLY snapshot).
# NOT executed. The live path = the guarded hooks in trainlib importing root mask_traj_blend.py.
# Read this to see the ENTIRE trainlib footprint of this run without opening trainlib.
# (Function-body indentation is preserved verbatim, so this file is for READING, not importing.)

# ==== [1] trainlib/state.py : module singletons ====
_GARMENT_VEL_DENOISER = None  # masked-trajectory-blend: garment-region velocity denoiser
_MASK_TRAJ_BYPASS = False      # two-track rollout: True -> blend returns pure run37 (the x_base track)

# ==== [2] trainlib/data.py : VTONDataset.__getitem__ + collate_fn (POSE_USE_WARPED_RGB=1) ====
        if int(os.environ.get("POSE_USE_WARPED_RGB", "0")):
            _wrgb_p = os.path.join(self.latent_dir, f"{i}_warped_rgb_128.pt")
            if os.path.exists(_wrgb_p):
                item["warped_garment_rgb"] = torch.load(_wrgb_p, weights_only=True).float() * 2.0 - 1.0
            else:
                item["warped_garment_rgb"] = torch.zeros(3, 128, 96)             # neutral (mid-grey in [-1,1])
    if "warped_garment_rgb" in batch[0]:
        out["warped_garment_rgb"] = torch.stack([b["warped_garment_rgb"] for b in batch])

# ==== [3] trainlib/forward.py : train_step._fwd closure — THE MASK BLEND HOOK ====
    def _fwd(C_lat_in, sig_in):
        _Cp = pack_latents(C_lat_in, B, C, H, W)
        _hid = torch.cat([_Cp] + cond_seq, dim=1)
        _o = transformer(hidden_states=_hid, timestep=sig_in, encoder_hidden_states=pe_batch,
                         encoder_hidden_states_mask=pm_batch, img_shapes=img_shapes,
                         txt_seq_lens=txt_seq_lens, return_dict=False)[0]
        _v = _o[:, :_Cp.size(1), :]                              # v_run37 (packed)
        # ── MASKED TRAJECTORY BLEND: v_final = v_run37*(1-M) + v_garment*M (bg/skin locked). ──
        if int(os.environ.get("USE_MASK_TRAJ_BLEND", "0")) and getattr(state, "_GARMENT_VEL_DENOISER", None) is not None:
            from mask_traj_blend import blend_velocity
            _v = blend_velocity(_v, C_lat_in, sig_in, batch, B, C, H, W, device, weight_dtype)
        return _v

# ==== [4] trainlib/run.py : build GarmentVelocityDenoiser (guarded USE_MASK_TRAJ_BLEND) ====
    # ── MASKED TRAJECTORY BLENDING: separate garment velocity denoiser, blended with frozen run37. ──
    if int(os.environ.get("USE_MASK_TRAJ_BLEND", "0")):
        from mask_traj_blend import GarmentVelocityDenoiser
        _mt_dtype = torch.float32                                        # trainable -> fp32 (avoid bf16 Adam quantize)
        _mt_nb = int(os.environ.get("MASK_TRAJ_BLOCKS", "4"))
        state._GARMENT_VEL_DENOISER = GarmentVelocityDenoiser(n_blocks=_mt_nb).to(device, _mt_dtype)
        _mt_init = os.environ.get("MASK_TRAJ_INIT_PATH", "")
        if _mt_init and os.path.exists(_mt_init):
            state._GARMENT_VEL_DENOISER.load_state_dict(torch.load(_mt_init, map_location="cpu", weights_only=True), strict=False)
            log.info(f"mask_traj: denoiser warm-started from {_mt_init}")
        _mt_lr = float(os.environ.get("MASK_TRAJ_LR", "1e-4"))
        _mt_tr = [p_ for p_ in state._GARMENT_VEL_DENOISER.parameters() if p_.requires_grad]
        param_groups.append({"params": _mt_tr, "lr": _mt_lr})
        log.info(f"mask_traj: GarmentVelocityDenoiser {_mt_nb} blocks (fp32) | {sum(p_.numel() for p_ in _mt_tr):,} params @ {_mt_lr:.1e} "
                 f"| vel_head ZERO-INIT (v_garment=0 at step0 -> v_final==run37)")

# ==== [5] trainlib/run.py : save garment_vel_denoiser.pt (final + per-ckpt snapshot) ====
    # masked-trajectory-blend: garment velocity denoiser ckpt (final)
    if int(os.environ.get("USE_MASK_TRAJ_BLEND", "0")) and getattr(state, "_GARMENT_VEL_DENOISER", None) is not None:
        torch.save({k: v.cpu() for k, v in state._GARMENT_VEL_DENOISER.state_dict().items()},
                   os.path.join(final_path, "garment_vel_denoiser.pt"))
        log.info("Saved garment_vel_denoiser.pt")
        # masked-trajectory-blend: snapshot the garment velocity denoiser
        if int(os.environ.get("USE_MASK_TRAJ_BLEND", "0")) and getattr(state, "_GARMENT_VEL_DENOISER", None) is not None:
            torch.save({k: v.cpu() for k, v in state._GARMENT_VEL_DENOISER.state_dict().items()},
                       os.path.join(ckpt_dir, "garment_vel_denoiser.pt"))

# ==== [6] trainlib/rollouts/halo_eval.py : TWO-TRACK state-quarantine deploy rollout (MASK_TRAJ_TWOTRACK) ====
        # ── TWO-TRACK STATE QUARANTINE (MASK_TRAJ_TWOTRACK): maintain x_base (pure run37) and
        #    x_edit (garment track). HARD-composite x_edit = x_base*(1-M) + x_edit*M after EVERY
        #    scheduler step so garment changes can NEVER accumulate outside the warped mask. Fixes
        #    the single-track leak (masked velocity but shared x -> bg drift over the rollout). ──
        import trainlib.state as _st
        _twotrack = (int(os.environ.get("MASK_TRAJ_TWOTRACK", "0"))
                     and getattr(_st, "_GARMENT_VEL_DENOISER", None) is not None)
        if _twotrack:
            _wm = batch["warped_mask"].to(device, person.dtype)
            if _wm.dim() == 3: _wm = _wm.unsqueeze(1)
            _Mc = (_wm > 0.5).to(person.dtype)                       # (B,1,H,W) warped garment region
            _xb = _x.clone(); _xe = _x.clone()                       # same initial noise
            for _i in range(_ns):
                _sc = torch.full((B,), float(_sigs[_i]), device=device, dtype=sigma.dtype)
                _ds = (float(_sigs[_i]) - float(_sigs[_i + 1]))
                _st._MASK_TRAJ_BYPASS = True
                _vb = unpack_latents(fwd(_xb, _sc), B, C, H, W)      # pure run37 (whole image)
                _st._MASK_TRAJ_BYPASS = False
                _ve = unpack_latents(fwd(_xe, _sc), B, C, H, W)      # blended garment velocity
                _xb = _xb - _ds * _vb
                _xe = _xe - _ds * _ve
                _xe = _xb * (1.0 - _Mc) + _xe * _Mc                  # HARD composite: bg/skin reset to x_base
            _st._MASK_TRAJ_BYPASS = False
            _x = _xe
        else:
            for _i in range(_ns):
                _sc = torch.full((B,), float(_sigs[_i]), device=device, dtype=sigma.dtype)
                _x = _x - (float(_sigs[_i]) - float(_sigs[_i + 1])) * unpack_latents(fwd(_x, _sc), B, C, H, W)
        # (the SAVE_DEPLOY_IMG block also adds, per id:)
        # _PImg.fromarray(_pp).save(f"{_od}/{_iid}_rawpred.png")   # standalone RAW pred for metric_rawpred

# ==== [7] trainlib/losses/total.py : PURE_LATENT None-coalesce (flow-only single-step training skips the VAE decode) ====
# guarded by PURE_LATENT; ~50-line coalesce block that sets missing image-loss terms/weights to 0 so
# the sum+metrics never see None when the per-step VAE decode is skipped. (elided; see trainlib/losses/total.py)
