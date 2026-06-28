"""main() — training loop + checkpointing/val."""
import json, logging, os, shutil, sys, time, traceback
from dataclasses import asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenDoubleStreamAttnProcessor2_0
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers import QwenImageEditPlusPipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
try:
    from multi_block_garment import (MultiBlockGarmentInjection, install_multi_block_hooks,
                                      build_spatial_mask_from_warped, per_block_norms_table)
except Exception:
    pass
from trainlib import state

from trainlib.config import Args
from trainlib.models import (GarmentCrossAttn, GarmentLatentEnhancer, GarmentNet, GarmentNetAdaLN, GarmentNetCrossAttn, GarmentNetOutput, GarmentRepairGate, GarmentSlotEncoder, OOTDInjector, PatchDiscriminator, QwenAuxSlotEncoder, QwenControlNet, QwenGarmentEncoder, QwenGarmentNetAdaLN, QwenGarmentNetBlockCopyAdaLN, QwenGarmentNetCrossAttn, QwenLatentRefiner, QwenSlotEnricher, V6Heads, _make_qwen_block, perceptual_loss)
from trainlib.data import (pack_latents, unpack_latents, vae_decode_to_pil, precompute_rough_pils, precompute_prompt_embeds, load_pose_latents, VTONDataset, collate_fn)
from trainlib.builders import (get_vgg_features, _get_v6_heads, _v6_hidden_hook, _get_invalid_token, CrossAttnGarmentProcessor, QwenGarmentBranch, OOTDQwenAttnProcessor, _get_garment_branch, _get_qwen_slot_enricher, _get_qwen_aux_slot, _get_qwen_refiner, _get_garment_encoder, _get_garment_xattn, _get_controlnet, _make_controlnet_block_hook, _proj_out_pre_hook, _get_garment_net, _garment_inject_hook, _get_discriminator, _get_critic, HoleyAttnProcessor, AgnosticCtrlProcessor, build_repair_attn_mask, install_garment_gates)
from trainlib.forward import train_step
from trainlib.constants import *


def main():
    args = Args()
    args.sigma_beta_alpha = float(os.environ.get("SIGMA_BETA_ALPHA", str(args.sigma_beta_alpha)))
    args.sigma_beta_beta = float(os.environ.get("SIGMA_BETA_BETA", str(args.sigma_beta_beta)))
    device = torch.device(args.device_transformer); dtype = torch.bfloat16
    vae_device = torch.device("cuda:1")

    run_prefix = "garment_net_vton" if int(os.environ.get("USE_GARMENT_NET", "0")) else "vton"
    run_name = os.environ.get("RUN_NAME", f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = asdict(args)
    cfg["slot_order"] = SLOT_ORDER
    cfg["slot_order_idx"] = SLOT_ORDER_IDX
    # Region weights (read from env or default) — picked up by generate_panel.py
    cfg["w_out"]  = float(os.environ.get("W_OUTSIDE",  "0.05"))
    cfg["w_core"] = float(os.environ.get("W_CORE",     "1.0"))
    cfg["w_rep"]  = float(os.environ.get("W_REPAIR",   "0.25"))
    cfg["w_bdy"]  = float(os.environ.get("W_BOUNDARY", "1.0"))
    cfg["lambda_repair"] = float(os.environ.get("LAMBDA_REPAIR", "0.25"))
    cfg["cfg_dropout"] = float(os.environ.get("CFG_DROPOUT", "0.0"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = os.path.join(args.output_dir, "train.log")
    log = logging.getLogger("vton"); log.setLevel(logging.INFO); log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path)); log.addHandler(logging.StreamHandler())
    torch.manual_seed(args.seed)
    # Strict determinism (for reproducibility across runs)
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if int(os.environ.get("DETERMINISTIC", "0")):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    # ── Stage 1: load QwenImageEditPlusPipeline on GPU 1, keep VAE + text_encoder, drop transformer ──
    log.info("Loading QwenImageEditPlusPipeline on cuda:1 (text encoder + VAE only)...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=dtype, transformer=None,
    )
    pipe = pipe.to(vae_device)

    if int(os.environ.get("USE_FULL_TRAIN", "0")):
        # Full-train: skip per-sample VL encode (use precomputed _prompt_embeds.pt files)
        # and skip pose_cache (slot_order has no "pose" in baseline).
        log.info(f"[USE_FULL_TRAIN] Lazy prompt cache from {args.text_cache_dir} ({len(TRAIN_IDS)} samples)")
        class _LazyPromptCache:
            """Per-sample lazy loader. Disk format: pe (seq, dim), pm (seq,).
            Train code expects pe (1, seq, dim) and pm (1, seq) — add batch dim."""
            def __init__(self, text_dir, device, dtype):
                self.text_dir = text_dir
                self.device = device
                self.dtype = dtype
            def __getitem__(self, sid):
                pe = torch.load(os.path.join(self.text_dir, f"{sid}_prompt_embeds.pt"),
                                weights_only=True).to(self.device, dtype=self.dtype)
                pm = torch.load(os.path.join(self.text_dir, f"{sid}_prompt_mask.pt"),
                                weights_only=True).to(self.device, dtype=torch.long)
                if pe.dim() == 2: pe = pe.unsqueeze(0)
                if pm.dim() == 1: pm = pm.unsqueeze(0)
                return (pe, pm)
        prompt_cache = _LazyPromptCache(args.text_cache_dir, device, dtype)
        del pipe
        torch.cuda.empty_cache()
        # Pose: keep empty dict; SLOT_ORDER must not include "pose" for full-train mode.
        pose_cache = {}
        if "pose" in SLOT_ORDER:
            raise SystemExit("USE_FULL_TRAIN=1 requires SLOT_ORDER without 'pose' (no per-sample pose cache for 11k samples).")
    else:
        # VAE-decode degraded_rough_latent → PIL per sample (for Qwen2.5-VL semantic pass)
        log.info("VAE-decoding rough latents to PIL...")
        rough_pils = precompute_rough_pils(args.latent_cache_dir, pipe.vae, vae_device, dtype)

        # Per-sample Qwen2.5-VL encode: [agnostic_pil, pose_pil, rough_pil, garment_pil] + fixed prompt
        log.info("Encoding per-sample prompts via Qwen2.5-VL...")
        prompt_cache = precompute_prompt_embeds(pipe, args.latent_cache_dir, LOCAL_CACHE, rough_pils, vae_device, dtype)
        for sid, (pe, pm) in prompt_cache.items():
            log.info(f"  prompt[{sid}]: pe={tuple(pe.shape)} pm={tuple(pm.shape)} {pe.dtype}")
        # move prompt cache to GPU 0 for training
        prompt_cache = {k: (v[0].to(device, dtype=dtype), v[1].to(device, dtype=torch.long))
                        for k, v in prompt_cache.items()}

        # Free the text encoder and pipeline VAE
        del pipe
        torch.cuda.empty_cache()

        # Load pose latents (precomputed VAE-encoded densepose RGB, one per sample)
        log.info("Loading pose latents from local cache...")
        pose_cache = load_pose_latents(LOCAL_CACHE, device, dtype)
        for sid, lat in pose_cache.items():
            log.info(f"  pose[{sid}]: {tuple(lat.shape)} {lat.dtype}")

    # Load VAE on cuda:1 for image-space L1 loss
    log.info("Loading VAE on cuda:1 for image-space L1 loss...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=dtype)
    vae.to(vae_device).eval()
    vae.requires_grad_(False)

    m_v = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    s_v = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(vae_device, dtype)
    if int(os.environ.get("USE_FULL_TRAIN", "0")):
        # Lazy: decode person_latent on demand per __getitem__
        log.info(f"[USE_FULL_TRAIN] Lazy person_image cache (VAE decode on-demand) for {len(TRAIN_IDS)} samples")
        class _LazyPersonImageCache:
            def __init__(self, latent_dir, vae, vae_device, dtype, m_v, s_v):
                self.latent_dir = latent_dir; self.vae = vae; self.dtype = dtype
                self.vae_device = vae_device; self.m_v = m_v; self.s_v = s_v
            @torch.no_grad()
            def __getitem__(self, sid):
                plat = torch.load(os.path.join(self.latent_dir, f"{sid}_person_latent.pt"),
                                  weights_only=True).unsqueeze(0).unsqueeze(2).to(self.vae_device, self.dtype)
                denorm = plat * self.s_v + self.m_v
                decoded = self.vae.decode(denorm, return_dict=False)[0][:, :, 0]
                return decoded.clamp(-1, 1)[0].to(self.vae_device, dtype=self.dtype).detach()
        person_image_cache = _LazyPersonImageCache(args.latent_cache_dir, vae, vae_device, dtype, m_v, s_v)
    else:
        # Precompute decoded person image per sample (cached at startup, no per-step recompute)
        log.info("Precomputing decoded person images for image-space loss...")
        person_image_cache = {}
        for sid in TRAIN_IDS:
            plat = torch.load(os.path.join(args.latent_cache_dir, f"{sid}_person_latent.pt"),
                              weights_only=True).unsqueeze(0).unsqueeze(2).to(vae_device, dtype)  # (1,16,1,128,96)
            denorm = plat * s_v + m_v
            with torch.no_grad():
                decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0]                      # (1,3,Hi,Wi)
            person_image_cache[sid] = decoded.clamp(-1, 1)[0].to(vae_device, dtype=dtype).detach()
        for sid, img in person_image_cache.items():
            log.info(f"  person_image[{sid}]: {tuple(img.shape)} {img.dtype}")

    # ── Stage 2: load transformer + tryon LoRA on GPU 0 ──
    log.info("Loading transformer on cuda:0...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
    transformer = get_peft_model(
        transformer,
        LoraConfig(r=args.rank, lora_alpha=args.alpha,
                   init_lora_weights=args.init_lora_weights,
                   target_modules=args.lora_targets, lora_dropout=0.0),
        adapter_name="tryon")
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(device)

    # Install HoleyAttnProcessor when attention masking is enabled
    if int(os.environ.get("USE_REPAIR_ATTN_MASK", "0")):
        n_proc = 0
        for mod in transformer.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = HoleyAttnProcessor()
                n_proc += 1
        log.info(f"HoleyAttnProcessor installed on {n_proc} attention blocks")

    # v19/v20/v21: install AgnosticCtrlProcessor for trust-map agnostic suppression
    if int(os.environ.get("USE_AGN_CTRL", "0")):
        n_proc = 0
        for mod in transformer.modules():
            if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                mod.processor = AgnosticCtrlProcessor()
                n_proc += 1
        log.info(f"AgnosticCtrlProcessor installed on {n_proc} attention blocks "
                 f"(key_bias={os.environ.get('AGN_KEY_BIAS','0')}, "
                 f"v_scale={os.environ.get('AGN_V_SCALE','0')})")

    # exp420: install GarmentRepairGate on transformer blocks
    garment_gates, gate_hooks = [], []
    if int(os.environ.get("USE_GARMENT_GATE", "0")):
        # Figure out garment slot index (position after C in SLOT_ORDER)
        gar_idx = SLOT_ORDER.index("garment") if "garment" in SLOT_ORDER else -1
        if gar_idx >= 0:
            n_c = 3072  # tokens per slot at 128×96 with 2×2 packing
            every_n = int(os.environ.get("GATE_EVERY_N", "3"))
            garment_gates, gate_hooks = install_garment_gates(
                transformer, n_c, gar_idx, device, dtype,
                n_heads=4, head_dim=32, every_n=every_n)
            log.info(f"GarmentRepairGate installed on {len(garment_gates)} blocks "
                     f"(every {every_n}, gar_idx={gar_idx})")

    # Trainable params: tryon LoRA + garment gates
    tryon_params = [p for _, p in transformer.named_parameters() if p.requires_grad]
    gate_params = [p for g in garment_gates for p in g.parameters()]
    log.info(f"tryon_params: {sum(p.numel() for p in tryon_params):,}")
    log.info(f"gate_params:  {sum(p.numel() for p in gate_params):,}")
    state.GARMENT_GATES = garment_gates

    # Load pretrained LoRA weights if specified (for continued training or frozen gate-only)
    lora_path = os.environ.get("LORA_INIT_PATH", "")
    if lora_path and os.path.exists(lora_path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as sf_load
        set_peft_model_state_dict(transformer, sf_load(lora_path), adapter_name="tryon")
        log.info(f"Loaded pretrained LoRA from {lora_path}")

    if int(os.environ.get("FREEZE_LORA", "0")):
        for p in tryon_params:
            p.requires_grad_(False)
        tryon_params = []
        log.info("LoRA FROZEN")

    # User-directed v5: unfreeze the "garment output pathway" — norm_out + proj_out —
    # so the frozen base can adapt to the garment branch's injected features. Skin/bg/
    # route v6 heads remain frozen (they read pre-proj features, separate path).
    proj_out_params = []
    if int(os.environ.get("UNFREEZE_PROJ_OUT", "0")):
        # Get inner module (peft-wrapped or raw)
        _xfmr_inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
        for p in _xfmr_inner.norm_out.parameters():
            p.requires_grad_(True); proj_out_params.append(p)
        for p in _xfmr_inner.proj_out.parameters():
            p.requires_grad_(True); proj_out_params.append(p)
        log.info(f"UNFREEZE_PROJ_OUT: norm_out + proj_out unfrozen "
                 f"({sum(p.numel() for p in proj_out_params):,} params)")

    param_groups = []
    # Optional per-block LR split — higher LR for early transformer blocks so they
    # develop useful adaptations instead of being drowned by late-block gradients.
    # EARLY_BLOCK_CUTOFF: boundary block idx (inclusive). LR_EARLY_MULT: multiplier
    # applied to args.lr for blocks [0..cutoff]. Blocks > cutoff use args.lr.
    # Default (cutoff=-1): single group, all params at args.lr (exp524 behavior).
    early_cutoff = int(os.environ.get("EARLY_BLOCK_CUTOFF", "-1"))
    early_mult   = float(os.environ.get("LR_EARLY_MULT", "1.0"))
    if tryon_params and early_cutoff >= 0 and early_mult != 1.0:
        early_params, late_params = [], []
        for n, p in transformer.named_parameters():
            if not p.requires_grad: continue
            parts = n.split(".")
            try:
                idx = int(parts[parts.index("transformer_blocks") + 1])
            except (ValueError, IndexError):
                late_params.append(p); continue
            if idx <= early_cutoff:
                early_params.append(p)
            else:
                late_params.append(p)
        if early_params:
            param_groups.append({"params": early_params, "lr": args.lr * early_mult})
        if late_params:
            param_groups.append({"params": late_params, "lr": args.lr})
        log.info(f"per-block LR: blocks 0..{early_cutoff} at lr={args.lr*early_mult:.2e}, "
                 f"blocks {early_cutoff+1}..59 at lr={args.lr:.2e}; "
                 f"{sum(p.numel() for p in early_params):,} early params, "
                 f"{sum(p.numel() for p in late_params):,} late params")
    elif tryon_params:
        param_groups.append({"params": tryon_params, "lr": args.lr})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": args.lr * 3})
    if proj_out_params:
        proj_out_lr = float(os.environ.get("PROJ_OUT_LR", "3e-5"))
        param_groups.append({"params": proj_out_params, "lr": proj_out_lr})
        log.info(f"proj_out + norm_out @ lr={proj_out_lr:.1e}")
    # v6 specialized heads (Linear on 3072-dim Qwen features). Hooks norm_out
    # to capture pre-proj features. Heads have separate gradient paths from
    # the main transformer, so cross-region bleed is bounded by feature
    # sharing only — no direct mask-weight averaging on a single output.
    if int(os.environ.get("USE_V6", "0")):
        v6_heads = _get_v6_heads(device, dtype, hidden_dim=transformer.inner_dim)
        # Optionally load pretrained v6 heads (e.g., from a frozen base run)
        v6_init_path = os.environ.get("V6_HEADS_INIT_PATH", "")
        if v6_init_path and os.path.exists(v6_init_path):
            v6_heads.load_state_dict(torch.load(v6_init_path, weights_only=True))
            log.info(f"Loaded pretrained v6_heads from {v6_init_path}")
        if int(os.environ.get("FREEZE_V6", "0")):
            for p in v6_heads.parameters():
                p.requires_grad_(False)
            log.info("v6_heads FROZEN")
        else:
            param_groups.append({"params": list(v6_heads.parameters()), "lr": args.lr * 10})
            log.info(f"v6_heads: {sum(p.numel() for p in v6_heads.parameters()):,} params "
                     f"at lr={args.lr*10:.2e} (to_s + to_b + to_route)")
        # Register hook on transformer.norm_out to capture (B, N_total, 3072) features
        transformer.norm_out.register_forward_hook(_v6_hidden_hook)
        log.info("v6: registered forward_hook on transformer.norm_out")
    # Region-map adapter + gate (USE_REGION_MAP=1) — register here so first
    # training step finds them already attached to the optimizer.
    if int(os.environ.get("USE_REGION_MAP", "0")):
        import torch.nn as _rm_nn_init
        if state._REGION_ADAPTER is None:
            _rm_arch = os.environ.get("REGION_ADAPTER_ARCH", "linear")
            if _rm_arch == "mlp":
                state._REGION_ADAPTER = _rm_nn_init.Sequential(
                    _rm_nn_init.Conv2d(4, 32, kernel_size=3, padding=1, bias=True),
                    _rm_nn_init.SiLU(),
                    _rm_nn_init.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
                ).to(device, dtype=dtype)
                # Zero-init FINAL conv only (gate*output = 0 at start regardless of
                # what the first conv computes).
                _rm_nn_init.init.zeros_(state._REGION_ADAPTER[2].weight)
                _rm_nn_init.init.zeros_(state._REGION_ADAPTER[2].bias)
            else:  # "linear" — Conv2d(4, 16, 1) zero-init both weight and bias
                state._REGION_ADAPTER = _rm_nn_init.Conv2d(4, 16, kernel_size=1, bias=True).to(device, dtype=dtype)
                _rm_nn_init.init.zeros_(state._REGION_ADAPTER.weight); _rm_nn_init.init.zeros_(state._REGION_ADAPTER.bias)
            state._REGION_ADAPTER.requires_grad_(True)
        if state._REGION_GATE is None:
            _gate_init = float(os.environ.get("REGION_GATE_INIT", "0.0"))
            state._REGION_GATE = _rm_nn_init.Parameter(torch.tensor([_gate_init], device=device, dtype=torch.float32))
            state._REGION_GATE.requires_grad_(True)
        # 5_25 instructions8: optional resume + freeze for region_map
        _rm_init_path = os.environ.get("REGION_MAP_INIT_PATH", "")
        if _rm_init_path and os.path.exists(_rm_init_path):
            _rm_sd = torch.load(_rm_init_path, weights_only=True, map_location="cpu")
            state._REGION_ADAPTER.load_state_dict({k: v.to(device=device, dtype=dtype) for k,v in _rm_sd["adapter"].items()})
            state._REGION_GATE.data = _rm_sd["gate"].to(device=device, dtype=torch.float32)
            log.info(f"region_map: loaded from {_rm_init_path} (gate sigmoid={torch.sigmoid(state._REGION_GATE).item():.4f})")
        rm_lr = float(os.environ.get("REGION_MAP_LR", "1e-3"))
        if int(os.environ.get("FREEZE_REGION_MAP", "0")):
            for p in state._REGION_ADAPTER.parameters(): p.requires_grad_(False)
            state._REGION_GATE.requires_grad_(False)
            log.info("region_map FROZEN")
        else:
            param_groups.append({"params": list(state._REGION_ADAPTER.parameters()) + [state._REGION_GATE], "lr": rm_lr})
            log.info(f"region_map: arch={os.environ.get('REGION_ADAPTER_ARCH','linear')} "
                     f"adapter_params={sum(p.numel() for p in state._REGION_ADAPTER.parameters()):,} "
                     f"gate_init={float(os.environ.get('REGION_GATE_INIT','0.0')):.2f} (sigmoid={torch.sigmoid(state._REGION_GATE).item():.4f}) "
                     f"lr={rm_lr:.2e}")

    # v22: register the learned [INVALID_AGNOSTIC] token with the optimizer.
    if int(os.environ.get("USE_INVALID_TOKEN", "0")):
        _e_inv0 = _get_invalid_token(device, dtype, 64)
        # 5_25 instructions8: optional resume + freeze
        _iv_init_path = os.environ.get("INVALID_TOKEN_INIT_PATH", "")
        if _iv_init_path and os.path.exists(_iv_init_path):
            _iv_sd = torch.load(_iv_init_path, weights_only=True, map_location="cpu")
            _e_inv0.data = _iv_sd.to(device=device, dtype=dtype)
            log.info(f"invalid_token: loaded from {_iv_init_path}")
        _iv_lr = float(os.environ.get("INVALID_TOKEN_LR", "1e-3"))
        if int(os.environ.get("FREEZE_INVALID_TOKEN", "0")):
            _e_inv0.requires_grad_(False)
            log.info("invalid_token FROZEN")
        else:
            param_groups.append({"params": [_e_inv0], "lr": _iv_lr})
            log.info(f"invalid_token: 1x1x{_e_inv0.shape[-1]} learned embedding, lr={_iv_lr:.2e}")
    if int(os.environ.get("USE_GARMENT_NET", "0")):
        gn_mode = os.environ.get("GARMENT_NET_MODE", "norm_residual")
        # Output-space garment net runs on vae_device since it produces image-resolution outputs
        gn_device = vae_device if gn_mode == "output_space" else device
        garment_net = _get_garment_net(gn_device, dtype, hidden_dim=transformer.inner_dim)
        # Block-copy variant: copy state_dict from selected main-model blocks into the
        # garment_net's blocks. GARMENT_NET_COPY_BLOCKS=CSV (default: 0..N-1).
        # Main `transformer` is peft-wrapped (LoRA), so its block state_dict has
        # `base_layer.*` + lora keys — load a fresh non-LoRA copy purely for the
        # weight extraction, then dispose.
        if isinstance(garment_net, QwenGarmentNetBlockCopyAdaLN):
            n_blk = len(garment_net.blocks)
            csv = os.environ.get("GARMENT_NET_COPY_BLOCKS", "")
            if csv:
                indices = [int(s) for s in csv.split(",") if s.strip()]
            else:
                indices = list(range(n_blk))
            assert len(indices) == n_blk, \
                f"GARMENT_NET_COPY_BLOCKS must have {n_blk} indices, got {indices}"
            log.info(f"garment_net: loading fresh transformer for block-copy from indices {indices}...")
            _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
                args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
            garment_net.load_blocks_from_main(_src_xfmr, indices)
            del _src_xfmr
            torch.cuda.empty_cache()
            log.info(f"garment_net: copied main-model block weights from indices {indices}")
        # Optional warm-start from previous garment_net checkpoint.
        # Load to CPU first then move into model to avoid holding duplicate
        # state_dict on GPU.
        gn_init_path = os.environ.get("GARMENT_NET_INIT_PATH", "")
        if gn_init_path and os.path.exists(gn_init_path):
            from safetensors.torch import load_file as _sf_load_gn
            _gn_sd = (_sf_load_gn(gn_init_path, device="cpu") if gn_init_path.endswith(".safetensors")
                       else torch.load(gn_init_path, map_location="cpu", weights_only=True))
            garment_net.load_state_dict(_gn_sd, strict=False)
            del _gn_sd
            torch.cuda.empty_cache()
            log.info(f"garment_net: warm-started from {gn_init_path}")
        gn_lr = float(os.environ.get("GARMENT_NET_LR", "1e-4"))
        param_groups.append({"params": list(garment_net.parameters()), "lr": gn_lr})
        log.info(f"garment_net ({gn_mode}): {sum(p.numel() for p in garment_net.parameters()):,} params at lr={gn_lr:.2e}")
        if gn_mode in ("norm_residual", "adaln"):
            transformer.norm_out.register_forward_hook(_garment_inject_hook)
            log.info(f"garment_net: registered forward_hook on transformer.norm_out ({gn_mode})")
        elif gn_mode == "cross_attn":
            n_proc = 0
            for mod in transformer.modules():
                if hasattr(mod, "processor") and isinstance(mod.processor, QwenDoubleStreamAttnProcessor2_0):
                    mod.processor = CrossAttnGarmentProcessor()
                    n_proc += 1
            log.info(f"garment_net: installed CrossAttnGarmentProcessor on {n_proc} attention blocks")
        else:
            log.info("garment_net: output_space mode (no transformer hook; correction added to decoded pred_img)")

    # User-directed cross-attn-at-proj_out path. Independent of USE_GARMENT_NET.
    # F = transformer hidden after norm_out. G = QwenGarmentEncoder(garment_latent).
    # Inject A_g = CrossAttn(F, G) gated by per-token garment mask, BEFORE proj_out.
    # v6 heads still see un-enhanced F via the existing norm_out hook.
    if int(os.environ.get("USE_GARMENT_XATTN", "0")):
        n_xa_layers = int(os.environ.get("GARMENT_XATTN_LAYERS", "2"))
        gx_enc = _get_garment_encoder(device, dtype, n_layers=n_xa_layers)
        # Block-copy weights from main transformer (use fresh copy to avoid PEFT LoRA keys)
        csv = os.environ.get("GARMENT_XATTN_COPY_BLOCKS", "")
        if csv:
            xa_indices = [int(s) for s in csv.split(",") if s.strip()]
        else:
            xa_indices = list(range(n_xa_layers))
        assert len(xa_indices) == n_xa_layers, \
            f"GARMENT_XATTN_COPY_BLOCKS must have {n_xa_layers} indices, got {xa_indices}"
        log.info(f"garment_xattn: loading fresh transformer for block-copy from indices {xa_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        gx_enc.load_blocks_from_main(_src_xfmr, xa_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"garment_xattn: copied main-model block weights from indices {xa_indices}")
        gx_xattn = _get_garment_xattn(device, dtype)
        # Optional warm-start from previous garment_xattn checkpoint.
        # Load to CPU first to avoid GPU duplicate.
        gx_init_path = os.environ.get("GARMENT_XATTN_INIT_PATH", "")
        if gx_init_path and os.path.exists(gx_init_path):
            from safetensors.torch import load_file as _sf_load_gx
            _gx_sd = (_sf_load_gx(gx_init_path, device="cpu") if gx_init_path.endswith(".safetensors")
                       else torch.load(gx_init_path, map_location="cpu", weights_only=True))
            gx_xattn.load_state_dict(_gx_sd, strict=False)
            del _gx_sd
            torch.cuda.empty_cache()
            log.info(f"garment_xattn: warm-started from {gx_init_path}")
        # Also optionally load encoder weights — same CPU-staged loading.
        gxe_init_path = os.environ.get("GARMENT_ENCODER_INIT_PATH", "")
        if gxe_init_path and os.path.exists(gxe_init_path):
            from safetensors.torch import load_file as _sf_load_gxe
            _gxe_sd = (_sf_load_gxe(gxe_init_path, device="cpu") if gxe_init_path.endswith(".safetensors")
                       else torch.load(gxe_init_path, map_location="cpu", weights_only=True))
            gx_enc.load_state_dict(_gxe_sd, strict=False)
            del _gxe_sd
            torch.cuda.empty_cache()
            log.info(f"garment_encoder: warm-started from {gxe_init_path}")
        gx_lr = float(os.environ.get("GARMENT_XATTN_LR", "1e-4"))
        gx_lr_gate = float(os.environ.get("GARMENT_XATTN_LR_GATE", "1e-3"))
        # Split gate_logit into its own LR group (higher LR for the single scalar)
        gx_gate_params, gx_kv_params = [], []
        for n, p in gx_xattn.named_parameters():
            (gx_gate_params if "gate_logit" in n else gx_kv_params).append(p)
        param_groups.append({"params": list(gx_enc.parameters()) + gx_kv_params, "lr": gx_lr})
        if gx_gate_params:
            param_groups.append({"params": gx_gate_params, "lr": gx_lr_gate})
        log.info(f"garment_xattn: encoder {sum(p.numel() for p in gx_enc.parameters()):,} + "
                 f"xattn-kv {sum(p.numel() for p in gx_kv_params):,} @ lr={gx_lr:.2e}; "
                 f"gate {sum(p.numel() for p in gx_gate_params):,} @ lr={gx_lr_gate:.2e} (fp32)")
        transformer.proj_out.register_forward_pre_hook(_proj_out_pre_hook)
        log.info("garment_xattn: registered forward_pre_hook on transformer.proj_out")

    # 5_30: Multi-block garment injection (instructions13).
    # Per-block GarmentCrossAttn + AdaLN-β at selected transformer blocks.
    # NO scalar gates. Spatial mask only. Reuses garment_encoder G.
    if int(os.environ.get("USE_MULTI_BLOCK_INJ", "0")):
        _mb_targets_csv = os.environ.get(
            "MULTI_BLOCK_TARGETS",
            "4,8,12,16,20,24,28,32,36,40,44,48,52,56,59")
        _mb_targets = [int(s) for s in _mb_targets_csv.split(",") if s.strip()]
        log.info(f"multi_block_inj: targets = {_mb_targets} ({len(_mb_targets)} blocks)")
        # Get inner transformer (peft-wrapped main has .base_model.model)
        _inner_t_mb = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
        # 5_31: USE_MULTI_BLOCK_FULL=1 → full Qwen blocks at injection sites
        # (replaces GarmentCrossAttn + AdaLN-β with QwenImageTransformerBlock copies).
        _use_full = int(os.environ.get("USE_MULTI_BLOCK_FULL", "0"))
        if _use_full:
            from multi_block_garment_full import (GarmentChain, install_garment_chain_hooks)
            state._MULTI_GAR_INJECTION = GarmentChain(_inner_t_mb, _mb_targets,
                                                dim=transformer.inner_dim).to(device, dtype=dtype)
            log.info(f"multi_block_inj: USING FULL Qwen blocks (GarmentChain) at {len(_mb_targets)} sites "
                     f"({sum(p.numel() for p in state._MULTI_GAR_INJECTION.parameters())/1e9:.2f}B params)")
            # Warm-start the GarmentChain from a prior best_val/final (resume training).
            _mbf_init = os.environ.get("MULTI_GAR_INIT_PATH", "")
            if _mbf_init and os.path.exists(_mbf_init):
                _mbf_sd = torch.load(_mbf_init, map_location="cpu", weights_only=True)
                _miss, _unexp = state._MULTI_GAR_INJECTION.load_state_dict(_mbf_sd, strict=False)
                log.info(f"GarmentChain: warm-started from {_mbf_init} "
                         f"(missing={len(_miss)}, unexpected={len(_unexp)})")
            _freeze_mb = int(os.environ.get("FREEZE_MULTI_BLOCK", "0"))
            for p in state._MULTI_GAR_INJECTION.parameters(): p.requires_grad_(not _freeze_mb)
            state._MULTI_GAR_HOOKS = install_garment_chain_hooks(
                _inner_t_mb, state._MULTI_GAR_INJECTION,
                state._MULTI_GAR_HOLDER, gar_holder_key=None)
            if _freeze_mb:
                log.info("multi_block_inj (FULL chain): FROZEN (used in forward, not trained)")
            else:
                _mb_lr = float(os.environ.get("MULTI_BLOCK_LR", "2e-5"))
                param_groups.append({"params": list(state._MULTI_GAR_INJECTION.parameters()), "lr": _mb_lr})
                log.info(f"multi_block_inj (FULL chain): @ lr={_mb_lr:.2e}")
        else:
            _enc_dim = int(os.environ.get("MULTI_BLOCK_ENC_DIM", "3072"))
            state._MULTI_GAR_INJECTION = MultiBlockGarmentInjection(
                n_blocks=len(_mb_targets), dim=transformer.inner_dim,
                num_heads=24, head_dim=128, enc_dim=_enc_dim).to(device, dtype=dtype)
            # 5_31 OOTD-style init: copy Qwen attention weights into multi-block xattn
            # before any explicit warm-start load. Only applies if no MULTI_GAR_INIT_PATH set
            # OR if USE_OOTD_INIT=1 is forced.
            _ootd_init = int(os.environ.get("USE_OOTD_INIT", "0"))
            _mb_init = os.environ.get("MULTI_GAR_INIT_PATH", "")
            if _ootd_init or not (_mb_init and os.path.exists(_mb_init)):
                try:
                    state._MULTI_GAR_INJECTION.init_from_qwen_blocks(_inner_t_mb, _mb_targets)
                    log.info("multi_block_inj: OOTD-style init from Qwen attention weights")
                except Exception as _ootd_e:
                    log.warning(f"multi_block_inj: OOTD init failed ({_ootd_e}), falling back to fresh init")
            # Warm start from prior ckpt (overrides OOTD init if path exists)
            if _mb_init and os.path.exists(_mb_init):
                _mb_sd = torch.load(_mb_init, map_location="cpu", weights_only=True)
                state._MULTI_GAR_INJECTION.load_state_dict(_mb_sd, strict=False)
                log.info(f"multi_block_inj: warm-started from {_mb_init}")
            for p in state._MULTI_GAR_INJECTION.parameters(): p.requires_grad_(True)
            state._MULTI_GAR_HOOKS = install_multi_block_hooks(
                _inner_t_mb, state._MULTI_GAR_INJECTION, _mb_targets,
                state._MULTI_GAR_HOLDER, gar_holder_key=None)
            _mb_lr = float(os.environ.get("MULTI_BLOCK_LR", "2e-4"))
            param_groups.append({"params": list(state._MULTI_GAR_INJECTION.parameters()), "lr": _mb_lr})
            log.info(f"multi_block_inj: {sum(p.numel() for p in state._MULTI_GAR_INJECTION.parameters()):,} params @ lr={_mb_lr:.2e}")

    # ControlNet branch (user-directed 2026-05-02). Agnostic conditioning.
    # ~12 block-copies of main 0..11. Per-block zero-init residual added to
    # main's hidden_states C-slot via forward_hooks on main.transformer_blocks[i].
    if int(os.environ.get("USE_CONTROLNET", "0")):
        n_cn_layers = int(os.environ.get("CONTROLNET_LAYERS", "12"))
        cnet = _get_controlnet(device, dtype, n_layers=n_cn_layers)
        # Block-copy weights from main
        cn_copy_csv = os.environ.get("CONTROLNET_COPY_BLOCKS", "")
        if cn_copy_csv:
            cn_copy_indices = [int(s) for s in cn_copy_csv.split(",") if s.strip()]
        else:
            cn_copy_indices = list(range(n_cn_layers))
        assert len(cn_copy_indices) == n_cn_layers, \
            f"CONTROLNET_COPY_BLOCKS must have {n_cn_layers} indices, got {cn_copy_indices}"
        log.info(f"controlnet: loading fresh transformer for block-copy from indices {cn_copy_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        cnet.load_blocks_from_main(_src_xfmr, cn_copy_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"controlnet: copied main-model block weights from indices {cn_copy_indices}")
        cn_lr = float(os.environ.get("CONTROLNET_LR", "1e-4"))
        param_groups.append({"params": list(cnet.parameters()), "lr": cn_lr})
        log.info(f"controlnet: {sum(p.numel() for p in cnet.parameters()):,} params at lr={cn_lr:.2e}")
        # Install hooks on injection blocks (default 0..N-1, can override)
        cn_inject_csv = os.environ.get("CONTROLNET_INJECT_BLOCKS", "")
        if cn_inject_csv:
            cn_inject_indices = [int(s) for s in cn_inject_csv.split(",") if s.strip()]
        else:
            cn_inject_indices = list(range(n_cn_layers))
        _xfmr_inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
        for blk_i in cn_inject_indices:
            _xfmr_inner.transformer_blocks[blk_i].register_forward_hook(_make_controlnet_block_hook(blk_i))
        log.info(f"controlnet: registered forward_hook on main blocks {cn_inject_indices}")

    # OOTD-style garment branch with K,V injection at LATE main blocks (v2)
    if int(os.environ.get("USE_GARMENT_OOTD", "0")):
        # Branch config: 4 layers default, copy main blocks 0..3
        n_ootd_layers = int(os.environ.get("GARMENT_OOTD_LAYERS", "4"))
        gb = _get_garment_branch(device, dtype, n_layers=n_ootd_layers)
        csv = os.environ.get("GARMENT_OOTD_COPY_BLOCKS", "")
        if csv:
            ootd_indices = [int(s) for s in csv.split(",") if s.strip()]
        else:
            ootd_indices = list(range(n_ootd_layers))
        assert len(ootd_indices) == n_ootd_layers, \
            f"GARMENT_OOTD_COPY_BLOCKS must have {n_ootd_layers} indices, got {ootd_indices}"
        log.info(f"garment_ootd: loading fresh transformer for branch block-copy from indices {ootd_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        gb.load_blocks_from_main(_src_xfmr, ootd_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"garment_ootd: branch — copied main-model block weights from indices {ootd_indices}")

        # Inject at LATE main blocks (default [48, 56])
        inj_csv = os.environ.get("GARMENT_OOTD_INJECT_BLOCKS", "48,56")
        inject_indices = [int(s) for s in inj_csv.split(",") if s.strip()]
        # Build per-injection-block injectors; init from the corresponding main block's
        # effective (base+frozen-LoRA) attn.to_k/to_v.
        state._OOTD_INJECTORS = {}
        gate_init_logit = float(os.environ.get("GARMENT_OOTD_GATE_INIT_LOGIT", "-4.0"))
        for blk_i in inject_indices:
            inj = OOTDInjector(dim=transformer.inner_dim, num_heads=24, head_dim=128,
                               gate_init_logit=gate_init_logit).to(device, dtype=dtype)
            inj.init_from_main_block(transformer.base_model.model.transformer_blocks[blk_i])
            # Force gate_logit BACK to fp32 (Adam updates of ~3e-5 quantize to zero
            # near logit=-4 in bf16, where step size is ~0.04)
            inj.gate_logit.data = inj.gate_logit.data.to(torch.float32)
            inj.requires_grad_(True)
            state._OOTD_INJECTORS[blk_i] = inj
        # Install processors on those late blocks
        n_proc = 0
        for blk_i in inject_indices:
            mblk = transformer.base_model.model.transformer_blocks[blk_i]
            for sub_mod in mblk.modules():
                if hasattr(sub_mod, "processor") and isinstance(sub_mod.processor, QwenDoubleStreamAttnProcessor2_0):
                    sub_mod.processor = OOTDQwenAttnProcessor(block_idx=blk_i)
                    n_proc += 1
        log.info(f"garment_ootd: installed OOTDQwenAttnProcessor on main blocks {inject_indices} ({n_proc} attn blocks)")

        # Four LR groups: branch_blocks, branch_aux, injector_kv (to_k_g/to_v_g),
        # injector_gate (scalar — fp32, dedicated higher LR)
        lr_blocks = float(os.environ.get("GARMENT_OOTD_LR_BLOCKS", "3e-6"))
        lr_aux    = float(os.environ.get("GARMENT_OOTD_LR_AUX",    "3e-5"))
        lr_inj_kv = float(os.environ.get("GARMENT_OOTD_LR_INJ",    "3e-5"))
        lr_gate   = float(os.environ.get("GARMENT_OOTD_LR_GATE",   "1e-3"))
        block_params, aux_params = [], []
        for n, p in gb.named_parameters():
            if n.startswith("blocks."):
                block_params.append(p)
            else:
                aux_params.append(p)
        inj_kv_params, gate_params_list = [], []
        for inj in state._OOTD_INJECTORS.values():
            for n, p in inj.named_parameters():
                if "gate_logit" in n:
                    gate_params_list.append(p)
                else:
                    inj_kv_params.append(p)
        if block_params:    param_groups.append({"params": block_params,    "lr": lr_blocks})
        if aux_params:      param_groups.append({"params": aux_params,      "lr": lr_aux})
        if inj_kv_params:   param_groups.append({"params": inj_kv_params,   "lr": lr_inj_kv})
        if gate_params_list:param_groups.append({"params": gate_params_list,"lr": lr_gate})
        log.info(f"garment_ootd: branch_blocks {sum(p.numel() for p in block_params):,} @ lr={lr_blocks:.1e}; "
                 f"branch_aux {sum(p.numel() for p in aux_params):,} @ lr={lr_aux:.1e}; "
                 f"inj_kv {sum(p.numel() for p in inj_kv_params):,} @ lr={lr_inj_kv:.1e}; "
                 f"gates {sum(p.numel() for p in gate_params_list):,} @ lr={lr_gate:.1e} (fp32, init={gate_init_logit})")

    # v11 Qwen Latent Refiner (post-pred residual)
    if int(os.environ.get("USE_QWEN_REFINER", "0")):
        n_qr_layers = int(os.environ.get("QWEN_REFINER_LAYERS", "4"))
        qr = _get_qwen_refiner(device, dtype)
        csv = os.environ.get("QWEN_REFINER_COPY_BLOCKS", "")
        if csv:
            qr_indices = [int(s) for s in csv.split(",") if s.strip()]
        else:
            qr_indices = list(range(n_qr_layers))
        assert len(qr_indices) == n_qr_layers
        log.info(f"qwen_refiner: loading fresh transformer for block-copy from {qr_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        qr.load_blocks_from_main(_src_xfmr, qr_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"qwen_refiner: copied main blocks {qr_indices}")
        lr_qr_blocks = float(os.environ.get("QWEN_REFINER_LR_BLOCKS", "1e-5"))
        lr_qr_aux    = float(os.environ.get("QWEN_REFINER_LR_AUX",    "3e-5"))
        bp_, ap_ = [], []
        for n, p in qr.named_parameters():
            if n.startswith("blocks."):
                bp_.append(p)
            else:
                ap_.append(p)
        if bp_: param_groups.append({"params": bp_, "lr": lr_qr_blocks})
        if ap_: param_groups.append({"params": ap_, "lr": lr_qr_aux})
        log.info(f"qwen_refiner: blocks {sum(p.numel() for p in bp_):,} @ lr={lr_qr_blocks:.1e}, "
                 f"aux {sum(p.numel() for p in ap_):,} @ lr={lr_qr_aux:.1e}")

    # v9 Qwen Aux Slot (append slot to conditioning)
    if int(os.environ.get("USE_QWEN_AUX_SLOT", "0")):
        n_qas_layers = int(os.environ.get("QWEN_AUX_SLOT_LAYERS", "4"))
        qas = _get_qwen_aux_slot(device, dtype)
        csv = os.environ.get("QWEN_AUX_SLOT_COPY_BLOCKS", "")
        if csv:
            qas_indices = [int(s) for s in csv.split(",") if s.strip()]
        else:
            qas_indices = list(range(n_qas_layers))
        assert len(qas_indices) == n_qas_layers
        log.info(f"qwen_aux_slot: loading fresh transformer for block-copy from {qas_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        qas.load_blocks_from_main(_src_xfmr, qas_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"qwen_aux_slot: copied main blocks {qas_indices}")
        lr_qas_blocks = float(os.environ.get("QWEN_AUX_SLOT_LR_BLOCKS", "1e-5"))
        lr_qas_aux    = float(os.environ.get("QWEN_AUX_SLOT_LR_AUX",    "3e-5"))
        bp_, ap_ = [], []
        for n, p in qas.named_parameters():
            if n.startswith("blocks."):
                bp_.append(p)
            else:
                ap_.append(p)
        if bp_: param_groups.append({"params": bp_, "lr": lr_qas_blocks})
        if ap_: param_groups.append({"params": ap_, "lr": lr_qas_aux})
        log.info(f"qwen_aux_slot: blocks {sum(p.numel() for p in bp_):,} @ lr={lr_qas_blocks:.1e}, "
                 f"aux {sum(p.numel() for p in ap_):,} @ lr={lr_qas_aux:.1e}")

    # v8 Qwen Slot Enricher
    if int(os.environ.get("USE_QWEN_SLOT_ENRICH", "0")):
        n_qse_layers = int(os.environ.get("QWEN_SLOT_ENRICH_LAYERS", "4"))
        qse = _get_qwen_slot_enricher(device, dtype)
        csv = os.environ.get("QWEN_SLOT_ENRICH_COPY_BLOCKS", "")
        if csv:
            qse_indices = [int(s) for s in csv.split(",") if s.strip()]
        else:
            qse_indices = list(range(n_qse_layers))
        assert len(qse_indices) == n_qse_layers, \
            f"QWEN_SLOT_ENRICH_COPY_BLOCKS must have {n_qse_layers} indices"
        log.info(f"qwen_slot_enrich: loading fresh transformer for block-copy from {qse_indices}...")
        _src_xfmr = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model, subfolder="transformer", torch_dtype=dtype)
        qse.load_blocks_from_main(_src_xfmr, qse_indices)
        del _src_xfmr
        torch.cuda.empty_cache()
        log.info(f"qwen_slot_enrich: copied main blocks {qse_indices}")
        lr_qse_blocks = float(os.environ.get("QWEN_SLOT_ENRICH_LR_BLOCKS", "1e-5"))
        lr_qse_aux    = float(os.environ.get("QWEN_SLOT_ENRICH_LR_AUX",    "3e-5"))
        block_p, aux_p = [], []
        for n, p in qse.named_parameters():
            if n.startswith("blocks."):
                block_p.append(p)
            else:
                aux_p.append(p)
        if block_p: param_groups.append({"params": block_p, "lr": lr_qse_blocks})
        if aux_p:   param_groups.append({"params": aux_p,   "lr": lr_qse_aux})
        log.info(f"qwen_slot_enrich: blocks {sum(p.numel() for p in block_p):,} @ lr={lr_qse_blocks:.1e}, "
                 f"aux {sum(p.numel() for p in aux_p):,} @ lr={lr_qse_aux:.1e}")

    if not param_groups:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay)

    LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "constant")  # constant | cosine
    LR_COSINE_TOTAL = int(os.environ.get("LR_COSINE_TOTAL", str(MAX_STEPS)))
    LR_WARMUP = int(os.environ.get("LR_WARMUP", "200"))
    LR_MIN_FRAC = float(os.environ.get("LR_MIN_FRAC", "0.1"))
    if LR_SCHEDULE == "cosine":
        import math as _math
        def _lr_lambda(step):
            if step < LR_WARMUP:
                return float(step) / max(1, LR_WARMUP)
            t = (step - LR_WARMUP) / max(1, LR_COSINE_TOTAL - LR_WARMUP)
            t = min(1.0, max(0.0, t))
            cos = 0.5 * (1.0 + _math.cos(_math.pi * t))
            return LR_MIN_FRAC + (1.0 - LR_MIN_FRAC) * cos
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        log.info(f"LR cosine schedule: warmup={LR_WARMUP} total={LR_COSINE_TOTAL} min_frac={LR_MIN_FRAC}")
    else:
        lr_scheduler = None

    train_ds = VTONDataset(args, split=args.train_split)
    if int(os.environ.get("USE_DPO", "0")):
        _ldir = os.environ.get("DPO_LOSER_CACHE_DIR",
                               "/home/link/Desktop/Code/fashion gen testing/my_vton_cache/dpo_loser_v01")
        _suf = "_deployed_v01_latent.pt"
        _avail = {f[:-len(_suf)] for f in os.listdir(_ldir)} if os.path.isdir(_ldir) else set()
        _before = len(train_ds.image_ids)
        train_ds.image_ids = [i for i in train_ds.image_ids if i in _avail]
        log.info(f"[DPO] filtered train ids to {len(train_ds.image_ids)}/{_before} "
                 f"with frozen loser latents in {_ldir}")
        assert train_ds.image_ids, f"[DPO] no loser latents found in {_ldir}"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True, prefetch_factor=2)

    img_loss_weight = float(os.environ.get("IMG_LOSS_WEIGHT", "0.1"))
    # Soft-mask flow weights: (w_keep, w_garment, w_uncertain)
    w_keep      = float(os.environ.get("W_KEEP", "0.05"))
    w_garment   = float(os.environ.get("W_GARMENT", "1.0"))
    w_uncertain = float(os.environ.get("W_UNCERTAIN", "0.3"))
    loss_weights = (w_keep, w_garment, w_uncertain)

    log.info(f"Architecture: exp419 (soft-mask routing + anti-sludge) — "
             f"{NUM_SLOTS}-slot slot_order={SLOT_ORDER}; "
             f"raw agnostic+rough (no neutralization); "
             f"soft masks: garment_prior(warped_mask_128) + uncertain_band + keep; "
             f"loss = L_flow(w_keep={w_keep}, w_gar={w_garment}, w_ub={w_uncertain}) "
             f"+ {img_loss_weight}*L_img + "
             f"{os.environ.get('LAMBDA_RECON','0.3')}*L_recon_ub + "
             f"{os.environ.get('LAMBDA_ANTISLUDGE','0.1')}*L_antisludge + "
             f"{os.environ.get('LAMBDA_TV','0.01')}*L_tv")
    optimizer.zero_grad(); global_step = 0; micro_step = 0
    accum = {"flow": 0.0, "inv": 0.0, "ab": 0.0, "chroma": 0.0, "cratio": 0.0, "abdir": 0.0, "edge": 0.0, "img": 0.0, "recon": 0.0,
             "anti": 0.0, "tv": 0.0, "alloc": 0.0, "broad": 0.0, "percep": 0.0,
             "antirough": 0.0, "antigrey": 0.0, "adv": 0.0, "critic": 0.0, "late": 0.0, "sigma": 0.0,
             # 5_30 new (instructions7)
             "bnd": 0.0, "bgc": 0.0, "bgf": 0.0, "gem": 0.0, "brs": 0.0, "harm": 0.0, "bel1": 0.0, "belg": 0.0, "gcl": 0.0, "rbi": 0.0, "rsi": 0.0, "tve": 0.0,
             "img_g": 0.0, "img_s": 0.0, "img_b": 0.0, "img_other": 0.0, "img_k": 0.0, "img_ub": 0.0,
             "rou": 0.0, "rep": 0.0, "hf_g": 0.0, "band": 0.0, "bandg": 0.0, "bandhp": 0.0,
             "phk": 0.0, "bsk": 0.0, "bsab": 0.0, "iheab": 0.0, "ihehf": 0.0,
             "midab": 0.0, "midhf": 0.0,
             "dpo": 0.0, "gap": 0.0, "Lwin": 0.0, "Llose": 0.0}
    t_start = time.time()

    import signal
    stop_flag = {"stop": False}
    def _graceful(sig, frame):
        stop_flag["stop"] = True
        log.info(f"Received signal {sig} — finishing current step and saving.")
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS", "99999"))
    SAVE_PER_EPOCH = int(os.environ.get("SAVE_PER_EPOCH", "0"))
    VAL_PER_EPOCH  = int(os.environ.get("VAL_PER_EPOCH",  "0"))
    SAVE_EVERY_STEPS = int(os.environ.get("SAVE_EVERY_STEPS", "0"))
    VAL_EVERY_STEPS  = int(os.environ.get("VAL_EVERY_STEPS",  "0"))
    SAVE_EVERY_SECONDS = float(os.environ.get("SAVE_EVERY_SECONDS", "0"))  # wall-clock save+val cadence (overrides step cadence when >0)
    AUTO_STOP_EPS = float(os.environ.get("AUTO_STOP_EPS", "0.005"))  # rel. drop threshold
    log.info(f"NUM_EPOCHS={NUM_EPOCHS} SAVE_PER_EPOCH={SAVE_PER_EPOCH} VAL_PER_EPOCH={VAL_PER_EPOCH} "
             f"SAVE_EVERY_STEPS={SAVE_EVERY_STEPS} VAL_EVERY_STEPS={VAL_EVERY_STEPS} "
             f"AUTO_STOP_EPS={AUTO_STOP_EPS} VAL_IDS={len(VAL_IDS)}")
    val_img_hist = []  # filled by _run_val — used by auto-stop

    def _save_ckpt_label(label):
        ckpt_dir = os.path.join(args.output_dir, label)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_file(get_peft_model_state_dict(transformer, adapter_name="tryon"),
                  os.path.join(ckpt_dir, "tryon_lora.safetensors"))
        if int(os.environ.get("USE_V6", "0")) and state._V6_HEADS is not None:
            torch.save({k: v.cpu() for k, v in state._V6_HEADS.state_dict().items()},
                       os.path.join(ckpt_dir, "v6_heads.pt"))
        # 5_30 instructions7: save multi_block_injection at every val/best/snap ckpt
        if int(os.environ.get("USE_MULTI_BLOCK_INJ", "0")) and state._MULTI_GAR_INJECTION is not None:
            torch.save({k: v.cpu() for k, v in state._MULTI_GAR_INJECTION.state_dict().items()},
                       os.path.join(ckpt_dir, "multi_block_injection.pt"))
            _mb_tgt = os.environ.get("MULTI_BLOCK_TARGETS",
                "4,8,12,16,20,24,28,32,36,40,44,48,52,56,59")
            with open(os.path.join(ckpt_dir, "multi_block_targets.txt"), "w") as f:
                f.write(_mb_tgt)
        if int(os.environ.get("USE_REGION_MAP", "0")) and state._REGION_ADAPTER is not None:
            torch.save({"adapter": {k: v.cpu() for k, v in state._REGION_ADAPTER.state_dict().items()},
                        "gate": state._REGION_GATE.detach().cpu()},
                       os.path.join(ckpt_dir, "region_map.pt"))
        if int(os.environ.get("USE_INVALID_TOKEN", "0")) and state._INVALID_TOKEN is not None:
            torch.save(state._INVALID_TOKEN.detach().cpu(),
                       os.path.join(ckpt_dir, "invalid_token.pt"))
        log.info(f"[ckpt] saved {label} → {ckpt_dir}")

    def _save_epoch_ckpt(ep):
        if not SAVE_PER_EPOCH: return
        _save_ckpt_label(f"epoch_{ep}")

    def _save_critic(step):
        # Phase-1 critic-only checkpoint (generator untouched in this mode).
        if state._CRITIC is None: return
        cdir = os.path.join(args.output_dir, "critic"); os.makedirs(cdir, exist_ok=True)
        torch.save({k: v.cpu() for k, v in state._CRITIC.state_dict().items()},
                   os.path.join(cdir, "critic.pt"))
        log.info(f"[critic] saved critic.pt @ step {step} → {cdir}")

    VAL_SEED = int(os.environ.get("VAL_SEED", "12345"))

    @torch.no_grad()
    def _run_val_label(label):
        if not VAL_IDS: return
        transformer.eval()
        # Fork RNG so val is deterministic — same model + same seed → same loss.
        _prev_cpu = torch.random.get_rng_state()
        _prev_cuda = torch.cuda.get_rng_state(device=device) if torch.cuda.is_available() else None
        torch.manual_seed(VAL_SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(VAL_SEED)
        try:
            val_ds = VTONDataset(args, split=args.train_split)
            _val_required = ["_person_latent.pt", "_garment_latent.pt", "_degraded_rough_latent.pt",
                              "_agnostic_latent.pt", "_agnostic_mask_latent.pt", "_target_mask.pt",
                              "_warped_mask_128.pt"]
            val_ds.image_ids = [i for i in VAL_IDS if all(
                os.path.exists(os.path.join(val_ds.latent_dir, f"{i}{s}")) for s in _val_required)]
            log.info(f"[val] using {len(val_ds.image_ids)} / {len(VAL_IDS)} requested ids")
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True, collate_fn=collate_fn,
                                    persistent_workers=False)
            accs = {"flow": 0.0, "img": 0.0, "n": 0}
            for vb in val_loader:
                try:
                    with torch.amp.autocast("cuda", dtype=dtype):
                        _, m = train_step(
                            transformer, pose_cache, prompt_cache,
                            vae, person_image_cache, vae_device, img_loss_weight,
                            loss_weights, vb, device, dtype,
                            sigma_beta_alpha=args.sigma_beta_alpha,
                            sigma_beta_beta=args.sigma_beta_beta,
                            global_step=global_step, max_steps=MAX_STEPS)
                    accs["flow"] += m.get("flow", 0.0); accs["img"] += m.get("img", 0.0); accs["n"] += 1
                except Exception as e:
                    log.error(f"[val] {e}")
            if accs["n"]:
                vi = accs["img"] / accs["n"]
                vf = accs["flow"] / accs["n"]
                log.info(f"[val] {label} n={accs['n']} flow={vf:.5f} img={vi:.4f}")
                val_img_hist.append(vi)
            # ── hourly in-loop DEPLOYED-HALO eval (DEPLOY_HALO_EVAL=1): full from-noise
            #    rollout + ring-contrast on the reserved ids, so we WATCH the deployed halo. ──
            if int(os.environ.get("DEPLOY_HALO_EVAL", "0")) and \
               (time.time() - _last_halo_eval["t"]) >= float(os.environ.get("DEPLOY_EVAL_HOURS", "1.0")) * 3600.0:
                _last_halo_eval["t"] = time.time()
                state._DEPLOY_HALO_EVAL = True
                _hc = {"c": 0.0, "r": 0.0, "s": 0.0, "n": 0}
                try:
                    for vb in val_loader:
                        try:
                            with torch.amp.autocast("cuda", dtype=dtype):
                                _, hm = train_step(
                                    transformer, pose_cache, prompt_cache, vae, person_image_cache,
                                    vae_device, img_loss_weight, loss_weights, vb, device, dtype,
                                    sigma_beta_alpha=args.sigma_beta_alpha, sigma_beta_beta=args.sigma_beta_beta,
                                    global_step=global_step, max_steps=MAX_STEPS)
                            _hc["c"] += hm.get("deploy_halo", 0.0); _hc["r"] += hm.get("halo_ring", 0.0)
                            _hc["s"] += hm.get("halo_surr", 0.0); _hc["n"] += 1
                        except Exception as e:
                            log.error(f"[halo] {e}")
                finally:
                    state._DEPLOY_HALO_EVAL = False
                if _hc["n"]:
                    log.info(f"[HALO] {label} step={global_step} n={_hc['n']} "
                             f"contrast={_hc['c']/_hc['n']:+.2f} ring={_hc['r']/_hc['n']:+.2f} surr={_hc['s']/_hc['n']:+.2f}")
        except Exception as e:
            log.error(f"[val] outer error: {e}")
        finally:
            transformer.train()
            torch.random.set_rng_state(_prev_cpu)
            if _prev_cuda is not None:
                torch.cuda.set_rng_state(_prev_cuda, device=device)

    _last_val_step = {"step": -1}
    _last_halo_eval = {"t": 0.0}  # wall-clock timer for the in-loop deployed-halo eval (DEPLOY_HALO_EVAL)
    _last_save_time = {"t": time.time()}  # wall-clock save timer (SAVE_EVERY_SECONDS)
    def _run_val(ep):
        if not VAL_PER_EPOCH: return
        if VAL_EVERY_STEPS > 0 and _last_val_step["step"] == global_step:
            log.info(f"[val] skip epoch={ep} — already val'd at step {global_step}")
            return
        _run_val_label(f"epoch={ep}")
        _last_val_step["step"] = global_step

    # 5_25 instructions8: time-based snapshot for the checkpoint sweep
    _ckpt_interval_sec = int(os.environ.get("CKPT_INTERVAL_SEC", "0"))
    _next_snap_sec = _ckpt_interval_sec
    # Stage-switch for T14/T15: at this step, swap freeze/unfreeze of LoRA vs heads
    _stage_switch_step = int(os.environ.get("STAGE_SWITCH_STEP", "0"))
    _stage_swapped = False

    # 5_29/instructions5: FREEZE_LORA_STEPS=N — freeze LoRA + v6 heads for
    # first N steps to give the garment branches a head start. Auto-unfreeze
    # at step N by piggybacking on the stage-switch mechanism above.
    _freeze_lora_first_n = int(os.environ.get("FREEZE_LORA_STEPS", "0"))
    if _freeze_lora_first_n > 0:
        _stage_switch_step = _freeze_lora_first_n
        os.environ["STAGE2_FREEZE_LORA"] = "0"   # unfreeze at switch
        os.environ["STAGE2_FREEZE_HEADS"] = "0"
        # Pre-freeze NOW: turn off LoRA + v6 grads
        for n, p in transformer.named_parameters():
            if "lora_" in n.lower():
                p.requires_grad_(False)
        if state._V6_HEADS is not None:
            for p in state._V6_HEADS.parameters():
                p.requires_grad_(False)
        log.info(f"[freeze-lora-first-n] LoRA + v6 frozen for first {_freeze_lora_first_n} steps")
    # ── GAN rollout branch (USE_GAN_ROLLOUT): 1 rollout update per GAN_EVERY base updates ──
    _GAN_ON = int(os.environ.get("USE_GAN_ROLLOUT", "0"))
    _gan = {"lambda": None, "count": 0, "base_gnorm": 1.0}
    if _GAN_ON:
        import importlib as _il
        gan_loss = _il.import_module("trainlib.gan_loss")
        gan_rollout = _il.import_module("trainlib.gan_rollout")
        _GAN_EVERY = int(os.environ.get("GAN_EVERY", "4"))
        _GAN_GRAD_START = int(os.environ.get("GAN_GRAD_START", "12"))
        _GAN_RAMP = int(os.environ.get("GAN_RAMP", "250"))
        _GAN_TARGET = float(os.environ.get("GAN_TARGET_RATIO", "0.10"))
        _GAN_RUN_FINAL = os.environ.get("GAN_RUN_FINAL", "")
        _gan_critic = gan_loss.load_gan_critic(vae_device)   # images live on vae_device; cross-device grad → cuda:0
        _gvm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(vae_device, torch.bfloat16)
        _gvs = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(vae_device, torch.bfloat16)
        _gan_params = [p for p in tryon_params if p.requires_grad]
        log.info(f"[gan] ON: every {_GAN_EVERY} base updates, grad_start={_GAN_GRAD_START}, "
                 f"target_ratio={_GAN_TARGET}, ramp={_GAN_RAMP}, run_final={_GAN_RUN_FINAL}, critic FROZEN")

    def _gan_update(batch):
        if not _GAN_ON:
            return
        try:
            seed = 70000 + _gan["count"]
            L_gan, _pf, gd = gan_rollout.rollout_gan_loss(
                transformer, batch, vae, _gvm, _gvs, vae_device, _gan_critic,
                prompt_cache, pose_cache, dtype, device, args, _GAN_RUN_FINAL,
                person_image_cache, seed, grad_start=_GAN_GRAD_START)
            if L_gan is None:
                return
            if _gan["lambda"] is None:                         # calibrate once from grad magnitudes
                g_gan = gan_loss.grad_norm_of(L_gan, _gan_params, retain=True)
                _gan["lambda"] = gan_loss.calibrate_lambda_gan(max(_gan["base_gnorm"], 1e-6), g_gan, _GAN_TARGET)
                log.info(f"[gan] calibrated lambda_gan={_gan['lambda']:.4g} (g_base={_gan['base_gnorm']:.3f} g_gan={g_gan:.3f})")
            lam = _gan["lambda"] * min(1.0, _gan["count"] / max(1, _GAN_RAMP))
            optimizer.zero_grad()
            (lam * L_gan).backward()
            torch.nn.utils.clip_grad_norm_(tryon_params, args.max_grad_norm)
            if state._MULTI_GAR_INJECTION is not None:
                torch.nn.utils.clip_grad_norm_(list(state._MULTI_GAR_INJECTION.parameters()), args.max_grad_norm)
            optimizer.step()                                  # LR scheduler driven by base steps only
            optimizer.zero_grad()
            _gan["count"] += 1
            if _gan["count"] % 25 == 1:
                log.info(f"[gan] upd={_gan['count']} step={global_step} L_gan={float(L_gan):.3f} "
                         f"D(gt)={gd['d_gt']:+.2f} D(pred)={gd['d_fake']:+.2f} gap={gd['gap']:+.2f} lam={lam:.4g}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.error(f"[gan] OOM at step {global_step} — skipped rollout (consider GAN_GRAD_START up)")
        except Exception as _ge:
            import traceback as _tb
            log.error(f"[gan] rollout error at step {global_step}: {_ge}\n{_tb.format_exc()}")

    # ── RUN_DIAGNOSTIC: gradient-conflict probe + Check C one-sample overfit, then exit ──
    if int(os.environ.get("RUN_DIAGNOSTIC", "0")):
        import torch.autograd as _ag
        transformer.train()
        _lp = [p for n, p in transformer.named_parameters() if p.requires_grad and "lora_" in n.lower()]
        _batch = next(iter(train_loader))
        os.environ["FIXED_SIGMA"] = os.environ.get("DIAG_SIGMA", "0.30")
        os.environ["BAND_SIGMA_BIAS"] = "0"
        _ts_args = (pose_cache, prompt_cache, vae, person_image_cache, vae_device, img_loss_weight, loss_weights)
        def _step(seed, gstep):
            torch.manual_seed(seed)
            return train_step(transformer, *_ts_args, _batch, device, dtype,
                              sigma_beta_alpha=args.sigma_beta_alpha, sigma_beta_beta=args.sigma_beta_beta,
                              global_step=gstep, max_steps=MAX_STEPS)
        # (1) gradient conflict at σ=DIAG_SIGMA, fixed noise
        os.environ["RETURN_LOSS_TENSORS"] = "1"
        _loss, _m = _step(1234, 0)
        _T = _m["_T"]
        def _fg(t):
            gs = _ag.grad(t, _lp, retain_graph=True, allow_unused=True)
            return torch.cat([(g.detach().flatten() if g is not None else torch.zeros(p.numel(), device=p.device)) for g, p in zip(gs, _lp)])
        gF, gB, gR = _fg(_T["flow"]), _fg(_T["band"]), _fg(_T["recon"])
        def _cos(a, b): return float((a @ b) / (a.norm() * b.norm() + 1e-12))
        log.info(f"[diag] ||g_flow||={gF.norm():.4f} ||g_band||={gB.norm():.4f} ||g_recon||={gR.norm():.4f}")
        log.info(f"[diag] cos(band,flow)={_cos(gB, gF):+.4f}  cos(band,recon)={_cos(gB, gR):+.4f}  cos(flow,recon)={_cos(gF, gR):+.4f}")
        os.environ["RETURN_LOSS_TENSORS"] = "0"
        # (1b) does the LoRA see gradient through loss.backward() under the REAL autocast loop?
        for _ac in (False, True):
            optimizer.zero_grad()
            if _ac:
                with torch.amp.autocast("cuda", dtype=dtype):
                    _la, _ = _step(2024, 0)
                _la.backward()
            else:
                _la, _ = _step(2024, 0); _la.backward()
            _gn = sum(float(p.grad.abs().sum()) for n, p in transformer.named_parameters()
                      if "lora_" in n.lower() and p.grad is not None)
            log.info(f"[diag] LoRA grad-sum via backward (autocast={_ac}): {_gn:.4e}")
        optimizer.zero_grad()
        # (2) Check C: overfit this one sample (fixed σ + fixed noise) for DIAG_OVERFIT_STEPS
        # Set a real LR — the cosine scheduler starts at lr_lambda(0)=0 (warmup), so the
        # optimizer LR is 0 here and we never step the scheduler. Pin it explicitly.
        _diag_lr = float(os.environ.get("DIAG_LR", "2e-4"))
        for _pg in optimizer.param_groups:
            _pg["lr"] = _diag_lr
        log.info(f"[diag-overfit] LR pinned to {_diag_lr}")
        _ns = int(os.environ.get("DIAG_OVERFIT_STEPS", "300"))
        for _st in range(1, _ns + 1):
            optimizer.zero_grad()
            _l, _mm = _step(777, _st)
            _l.backward(); torch.nn.utils.clip_grad_norm_(tryon_params, args.max_grad_norm)
            if state._MULTI_GAR_INJECTION is not None:
                torch.nn.utils.clip_grad_norm_(list(state._MULTI_GAR_INJECTION.parameters()), args.max_grad_norm)
            optimizer.step()
            if _st == 1 or _st % 20 == 0:
                log.info(f"[diag-overfit] step {_st} band={_mm.get('band', 0):.5f} flow={_mm.get('flow', 0):.5f} recon={_mm.get('recon', 0):.5f}")
        log.info("[diag] done"); return

    for epoch in range(NUM_EPOCHS):
        transformer.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            if time.time() - t_start >= TIME_BUDGET or stop_flag["stop"] or global_step >= MAX_STEPS: break
            # time-based snapshot
            if _ckpt_interval_sec > 0:
                _el = time.time() - t_start
                if _el >= _next_snap_sec:
                    _save_ckpt_label(f"snap_{int(_next_snap_sec)}s")
                    _next_snap_sec += _ckpt_interval_sec
            # stage switch (T14/T15)
            if _stage_switch_step > 0 and not _stage_swapped and global_step >= _stage_switch_step:
                _stage_swapped = True
                # toggle: whatever was frozen becomes trainable and vice versa
                # (driven by STAGE2_FREEZE_LORA / STAGE2_FREEZE_HEADS env)
                _s2_freeze_lora = int(os.environ.get("STAGE2_FREEZE_LORA", "0"))
                _s2_freeze_heads = int(os.environ.get("STAGE2_FREEZE_HEADS", "0"))
                for n, p in transformer.named_parameters():
                    if "lora_" in n.lower():
                        p.requires_grad_(not _s2_freeze_lora)
                if state._V6_HEADS is not None:
                    for p in state._V6_HEADS.parameters():
                        p.requires_grad_(not _s2_freeze_heads)
                if state._REGION_ADAPTER is not None:
                    for p in state._REGION_ADAPTER.parameters():
                        p.requires_grad_(not _s2_freeze_heads)
                    state._REGION_GATE.requires_grad_(not _s2_freeze_heads)
                if state._INVALID_TOKEN is not None:
                    state._INVALID_TOKEN.requires_grad_(not _s2_freeze_heads)
                log.info(f"[stage-switch] step={global_step}: lora_frozen={bool(_s2_freeze_lora)} heads_frozen={bool(_s2_freeze_heads)}")
            try:
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss, metrics = train_step(
                        transformer,
                        pose_cache, prompt_cache,
                        vae, person_image_cache, vae_device, img_loss_weight,
                        loss_weights,
                        batch, device, dtype,
                        sigma_beta_alpha=args.sigma_beta_alpha,
                        sigma_beta_beta=args.sigma_beta_beta,
                        global_step=global_step,
                        max_steps=MAX_STEPS)
                if int(os.environ.get("CRITIC_PRETRAIN", "0")):
                    # ── Phase 1: critic-only. Generator forward produced the model-fake
                    #    inside train_step; the critic already stepped. Generator is NOT
                    #    updated (no backward / no optimizer.step). ──
                    for k in accum: accum[k] += metrics.get(k, 0.0)
                    micro_step += 1
                    if micro_step % args.grad_accum == 0:
                        global_step += 1
                        if global_step % args.logging_steps == 0 and state._LAST_CRIT:
                            _c = state._LAST_CRIT
                            _cs = " ".join(f"{k[2:]}={v:+.2f}" for k, v in _c.items() if k.startswith("c_"))
                            log.info(f"[critic-pretrain] step={global_step}  "
                                     f"real={_c.get('real',0):+.2f}  fake={_c.get('fake',0):+.2f}  "
                                     f"rankd={_c.get('rankd',float('nan')):+.2f}  "
                                     f"harm={_c.get('harm',float('nan')):+.2f}  "
                                     f"wrong={_c.get('wrong',float('nan')):+.2f}  | {_cs}")
                        if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                            _save_critic(global_step)
                    continue
                (loss / args.grad_accum).backward()
                for k in accum: accum[k] += metrics.get(k, 0.0)
                micro_step += 1
                if micro_step % args.grad_accum == 0:
                    _bn = torch.nn.utils.clip_grad_norm_(tryon_params, args.max_grad_norm)
                    _gan["base_gnorm"] = float(_bn)              # for lambda_gan calibration
                    # 5_30: also clip multi-block injection params (708M, fresh init, can blow up)
                    if state._MULTI_GAR_INJECTION is not None:
                        torch.nn.utils.clip_grad_norm_(
                            list(state._MULTI_GAR_INJECTION.parameters()), args.max_grad_norm)
                    optimizer.step()
                    if lr_scheduler is not None: lr_scheduler.step()
                    optimizer.zero_grad(); global_step += 1
                    if _GAN_ON and global_step % _GAN_EVERY == 0:   # GAN rollout cadence
                        _gan_update(batch)
                    # Per-step ckpt + val (e.g., half-epoch)
                    # Lean default: keep only a rolling "latest" (one recent) +
                    # "best_val" (one best). Set SAVE_ALL_CKPTS=1 to retain every
                    # step_/val_ snapshot (old, disk-heavy behavior: ~10 copies/run).
                    _save_all = int(os.environ.get("SAVE_ALL_CKPTS", "0"))
                    # wall-clock trigger: fire a save+val every SAVE_EVERY_SECONDS regardless of step rate
                    _time_trig = SAVE_EVERY_SECONDS > 0 and (time.time() - _last_save_time["t"]) >= SAVE_EVERY_SECONDS
                    if _time_trig: _last_save_time["t"] = time.time()
                    _is_val_step = (VAL_EVERY_STEPS > 0 and global_step % VAL_EVERY_STEPS == 0) or _time_trig
                    if (SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0) or _time_trig:
                        if _save_all:
                            _save_ckpt_label(f"step_{global_step}")
                        elif not _is_val_step:
                            _save_ckpt_label("latest")  # a val step writes "latest" itself
                    if _is_val_step:
                        _run_val_label(f"step={global_step}")
                        _last_val_step["step"] = global_step
                        # Recent-val ckpt → rolling "latest" (overwrites). Best-so-far
                        # also mirrored to "best_val/" for easy retrieval.
                        if _save_all and int(os.environ.get("SAVE_EVERY_VAL", "1")):
                            _save_ckpt_label(f"val_{global_step}")
                        else:
                            _save_ckpt_label("latest")
                        if val_img_hist and val_img_hist[-1] == min(val_img_hist):
                            log.info(f"[save-best] new best val_img={val_img_hist[-1]:.4f} at step {global_step} — saving best_val/")
                            _save_ckpt_label("best_val")
                        # 5_30 v2: stop only if val_img fails to improve the
                        # BEST val seen so far for 2 consecutive checks.
                        if len(val_img_hist) >= 3:
                            best = min(val_img_hist[:-2])  # best before the last 2
                            cur  = val_img_hist[-1]
                            prev = val_img_hist[-2]
                            cur_fail  = cur  > best - AUTO_STOP_EPS
                            prev_fail = prev > best - AUTO_STOP_EPS
                            if cur_fail and prev_fail:
                                log.info(f"[auto-stop] val_img failed to improve best={best:.4f} for 2 consecutive vals (prev={prev:.4f}, cur={cur:.4f}) — stopping.")
                                stop_flag["stop"] = True
                            else:
                                log.info(f"[auto-stop] val_img: best={best:.4f} prev={prev:.4f} cur={cur:.4f} (cur_fail={cur_fail} prev_fail={prev_fail}) — continuing.")
                    if global_step % args.logging_steps == 0:
                        n = args.logging_steps * args.grad_accum
                        log.info(f"step {global_step} flow={accum['flow']/n:.5f} "
                                 f"inv={accum.get('inv', 0.0)/n:.5f} "
                                 f"ab={accum.get('ab', 0.0)/n:.4f} chroma={accum.get('chroma', 0.0)/n:.4f} "
                                 f"cratio={accum.get('cratio', 0.0)/n:.4f} abdir={accum.get('abdir', 0.0)/n:.4f} edge={accum.get('edge', 0.0)/n:.4f} "
                                 f"img={accum['img']/n:.4f} recon={accum['recon']/n:.4f} "
                                 f"anti={accum['anti']/n:.4f} tv={accum['tv']/n:.5f} "
                                 f"band={accum.get('band', 0.0)/n:.4f} bandg={accum.get('bandg', 0.0)/n:.3f} bandhp={accum.get('bandhp', 0.0)/n:.4f} "
                                 f"alloc={accum['alloc']/n:.4f} broad={accum['broad']/n:.4f} "
                                 f"percep={accum['percep']/n:.4f} "
                                 f"antirough={accum.get('antirough', 0.0)/n:.5f} "
                                 f"phk={accum.get('phk', 0.0)/n:.4f} "
                                 f"bsk={accum.get('bsk', 0.0)/n:.4f} bsab={accum.get('bsab', 0.0)/n:.4f} "
                                 f"iheab={accum.get('iheab', 0.0)/n:.4f} ihehf={accum.get('ihehf', 0.0)/n:.4f} "
                                 f"midab={accum.get('midab', 0.0)/n:.4f} midhf={accum.get('midhf', 0.0)/n:.4f} "
                                 # 5_30 new losses (instructions7) — raw values, multiply by env-var λ for contribution
                                 f"bnd={accum.get('bnd', 0.0)/n:.4f} "
                                 f"bgc={accum.get('bgc', 0.0)/n:.4f} "
                                 f"bgf={accum.get('bgf', 0.0)/n:.4f} "
                                 f"gem={accum.get('gem', 0.0)/n:.4f} "
                                 f"brs={accum.get('brs', 0.0)/n:.4f} "
                                 f"harm={accum.get('harm', 0.0)/n:.4f} "
                                 f"bel1={accum.get('bel1',0.0)/n:.4f} belg={accum.get('belg',0.0)/n:.4f} "
                                 f"img_g={accum.get('img_g',0.0)/n:.4f} img_s={accum.get('img_s',0.0)/n:.4f} "
                                 f"img_b={accum.get('img_b',0.0)/n:.4f} img_other={accum.get('img_other',0.0)/n:.4f} "
                                 f"img_k={accum.get('img_k',0.0)/n:.4f} img_ub={accum.get('img_ub',0.0)/n:.4f} "
                                 f"gcl={accum.get('gcl', 0.0)/n:.4f} "
                                 f"rbi={accum.get('rbi', 0.0)/n:.4f} "
                                 f"rsi={accum.get('rsi', 0.0)/n:.4f} "
                                 f"tve={accum.get('tve', 0.0)/n:.4f} "
                                 f"rou={accum.get('rou', 0.0)/n:.4f} "
                                 f"rep={accum.get('rep', 0.0)/n:.4f} "
                                 f"hf_g={accum.get('hf_g', 0.0)/n:.4f} "
                                 f"σ={accum['sigma']/n:.3f}")
                        if int(os.environ.get("USE_DPO", "0")):
                            log.info(f"[DPO] step {global_step} dpo={accum['dpo']/n:.5f} "
                                     f"gap(Llose-Lwin)={accum['gap']/n:+.6f} "
                                     f"Lwin={accum['Lwin']/n:.5f} Llose={accum['Llose']/n:.5f} "
                                     f"σ={accum['sigma']/n:.3f}  (want gap → positive & growing)")
                        accum = {k: 0.0 for k in accum}
            except Exception as e:
                log.error(f"Error step {global_step}: {e}")
                log.error(traceback.format_exc())
                if "cuda" in str(e).lower(): sys.exit(1)
                continue
        if time.time() - t_start >= TIME_BUDGET or stop_flag["stop"] or global_step >= MAX_STEPS: break
        log.info(f"Epoch {epoch} done at step {global_step}")
        _save_epoch_ckpt(epoch)
        _run_val(epoch)
        # 5_30 v2: stop if val_img fails to improve BEST for 2 consecutive vals
        if len(val_img_hist) >= 3:
            best = min(val_img_hist[:-2])
            cur  = val_img_hist[-1]
            prev = val_img_hist[-2]
            cur_fail  = cur  > best - AUTO_STOP_EPS
            prev_fail = prev > best - AUTO_STOP_EPS
            if cur_fail and prev_fail:
                log.info(f"[auto-stop] val_img failed to improve best={best:.4f} for 2 consecutive vals (prev={prev:.4f}, cur={cur:.4f}) — stopping.")
                stop_flag["stop"] = True
            else:
                log.info(f"[auto-stop] val_img: best={best:.4f} prev={prev:.4f} cur={cur:.4f} — continuing.")

    # ── Save for inference ──
    final_path = os.path.join(args.output_dir, "final"); os.makedirs(final_path, exist_ok=True)
    if state._CRITIC is not None:
        _save_critic("final")
    save_file(get_peft_model_state_dict(transformer, adapter_name="tryon"),
              os.path.join(final_path, "tryon_lora.safetensors"))
    # If proj_out / norm_out were unfrozen (UNFREEZE_PROJ_OUT=1), save them so
    # inference can reload the trained values. Otherwise inference uses the base
    # pretrained transformer's proj_out/norm_out untouched.
    if int(os.environ.get("UNFREEZE_PROJ_OUT", "0")):
        _xfmr_inner = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
        proj_out_sd = {f"norm_out.{k}": v.cpu() for k, v in _xfmr_inner.norm_out.state_dict().items()}
        proj_out_sd.update({f"proj_out.{k}": v.cpu() for k, v in _xfmr_inner.proj_out.state_dict().items()})
        torch.save(proj_out_sd, os.path.join(final_path, "proj_out_norm_out.pt"))
        log.info(f"Saved proj_out_norm_out.pt ({sum(v.numel() for v in proj_out_sd.values()):,} params)")
    if int(os.environ.get("USE_V6", "0")) and state._V6_HEADS is not None:
        torch.save({k: v.cpu() for k, v in state._V6_HEADS.state_dict().items()},
                   os.path.join(final_path, "v6_heads.pt"))
        log.info(f"Saved v6_heads.pt")
    if int(os.environ.get("USE_REGION_MAP", "0")) and state._REGION_ADAPTER is not None:
        torch.save({"adapter": {k: v.cpu() for k, v in state._REGION_ADAPTER.state_dict().items()},
                    "gate": state._REGION_GATE.detach().cpu()},
                   os.path.join(final_path, "region_map.pt"))
        log.info(f"Saved region_map.pt")
    if int(os.environ.get("USE_INVALID_TOKEN", "0")) and state._INVALID_TOKEN is not None:
        torch.save(state._INVALID_TOKEN.detach().cpu(),
                   os.path.join(final_path, "invalid_token.pt"))
        log.info(f"Saved invalid_token.pt")
    if int(os.environ.get("USE_GARMENT_NET", "0")) and state._GARMENT_NET is not None:
        torch.save({k: v.cpu() for k, v in state._GARMENT_NET.state_dict().items()},
                   os.path.join(final_path, "garment_net.pt"))
        log.info(f"Saved garment_net.pt")
    if int(os.environ.get("USE_GARMENT_XATTN", "0")):
        if state._GARMENT_ENCODER is not None:
            torch.save({k: v.cpu() for k, v in state._GARMENT_ENCODER.state_dict().items()},
                       os.path.join(final_path, "garment_encoder.pt"))
            log.info(f"Saved garment_encoder.pt")
        if state._GARMENT_XATTN is not None:
            torch.save({k: v.cpu() for k, v in state._GARMENT_XATTN.state_dict().items()},
                       os.path.join(final_path, "garment_xattn.pt"))
            log.info(f"Saved garment_xattn.pt")
    # 5_30: multi-block injection ckpt
    if int(os.environ.get("USE_MULTI_BLOCK_INJ", "0")) and state._MULTI_GAR_INJECTION is not None:
        torch.save({k: v.cpu() for k, v in state._MULTI_GAR_INJECTION.state_dict().items()},
                   os.path.join(final_path, "multi_block_injection.pt"))
        # Persist target blocks
        _mb_tgt = os.environ.get("MULTI_BLOCK_TARGETS",
            "4,8,12,16,20,24,28,32,36,40,44,48,52,56,59")
        with open(os.path.join(final_path, "multi_block_targets.txt"), "w") as f:
            f.write(_mb_tgt)
        log.info(f"Saved multi_block_injection.pt")
    if int(os.environ.get("USE_QWEN_REFINER", "0")) and state._QWEN_REFINER is not None:
        torch.save({k: v.cpu() for k, v in state._QWEN_REFINER.state_dict().items()},
                   os.path.join(final_path, "qwen_refiner.pt"))
        log.info("Saved qwen_refiner.pt")
    if int(os.environ.get("USE_CONTROLNET", "0")) and state._CONTROLNET is not None:
        torch.save({k: v.cpu() for k, v in state._CONTROLNET.state_dict().items()},
                   os.path.join(final_path, "controlnet.pt"))
        # Also persist the inject blocks so inference can reproduce wiring
        cn_inject = os.environ.get("CONTROLNET_INJECT_BLOCKS", "")
        if not cn_inject:
            cn_inject = ",".join(str(i) for i in range(int(os.environ.get("CONTROLNET_LAYERS", "12"))))
        with open(os.path.join(final_path, "controlnet_meta.json"), "w") as f:
            import json as _json
            _json.dump({"inject_blocks": [int(s) for s in cn_inject.split(",") if s.strip()],
                        "n_layers": int(os.environ.get("CONTROLNET_LAYERS", "12"))}, f)
        log.info("Saved controlnet.pt + controlnet_meta.json")
    if int(os.environ.get("USE_QWEN_AUX_SLOT", "0")) and state._QWEN_AUX_SLOT is not None:
        torch.save({k: v.cpu() for k, v in state._QWEN_AUX_SLOT.state_dict().items()},
                   os.path.join(final_path, "qwen_aux_slot.pt"))
        log.info("Saved qwen_aux_slot.pt")
    if int(os.environ.get("USE_QWEN_SLOT_ENRICH", "0")) and state._QWEN_SLOT_ENRICHER is not None:
        torch.save({k: v.cpu() for k, v in state._QWEN_SLOT_ENRICHER.state_dict().items()},
                   os.path.join(final_path, "qwen_slot_enricher.pt"))
        log.info("Saved qwen_slot_enricher.pt")
    if int(os.environ.get("USE_GARMENT_OOTD", "0")) and state._GARMENT_BRANCH is not None:
        torch.save({k: v.cpu() for k, v in state._GARMENT_BRANCH.state_dict().items()},
                   os.path.join(final_path, "garment_branch.pt"))
        log.info(f"Saved garment_branch.pt")
        if state._OOTD_INJECTORS:
            inj_save = {
                "inject_block_indices": sorted(state._OOTD_INJECTORS.keys()),
                "injectors": {str(k): {kk: vv.cpu() for kk, vv in v.state_dict().items()}
                              for k, v in state._OOTD_INJECTORS.items()},
            }
            torch.save(inj_save, os.path.join(final_path, "ootd_injectors.pt"))
            log.info(f"Saved ootd_injectors.pt for blocks {inj_save['inject_block_indices']}")
    # Lazy caches (full-train mode) don't have .items(). For inference we only need
    # the 5 VTON IDs (panel/eval set), so populate prompt_cache with those entries
    # from disk and save for downstream inference compatibility.
    if hasattr(prompt_cache, "items"):
        torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in prompt_cache.items()},
                   os.path.join(final_path, "prompt_cache.pt"))
    else:
        _eval_ids = ["00006_00", "00008_00", "00013_00", "00017_00", "00034_00"]
        _pc_save = {}
        for _sid in _eval_ids:
            try:
                pe, pm = prompt_cache[_sid]
                _pc_save[_sid] = (pe.cpu(), pm.cpu())
            except FileNotFoundError:
                pass
        torch.save(_pc_save, os.path.join(final_path, "prompt_cache.pt"))
    if hasattr(pose_cache, "items"):
        torch.save({k: v.cpu() for k, v in pose_cache.items()},
                   os.path.join(final_path, "pose_latent_cache.pt"))
    else:
        torch.save({}, os.path.join(final_path, "pose_latent_cache.pt"))
    # Patch inference template with this run's slot order + rough-neutralize flag
    with open(INFERENCE_TEMPLATE) as f:
        template_src = f.read()
    template_src = template_src.replace(
        "DEFAULT_SLOT_ORDER = [0, 1, 2, 3]",
        f"DEFAULT_SLOT_ORDER = {SLOT_ORDER_IDX}  # {SLOT_ORDER}"
    )
    neutralize_rough_val = int(os.environ.get("NEUTRALIZE_ROUGH", "1"))
    template_src = template_src.replace(
        "NEUTRALIZE_ROUGH = 1",
        f"NEUTRALIZE_ROUGH = {neutralize_rough_val}"
    )
    lrr_val = float(os.environ.get("LAMBDA_REPAIR_ROUGH", "0.25"))
    template_src = template_src.replace(
        "LAMBDA_REPAIR_ROUGH = 0.25",
        f"LAMBDA_REPAIR_ROUGH = {lrr_val}"
    )
    proxy_val = float(os.environ.get("PROXY_CORE_THRESH", "0.6"))
    template_src = template_src.replace(
        "diff_in_agn > 0.6 * max_diff",
        f"diff_in_agn > {proxy_val} * max_diff"
    )
    attn_mode_map = {"none": 0, "one_way": 1, "both": 2}
    attn_mode_val = attn_mode_map.get(os.environ.get("ATTN_MASK_MODE", "none"), 0)
    template_src = template_src.replace(
        "ATTN_MASK_MODE = 0",
        f"ATTN_MASK_MODE = {attn_mode_val}"
    )
    pin_ring_val = int(os.environ.get("PIN_RING_ROUGH", "0"))
    template_src = template_src.replace(
        "PIN_RING_ROUGH = 0   # Set by train.py at save time",
        f"PIN_RING_ROUGH = {pin_ring_val}"
    )
    sigma_sched_val = int(os.environ.get("USE_SIGMA_SCHED", "0"))
    template_src = template_src.replace(
        "USE_SIGMA_SCHED = 0  # Set by train.py at save time",
        f"USE_SIGMA_SCHED = {sigma_sched_val}"
    )
    sched_lo = float(os.environ.get("SIGMA_SCHED_LO", "0.8"))
    sched_hi = float(os.environ.get("SIGMA_SCHED_HI", "1.2"))
    template_src = template_src.replace(
        "SIGMA_SCHED_LO = 0.8 # Set by train.py at save time",
        f"SIGMA_SCHED_LO = {sched_lo}"
    )
    template_src = template_src.replace(
        "SIGMA_SCHED_HI = 1.2 # Set by train.py at save time",
        f"SIGMA_SCHED_HI = {sched_hi}"
    )
    pure_noise_val = int(os.environ.get("USE_PURE_NOISE", "0"))
    template_src = template_src.replace(
        "USE_PURE_NOISE = 0   # Set by train.py at save time",
        f"USE_PURE_NOISE = {pure_noise_val}"
    )
    rough_masked_val = int(os.environ.get("USE_ROUGH_MASKED", "0"))
    template_src = template_src.replace(
        "USE_ROUGH_MASKED = 0  # Set by train.py at save time",
        f"USE_ROUGH_MASKED = {rough_masked_val}"
    )
    agn_mean_fill_val = int(os.environ.get("USE_AGNOSTIC_MEAN_FILL", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_MEAN_FILL = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_MEAN_FILL = {agn_mean_fill_val}"
    )
    agn_rough_fill_val = int(os.environ.get("USE_AGNOSTIC_ROUGH_FILL", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_ROUGH_FILL = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_ROUGH_FILL = {agn_rough_fill_val}"
    )
    agn_inpaint_val = int(os.environ.get("USE_AGNOSTIC_INPAINT", "0"))
    template_src = template_src.replace(
        "USE_AGNOSTIC_INPAINT = 0  # Set by train.py at save time",
        f"USE_AGNOSTIC_INPAINT = {agn_inpaint_val}"
    )
    sil_scale_val = float(os.environ.get("SILHOUETTE_SCALE", "1.0"))
    template_src = template_src.replace(
        "SILHOUETTE_SCALE = 1.0  # Set by train.py at save time",
        f"SILHOUETTE_SCALE = {sil_scale_val}"
    )
    sil_soft_val = int(os.environ.get("SILHOUETTE_SOFT", "0"))
    template_src = template_src.replace(
        "SILHOUETTE_SOFT = 0  # Set by train.py at save time",
        f"SILHOUETTE_SOFT = {sil_soft_val}"
    )
    vae_sil_val = int(os.environ.get("USE_VAE_SILHOUETTE", "0"))
    template_src = template_src.replace(
        "USE_VAE_SILHOUETTE = 0  # Set by train.py at save time",
        f"USE_VAE_SILHOUETTE = {vae_sil_val}"
    )
    agn_inp_soft_val = float(os.environ.get("AGNOSTIC_INPAINT_SOFT_SIG", "0.0"))
    template_src = template_src.replace(
        "AGNOSTIC_INPAINT_SOFT_SIG = 0.0  # Set by train.py at save time",
        f"AGNOSTIC_INPAINT_SOFT_SIG = {agn_inp_soft_val}"
    )
    agn_zero_rep_val = int(os.environ.get("AGNOSTIC_ZERO_REPAIR", "0"))
    template_src = template_src.replace(
        "AGNOSTIC_ZERO_REPAIR = 0  # Set by train.py at save time",
        f"AGNOSTIC_ZERO_REPAIR = {agn_zero_rep_val}"
    )
    use_repair_mask_val = int(os.environ.get("USE_REPAIR_ATTN_MASK", "0"))
    template_src = template_src.replace(
        "USE_REPAIR_ATTN_MASK = 0  # Set by train.py at save time",
        f"USE_REPAIR_ATTN_MASK = {use_repair_mask_val}"
    )
    bg_hint_val = int(os.environ.get("USE_BG_HINT", "0"))
    template_src = template_src.replace(
        "USE_BG_HINT = 0  # Set by train.py at save time",
        f"USE_BG_HINT = {bg_hint_val}"
    )
    bg_hint_scale_val = float(os.environ.get("BG_HINT_SCALE", "1.0"))
    template_src = template_src.replace(
        "BG_HINT_SCALE = 1.0  # Set by train.py at save time",
        f"BG_HINT_SCALE = {bg_hint_scale_val}"
    )
    v6_zero_g_val = int(os.environ.get("V6_ZERO_G_CORE", "0")) if int(os.environ.get("USE_V6", "0")) else 0
    template_src = template_src.replace(
        "V6_ZERO_G_CORE = 0  # Set by train.py at save time",
        f"V6_ZERO_G_CORE = {v6_zero_g_val}"
    )
    v6_r_in_val = int(os.environ.get("V6_R_IN", "2"))
    template_src = template_src.replace(
        "V6_R_IN = 2  # Set by train.py at save time",
        f"V6_R_IN = {v6_r_in_val}"
    )
    v6_r_out_val = int(os.environ.get("V6_R_OUT", "7"))
    template_src = template_src.replace(
        "V6_R_OUT = 7  # Set by train.py at save time",
        f"V6_R_OUT = {v6_r_out_val}"
    )
    sil_soft_sig_val = float(os.environ.get("SILHOUETTE_SOFT_SIG", "2.0"))
    template_src = template_src.replace(
        "SILHOUETTE_SOFT_SIG = 2.0  # Set by train.py at save time",
        f"SILHOUETTE_SOFT_SIG = {sil_soft_sig_val}"
    )
    rough_mask_soft_val = int(os.environ.get("ROUGH_MASK_SOFT", "0"))
    template_src = template_src.replace(
        "ROUGH_MASK_SOFT = 0  # Set by train.py at save time",
        f"ROUGH_MASK_SOFT = {rough_mask_soft_val}"
    )
    rough_mask_soft_sig_val = float(os.environ.get("ROUGH_MASK_SOFT_SIG", "3.0"))
    template_src = template_src.replace(
        "ROUGH_MASK_SOFT_SIG = 3.0  # Set by train.py at save time",
        f"ROUGH_MASK_SOFT_SIG = {rough_mask_soft_sig_val}"
    )
    rough_blur_fixed_val = int(os.environ.get("USE_ROUGH_BLUR_FIXED", "0"))
    rough_blur_sig_val   = float(os.environ.get("ROUGH_BLUR_FIXED_SIG", "4.0"))
    template_src = template_src.replace(
        "USE_ROUGH_BLUR_FIXED = 0  # Set by train.py at save time",
        f"USE_ROUGH_BLUR_FIXED = {rough_blur_fixed_val}"
    )
    template_src = template_src.replace(
        "ROUGH_BLUR_FIXED_SIG = 4.0  # Set by train.py at save time",
        f"ROUGH_BLUR_FIXED_SIG = {rough_blur_sig_val}"
    )
    # v19/v20/v21: bake AgnosticCtrlProcessor config into inference.py
    _agn_ctrl_defaults = {
        "USE_AGN_CTRL": ("0", "int"),
        "AGN_KEY_BIAS": ("0", "int"),
        "AGN_V_SCALE": ("0", "int"),
        "AGN_KEY_BIAS_ALPHA": ("0.5", "float"),
        "AGN_TRUST_CORE": ("0.3", "float"),
        "AGN_TRUST_BND": ("0.85", "float"),
        "AGN_TRUST_KEEP": ("1.0", "float"),
        "AGN_TRUST_EPS": ("0.001", "float"),
        "AGN_VSCALE_CORE": ("0.5", "float"),
        "AGN_VSCALE_BND": ("0.9", "float"),
        "AGN_VSCALE_KEEP": ("1.0", "float"),
        "AGN_TRUST_K": ("3", "int"),
        "AGN_ERODE": ("2", "int"),
        "AGN_DILATE": ("2", "int"),
        "USE_INVALID_TOKEN": ("0", "int"),
        "INVALID_TOKEN_K": ("3", "int"),
        "INVALID_TOKEN_ERODE": ("2", "int"),
        "INVALID_TOKEN_DILATE": ("2", "int"),
        "INVALID_TOKEN_BND_VALID": ("0.7", "float"),
        "USE_ZERO_AGNOSTIC_SLOT": ("0", "int"),
    }
    for _name, (_dflt, _typ) in _agn_ctrl_defaults.items():
        if _typ == "int":
            _val = int(os.environ.get(_name, _dflt))
        else:
            _val = float(os.environ.get(_name, _dflt))
        template_src = template_src.replace(
            f"{_name} = {_dflt}  # Set by train.py at save time",
            f"{_name} = {_val}  # Set by train.py at save time")

    with open(os.path.join(final_path, "inference.py"), "w") as f:
        f.write(template_src)

    # Copy the loss comparison / experiment plan doc into the run dir if present
    loss_doc = os.path.join(BASE, "vtonautoresearch", "loss_comparison_exp395_to_exp409.md")
    if os.path.exists(loss_doc):
        shutil.copy2(loss_doc, os.path.join(final_path, "loss_comparison.md"))

    # Lean default: "final/" is now the canonical recent ckpt, so drop the rolling
    # "latest/" written during training. End state per run = best_val/ + final/.
    if not int(os.environ.get("SAVE_ALL_CKPTS", "0")):
        _latest_dir = os.path.join(args.output_dir, "latest")
        if os.path.isdir(_latest_dir):
            shutil.rmtree(_latest_dir, ignore_errors=True)
            log.info("[ckpt] removed rolling latest/ (superseded by final/)")

    elapsed = time.time() - t_start
    print(f"\n---\ntraining_seconds: {elapsed:.0f}\nnum_steps: {global_step}\noutput_dir: {args.output_dir}")
    log.info("Done.")
