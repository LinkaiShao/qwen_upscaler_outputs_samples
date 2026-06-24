"""Args dataclass (split from train.py)."""
import os, sys
from dataclasses import dataclass, field
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing")

BASE = "/home/link/Desktop/Code/fashion gen testing"
LOCAL_CACHE = f"{BASE}/vtonautoresearch/local_cache"


@dataclass
class Args:
    pretrained_model: str = f"{BASE}/Qwen-Image-Edit-2511"
    vitonhd_dir: str = f"{BASE}/VITON-HD-dataset"
    latent_cache_dir: str = f"{BASE}/my_vton_cache/latents"
    text_cache_dir: str = f"{BASE}/my_vton_cache/text"
    output_dir: str = f"{BASE}/vtonautoresearch/runs"
    rank: int = int(os.environ.get("LORA_RANK", "32"))
    alpha: int = int(os.environ.get("LORA_ALPHA", "64"))
    init_lora_weights: str = "gaussian"
    lora_targets: list = field(default_factory=lambda: os.environ.get(
        "LORA_TARGETS", "to_k,to_q,to_v,to_out.0").split(","))
    inject_blocks: list = field(default_factory=lambda: list(range(60)))
    sigma_beta_alpha: float = 1.0
    sigma_beta_beta: float = 1.0
    lr: float = float(os.environ.get("LR", "3e-5"))
    projector_lr: float = 2e-3
    batch_size: int = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum: int = int(os.environ.get("GRAD_ACCUM", "1"))   # env-driven; effective batch = batch_size*grad_accum
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    gradient_checkpointing: bool = True
    device_transformer: str = "cuda:0"
    logging_steps: int = 10
    seed: int = 42
    train_split: str = "test"
