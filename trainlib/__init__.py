"""VTON training package (split from the old monolithic train.py).

Module map:
    state.py      shared mutable runtime globals (the model<->loss bus) — access as state.X
    config.py     Args dataclass
    constants.py  module-level config computed at import (SLOT_ORDER, prompt, id lists)
    data.py       latent pack/unpack, VAE decode, VTONDataset, collate
    models.py     pure nn.Module classes (no shared globals)
    builders.py   factories / forward-hooks / attention processors (use state.X)
    forward.py    train_step — forward pass + loss assembly
    run.py        main() — training loop, checkpointing, validation

Entry point is ../train.py -> trainlib.run.main().
"""
