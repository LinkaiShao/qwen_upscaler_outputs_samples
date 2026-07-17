"""WIRING.py — VERBATIM snapshot of the live trainlib hook for the detail_slot (RUN4B, ZERO-INIT).
Guard flag: USE_GARMENT_ADAPTER=1 GARMENT_ADAPTER_MODE=detail_slot.

──────────────────────────────────────────────────────────────────────────────
1) garment_adapter.py — detail_slot head is ZERO-INIT (step0 residual == 0 == run37):
──────────────────────────────────────────────────────────────────────────────
    elif mode == "detail_slot":
        self.slot = nn.Linear(in_ch * 4, pack_dim)      # learned residual detail slot tokens (64-d)
        nn.init.zeros_(self.slot.weight); nn.init.zeros_(self.slot.bias)   # <-- ZERO-INIT

    def project(self, garment_in):
        garment_in = garment_in.to(self.proj.weight.dtype if hasattr(self, "proj") else self.slot.weight.dtype)
        B, _, H, W = garment_in.shape
        x = garment_in.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(B, (H // 2) * (W // 2), self.in_ch * 4)
        head = self.proj if hasattr(self, "proj") else self.slot
        return head(x)      # zeros at init -> 0

──────────────────────────────────────────────────────────────────────────────
2) trainlib/forward.py — add the zero-init residual to the STANDARD garment slot
   (gated by _active(); if the std slot is starved this is the only garment identity path):
──────────────────────────────────────────────────────────────────────────────
    if (int(os.environ.get("USE_GARMENT_ADAPTER", "0")) and os.environ.get("GARMENT_ADAPTER_MODE", "") == "detail_slot"
            and getattr(state, "_GARMENT_ADAPTER", None) is not None):
        import garment_adapter as _ga3
        if _ga3._active() and state._ADAPTER_HOLDER.get("adapter_in") is not None:
            slot_tensors["garment"] = gar_p + state._GARMENT_ADAPTER.project(
                state._ADAPTER_HOLDER["adapter_in"].to(torch.float32)).to(gar_p.dtype)

CPU unit-test proof: GarmentAdapter(mode="detail_slot").project(randn(2,20,128,96)) ->
    out shape (2,3072,64), max|out| = 0.000e+00  =>  step0 == run37.
"""
