# learned_detail_slot_zeroinit — RUN4B (patched detail_slot)

**What changed.** The earlier detail_slot was **non-zero-init** → its random residual perturbed the
garment slot at step 0, destroying the garment (garL1 ≈ 36). It is now **ZERO-INIT**
(`nn.init.zeros_` on `self.slot` weight+bias), so **step0 == run37 exactly** and the residual is
learned from zero.

**Arch.** `GARMENT_ADAPTER_MODE=detail_slot`: a learned zero-init residual added to the STANDARD
garment conditioning slot (64-d tokens). If the standard slot is starved
(`GARMENT_SLOT_CORRUPT=zero`), this residual becomes the only garment identity path. Co-train with
LoRA (`FREEZE_LORA=0`).

**Files (self-contained snapshot).**
- `model.py`  = root `garment_adapter.py` (detail_slot head, zero-init).
- `run.sh`    = root `run_learned_detail_slot.sh` (launcher; RESDIR → this folder, RUN_NAME RUN4B_DETAILSLOT).
- `WIRING.py` = verbatim forward.py detail_slot block + the zero-init.

**Verified (CPU unit test, no GPU).** `GarmentAdapter("detail_slot").project(...)` → `max|out| =
0.000e+00` at init → step0 v_adapter ≈ 0 → step0 == run37. Runtime smoke deferred (GPU0 owned by
Run5). Ready for the coordinator to launch.
