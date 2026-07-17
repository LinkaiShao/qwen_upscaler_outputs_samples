# Overnight 2-Hour Garment-Net Plans — 2026-07-10

This is an overnight research queue for Claude. The goal is not to win by one lucky 5-image overfit. The goal is to find a training/setup recipe that can later justify a long run from the proper base. Each experiment is a 2-hour probe. If a 2-hour probe cannot show a real signal on the fixed 5 IDs, do not run it long.

## Global Rules For Every 2-Hour Plan

- Base: start from `runs/BGBAND37_012549/final` unless the user explicitly overrides.
- Train cap: `TIME_BUDGET=7200`. Do not exceed 2 hours per probe.
- Fixed overfit IDs: `00006_00,00008_00,00013_00,00017_00,00034_00`.
- Main denoiser frozen: `FREEZE_LORA=1`, `FREEZE_V6=1`. Train the garment net / garment bridge only.
- Garment net must use real copied Qwen blocks. Do not substitute a small U-Net, conv refiner, or output-space-only refiner.
- The garment net's purpose is detail/identity assistance: stripes, logos, texture, collar/cuffs/hem, pattern, color transitions. It must not become a second full try-on generator.
- The garment branch must be bypassable. Every run must produce branch OFF and branch ON predictions.
- Every run must test correct vs wrong vs zero vs shuffled garment. If `wrong ~= correct`, the run is invalid.
- Every run must save raw/final panels and inspect them by eye before calling anything a win.
- Every run must write a short `README.md` / `RESULTS.md` in its `bgsmearresearch/runNN_*` folder.
- After training finishes, immediately evaluate and decide. Do not sleep after the training signal returns.

## Required Metrics

For every run, report:

- branch OFF baseline metrics
- branch ON metrics
- correct garment vs wrong garment vs zero garment vs shuffled garment
- `CLOUD`, `whole_bg_dL`, `far_bg_dL`, `garL1`, `skinL1`, PSNR, SSIM
- visual verdict from actual panels

Acceptance requires all of:

- garment improves visibly and by metric versus branch OFF
- correct garment beats wrong/zero/shuffled
- bg does not materially regress versus branch OFF/run37
- skin does not materially regress versus branch OFF/run37

Hard reject:

- `wrong ~= correct`
- bg gets darker/cloudier while garment only slightly improves
- branch produces generic garment-region cleanup
- branch cannot be bypassed
- evaluation path does not include the trained garment branch

## Run 63 — OOTD Qwen-Block K/V Branch Feasibility

Hypothesis: the cleanest first real garment net is the existing OOTD-style path: copied Qwen garment branch + dedicated trainable K/V injectors into the frozen main attention. This is closer to "garment memory" than residual hidden injection.

Use:

```bash
RUN_NAME=BGBAND63_ootd_bridge_OVERFIT5
USE_GARMENT_OOTD=1
GARMENT_OOTD_LAYERS=2
GARMENT_OOTD_COPY_BLOCKS="0,1"
GARMENT_OOTD_INJECT_BLOCKS="44,50,55"
GARMENT_OOTD_DEPTH_SPECIFIC=1
GARMENT_OOTD_GATE_SOURCE=warped_mask
GARMENT_OOTD_DROPOUT=0.10
GARMENT_OOTD_VG_ZERO_INIT=1
GARMENT_OOTD_GATE_INIT_LOGIT=0.0
GARMENT_OOTD_LR_BLOCKS=1e-6
GARMENT_OOTD_LR_AUX=3e-5
GARMENT_OOTD_LR_INJ=3e-5
GARMENT_OOTD_LR_GATE=1e-3
USE_GARMENT_BRIDGE=1
BRIDGE_ONLY=1
W_BRIDGE_GAR=1.0
W_BRIDGE_GAR_EDGE=1.0
W_BRIDGE_BG=2.0
W_BRIDGE_SKIN=2.0
W_BRIDGE_FAR=5.0
W_BRIDGE_BND=5.0
W_BRIDGE_SENS=10.0
W_BRIDGE_ZERO=10.0
W_BRIDGE_ROLLOUT=0.0
BRIDGE_ROLLOUT_K=0
USE_ATTN_ENTROPY=0
USE_GARMENT_KV=0
USE_LATE_XATTN=0
USE_GARMENT_XATTN=0
FREEZE_LORA=1
FREEZE_V6=1
USE_SOAR=0
SOAR_PROB=0
TIME_BUDGET=7200
```

Before training, smoke-test that the bridge can produce branch OFF, branch ON, and wrong-garment forwards. If not, fix plumbing first.

Pass condition: correct garment is better than wrong/zero/shuffled and bg/skin hold.

Fail lesson:

- If `wrong ~= correct`, OOTD branch is still generic.
- If garment improves but bg worsens, injection is too early/global or preservation is too weak.
- If nothing moves, the branch/interface is too weak or gate/projection is not waking.

## Run 64 — OOTD Injection Timing Probe

Hypothesis: the garment signal may need a different amount of downstream main-model processing. Injection too late may be ignored; injection too early may leak into bg/skin.

Use the exact run63 setup, changing only injection blocks.

If run63 had no garment signal:

```bash
RUN_NAME=BGBAND64_ootd_mid_sites_OVERFIT5
GARMENT_OOTD_INJECT_BLOCKS="36,44,52"
```

If run63 had garment signal but bg/skin leakage:

```bash
RUN_NAME=BGBAND64_ootd_late_sites_OVERFIT5
GARMENT_OOTD_INJECT_BLOCKS="50,55,59"
```

Do not change losses, LR, branch depth, entropy, or masks. This is one lever only.

Pass condition: same as run63.

Fail lesson: if every timing either ignores garment or damages bg/skin, the branch needs better garment representation before injection, not more site tuning.

## Run 65 — Pose/Layout-Aware Garment Memory

Hypothesis: the branch is generic because it receives flat garment memory with no target-body layout. The main frozen model may know where the garment goes, but the garment branch cannot provide target-space details unless it receives pose/layout context.

Implement one architectural change:

- Feed the Qwen garment branch deploy-available layout context in addition to garment latent.
- Minimum useful input: `garment_latent + warped_mask`.
- Better input if already available cleanly: `garment_latent + pose/densepose_latent + warped_mask + agnostic/edit_mask`.
- Keep copied Qwen blocks. Only change the branch input projection or add a small linear bridge before the copied Qwen blocks.
- Do not change the frozen main denoiser.

Use the run63 bridge/eval protocol. No entropy. No agnostic-hole corruption in this run.

Pass condition: correct garment begins beating wrong/zero/shuffled. This run is mainly about identity specificity, not max garment L1.

Fail lesson: if pose/layout-aware memory still has `wrong ~= correct`, the frozen main is not reading garment identity through this interface.

## Run 66 — Target-Space Garment Detail Memory

Hypothesis: the branch needs target-space garment hypotheses, not just flat garment plus pose. It should carry detail into the region where the frozen denoiser will generate.

Implement one architectural change:

- Feed the branch a target-space garment/detail hypothesis, for example `rough_latent * warped_mask` or another deploy-available rough aligned garment signal, plus raw `garment_latent` and `warped_mask`.
- Keep copied Qwen blocks.
- Do not let the branch see GT target image.
- Do not use the final output-space refiner as the answer.

Use run63 bridge/eval protocol.

Pass condition: correct garment beats wrong/zero/shuffled and garment details improve without bg/skin regression.

Fail lesson: if this works better than run65, the missing piece was target-space garment detail alignment. If not, the interface is likely the bottleneck.

## Run 67 — Detail-Specialist / High-Pass Branch

Hypothesis: the branch keeps learning generic cleanup because it carries low-frequency studio/color priors instead of garment identity details.

Implement one change:

- Add a high-pass/detail projection into the Qwen garment branch input.
- Keep raw garment available. Do not use high-pass-only, because that can destroy color identity.
- Suggested input: `raw garment latent + high-pass garment latent + warped_mask`.
- The high-pass/detail stream must only supervise garment identity/detail. It must not add bg/skin losses or fixed bg-band losses.

Use run63 bridge/eval protocol.

Pass condition: garment detail improves and correct/wrong separates.

Fail lesson: if high-pass wakes the branch but correct/wrong remains equal, it is still generic sharpening, not identity conditioning.

## Run 68 — Prior-Starvation Probe

Hypothesis: the frozen denoiser ignores the garment branch because agnostic/rough/slot priors are enough for generic shirt reconstruction. Starve that shortcut during training.

Use the best setup from run63-run67 and add only this training-time condition:

- On 20-30% of batches, replace the agnostic edit hole with standard normal noise.
- Apply the exact same corruption to branch ON, branch OFF, wrong, zero, and shuffled forwards in the bridge step.
- Evaluate on the normal uncorrupted canonical inference.

Do not combine with a new architecture change. Do not add entropy. Do not change injection blocks at the same time.

Pass condition: correct garment becomes meaningfully better than wrong/zero/shuffled while bg/skin still hold.

Fail lesson: if starving agnostic priors still does not make the branch identity-specific, the frozen main in-loop branch is probably not a viable long-run path in this form.

## If A Run Works

If any run passes, do not immediately launch a long run. First run one 2-hour confirmation:

- same architecture
- same fixed IDs
- different seed if easy
- slightly tighter preservation only:

```bash
W_BRIDGE_BG=5.0
W_BRIDGE_SKIN=5.0
W_BRIDGE_FAR=10.0
W_BRIDGE_BND=10.0
```

If it confirms, write the long-run recipe clearly: architecture, init, trainable params, losses, inference path, evaluation path, and why it should scale.

## If Nothing Works

If runs 63-68 all fail identity specificity, stop the in-loop frozen-main garment-net line for the night. Write the negative finding plainly:

- real Qwen-block garment branches were trainable
- branch could or could not affect generation
- wrong/correct did or did not separate
- bg/skin did or did not hold
- which part is the bottleneck

Do not invent run69 as another small knob change.

