# BC01_state52_reader — branch-credit, state-enhancer @ block 52

**Main serious candidate.** State-aware enhancer `Delta = CrossAttn(Q=H_52, K=V=gnet(clean aligned garment))`,
mask-gated, injected after block 52 → blocks 53–59 are the downstream "reader" the main net must learn.

**The point vs RunA/v2 (both inert):** the new **branch-credit loss** makes `delta=0` no longer free.
Each step runs an ON pass (adapter on, grad) and a matched OFF pass (adapter bypassed, same C_t/σ/starvation,
no grad); the adapter is credited only when the ON pass beats OFF in the garment region:
`L = gt_loss(ON) + W_credit·relu(gar_ON − gar_OFF + margin) + W_keep·keep(ON,OFF outside garment) + W_delta·|delta|²`.

**Trainable:** garment net + cross-attn bridge + **run37 LoRA blocks 52–59 only** (0–51 FROZEN via LR_EARLY_MULT=0).
run37 LoRA is NOT masked; full LoRA is NOT trained. Wrong garment is EVAL-ONLY (never in the training loss).

**BC01 knobs:** block 52, no quarantine, normal σ (BC03 adds quarantine; BC12 adds high-σ starvation).
W_credit=1.0, W_keep=1.0, W_delta=0.01, margin=0.02, starve_frac=0.60. 1h 5-ID overfit.

**Watch during training:** `bc_on` should fall BELOW `bc_off` (enhancer helping); `bc_cred`→0; `bc_keep` small.

**Acceptance (all required, panel mandatory):** correct ON beats branch-off in garment AND correct ON beats
wrong ON AND bg/skin no degrade vs branch-off AND panel visibly improves garment detail. Also run the
STARVED eval (standard slot zeroed, enhancer sole source): correct-starved ≪ wrong-starved = carries identity.

**Verdict:** _pending run._
