
## Result → ✅ PLUMBING PROVEN, ⚠️ composition quality wrong

**Goal (prove trained v6 heads can affect raw prediction under the eval harness): ACHIEVED.**
Wired `inference_v6wired.py` (`USE_V6_HEADS_DEPLOY=1`): loads `v6_heads.pt`, hooks `norm_out`, applies `to_b/to_s/to_route`, composes `final = route[gar]·C_lat + route[skin]·(al+δ_s) + route[bg]·(al+δ_b) + route[keep]·al` in the RAW_FULL_PRED branch. Debug: `|final−C|=0.30–0.40` every ID (was 0), route frac gar/skin/bg/keep=[0.15,0.04,0.81,0.0], |δ_b|=0.25, |δ_s|=0.33. The heads are now deployable.

**Metric:** raw CLOUD **59.1** (soar 8.19, run03 4.84) — much WORSE. Two causes:
1. **My composition over-applies** — route assigns 81% of the WHOLE frame to bg and replaces it with `al+δ_b`, including far bg that should be `keep` (real). It must be gated to the **edit zone** (`M_edit` from warped_mask), like the original repair composition: `M_g·C_lat + (M_edit−M_g)·(route-blended al+δ) + (1−M_edit)·al`.
2. The heads were trained as **auxiliary** L1 predictors of `(person−agnostic)`, never against this deploy composition — so `al+δ_b` at inference (different noise fill than training) doesn't reconstruct.

**Decision (per RELAY):** wired heads DO change raw bg but quality is wrong → (a) add the edit-zone gate to the composition and re-smoke, then (b) run a short 5-ID overfit with LoRA frozen and this authoritative V6 route deployed **in the training eval**, so the heads learn to compose. Do NOT return to LoRA-only yet — the path is proven, the composition just needs the gate + matched training.

**Artifacts:** `inference_v6wired.py` (the wired deploy inference), `/tmp/raw_deploy_v6.py` (harness), panels in `images/`.

## Re-smoke — GATED composition (edit-zone only)

Composition fixed to: `final = C_lat·M_g + v6_blend·M_repair + al·(1−M_edit)`, `M_edit`=dilate(warped,V6_R_OUT), `M_g`=warped, `M_repair`=M_edit−M_g; v6_blend = route-renormalized skin/bg (al+δ) inside M_repair only.

Debug (route% INSIDE edit): gar/skin/bg/keep=[0.34,0.06,0.60,0.0]; M_edit_frac=0.42, M_repair_frac=0.21; outside-edit is `al` (route ignored there — verified). `|final−C|`=0.29.

**Result:** the composition is now **structurally correct & non-destructive** — garment (C_lat) and outside-edit (al) are preserved; V6 affects ONLY the repair ring. ✅ Answers the smoke question: yes, gated V6 affects the prediction without destroying the rest of the frame.

**But CLOUD still ~58.8 and panels show CONFETTI in the repair ring.** Root cause: run06's v6 heads are barely trained (2h, zero-init, in-loop [DEPLOY] flat) → in the repair ring `al` is masked/noise-agnostic and the near-random δ doesn't reconstruct → garbage. This is a **head-training** problem, not a composition problem.

**Tension with the rule:** "no overfit until gated smoke gives sane images." The frame IS sane (preserved); only the repair ring is bad, and that ring can only become sane once the heads are trained *with this exact composition in the loop* (the overfit). So run06's undertrained heads cannot produce a sane ring in a pure smoke — it's chicken-and-egg. Composition is proven; heads need training. Awaiting user call: proceed to the 5-ID overfit (train heads with deploy composition) given the frame is non-destructive, or first relax to a smaller/better head init.
