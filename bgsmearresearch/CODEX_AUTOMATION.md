# Codex Automation Instructions — BG-Smear Research

This file is for Codex when the user's macro wakes this thread every 2 hours.
Follow it exactly. Do not start GPU work unless the user explicitly asks.

## Purpose

Every wake-up, inspect the newest bg-smear research run, decide whether it improved the background darkening problem, and update `vtonautoresearch/bgsmearresearch/RELAY.md` with the next instruction for Claude.

The goal is to reduce raw generated background darkening/smear around the person silhouette.

## Hard Rules

The training rules are mandatory. Read `/tmp/relay/TRAINING_RULES.md` if available before giving new instructions.

- True bifurcation: region objectives must use detach-normalized percentage shares, not fixed-weight patch losses.
- No soup: every active term for a region must pull toward the same target, real GT.
- Lower-level priority: garment/skin targets must not be overwritten by rollout/global objectives.
- Deployed-quality metric: judge by fixed, seeded, full from-noise inference metrics that match visual quality.
- Keep it simple: prefer one interpretable change per experiment. Do not add many losses.

Forbidden unless the user explicitly overrides:

- Do not recommend new fixed-weight `LAMBDA_BGBAND_*` losses.
- Do not score success from grey-hole pasted/deployed images alone; paste hides the raw defect.
- Do not accept a run as improved unless actual panel images look better.
- Do not instruct Claude to change multiple independent levers at once.

## Files To Inspect

On every wake-up:

1. Find newest run folder:

   ```bash
   find vtonautoresearch/bgsmearresearch -maxdepth 1 -type d -name 'run[0-9][0-9]_*' -printf '%T@ %p\n' | sort -nr | head
   ```

2. Read:

   - `vtonautoresearch/bgsmearresearch/README.md`
   - newest run `README.md`
   - newest run `RESULTS.md` if present
   - `vtonautoresearch/bgsmearresearch/RELAY.md`

3. Inspect images in newest run:

   - `images/*pred_vs_gt*.png`
   - `images/*final_vs_gt*.png`
   - any summary panel

Use `view_image` for representative IDs, especially `00006`, `00013`, and `00034`.

## Metrics

Primary metric:

- raw `predict_sample` `CLOUD` on fixed five IDs: `00006`, `00008`, `00013`, `00017`, `00034`

Secondary metrics:

- signed darkening, negative is darker than GT
- chroma/speckle if reported
- garment/skin regression if reported

Baselines:

- `soar_noise_20260619_170822`: mean raw `CLOUD = 8.19`
- `if5` target: mean raw `CLOUD ~= 3.4`
- clean GT floor: near `0`

A run is a valid improvement only if:

- mean raw `CLOUD < 8.19`
- signed darkening is less negative than soar or at least not worse
- panels visibly reduce the bg halo/cloud
- garment/skin quality does not visibly regress
- the experiment follows the training rules

If metrics improve but panels look worse, do not accept it.

## Decision Procedure

If newest run is still pending:

- Do not invent a result.
- Update `RELAY.md` only if Claude needs a reminder to complete eval and write README/images.

If newest run regressed:

- Tell Claude not to continue the same lever blindly.
- Preserve the lesson in `RELAY.md`.
- Recommend the smallest rule-compliant next A/B.

If newest run improved:

- Mark it as current best valid bg-smear run.
- Recommend one adjacent follow-up, usually a small share change or a confirmation run.

If newest run violates rules:

- Classify as invalid, even if metrics improve.
- Instruct Claude to rerun a rule-compliant version.

## Current Plan Bias

Current safe plan:

1. `run02`: bg share `0.35`, rule-compliant bifurcated flow + region-%.
2. If `run02` improves and garment is stable, try bg share `0.40`.
3. If `run02` regresses or does nothing, do not add fixed band losses. Consider either:
   - reducing bg share and restoring known-good if5 mechanisms, or
   - an explicit rule-compliant rollout/bias correction that remains one coherent bg-to-GT objective.

Prefer `PCT_FLOW_BG` and `RPCT_BG` to move together. If README says bg share changed but launcher leaves `PCT_FLOW_BG=0.25`, flag it as a setup mismatch.

## How To Update RELAY.md

Edit only the Claude-facing file:

`vtonautoresearch/bgsmearresearch/RELAY.md`

Use `apply_patch`.

RELAY update must include:

- latest run reviewed
- verdict: improved / regressed / pending / invalid
- exact next experiment for Claude
- exact forbidden changes
- eval requirement

Keep it short. Claude should be able to act without re-asking.

Suggested INBOX format:

```markdown
## INBOX (user writes here)

- Codex review YYYY-MM-DD HH:MM: Reviewed `runNN_name`. Verdict: ...
  Next: ...
  Do not: ...
  Eval: ...
```

If there is already an unprocessed INBOX item, prepend the new Codex item unless it conflicts. If it conflicts with a user-written item, do not overwrite the user item; note the conflict in the final response.

## Final Response To User

After updating RELAY.md, answer with:

- newest run reviewed
- metric verdict
- visual verdict
- what was written to `RELAY.md`

Keep it concise.
