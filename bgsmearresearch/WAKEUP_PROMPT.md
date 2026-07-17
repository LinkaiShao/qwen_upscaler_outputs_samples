# BG-Smear Wake-Up Prompt

Paste this to Codex every 2 hours:

```text
Read vtonautoresearch/bgsmearresearch/CODEX_AUTOMATION.md and follow it.

Review the newest run in vtonautoresearch/bgsmearresearch.
Check the latest design, metrics, README/RESULTS, and images.
Tell me whether the bg smear/darkening is gone, improved, unchanged, or worse.

Then update vtonautoresearch/bgsmearresearch/RELAY.md for Claude:
- summarize the newest run verdict
- state the exact next experiment Claude should run
- state what Claude must not do
- require the same fixed 5-ID raw predict_sample CLOUD eval and image panels
- enforce TRAINING_RULES.md: true bifurcation, no soup, lower-level priority, deployed-quality metrics, keep the loss stack simple

Do not run GPU inference unless I explicitly ask.
```

