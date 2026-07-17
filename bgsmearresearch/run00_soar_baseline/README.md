# run00 — soar baseline

**Checkpoint:** `runs/soar_noise_20260619_170822/final`
**What:** the untouched soar model — the baseline every bg-smear experiment warm-starts from and must beat. No bg-band loss.

## Result — raw `predict_sample` CLOUD (higher = worse bg smear/darkening)

| ID | CLOUD | darkening |
|---|---|---|
| 00006 | 7.04 | −6.0 |
| 00008 | 8.10 | −6.4 |
| 00013 | 8.10 | −8.6 |
| 00017 | 7.75 | −7.1 |
| 00034 | 9.98 | −10.2 |
| **mean** | **8.19** | **−7.7** |

The defect: generated bg is **darker than real** by ~7–10 luma near the silhouette, plus green/pink speckle (chroma). Target is if5's ~3.4.

## Images (`images/`)

`<ID>_final_vs_gt.png` — left = FINAL deployed image (what ships), right = real GT. The `raw-CLOUD` number (measured on the raw pre-paste generation, the honest signal) + darkening are burned in.
