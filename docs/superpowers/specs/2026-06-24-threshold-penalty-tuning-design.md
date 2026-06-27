# Design: Per-Genome Threshold Calibration + Penalty Retuning for the `--model` Path

**Date:** 2026-06-24  
**Status:** Approved  
**Scope:** PHANOTATE-rs `--model` (ONNX) path only

---

## 1. Problem

The new 34-feature model improves F1 to ~0.70, but prodigal-gv `-p meta` reaches ~0.82 on the same genomes. Our model has lower precision (0.68) and recall (0.73) than prodigal-gv's balanced ~0.82/0.82. Two issues contribute:

1. **Gap/overlap penalties** were tuned for the old 29-feature model; the new model's score distribution is different.
2. **Fixed decision threshold**: `--model-threshold` is constant across genomes, so large genomes accumulate false positives and small genomes may lose true positives.

## 2. Goal

Improve aggregated F1 on the first 50 annotated genomes above the current 0.7025 by:
1. Re-tuning gap/overlap target ratios for the new 34-feature model.
2. Adding per-genome `--auto-threshold` calibration.

## 3. Proposed Solution

### 3.1 Re-tune gap/overlap penalties

- Run `scripts/tune_penalty_balance.py` with the new `tests/golden/orf_model.onnx`.
- Search a finer grid around the current defaults (`gap_target = 0.5`, `overlap_target = 0.3`).
- Update `MODEL_GAP_TARGET_RATIO` and `MODEL_OVERLAP_TARGET_RATIO` in `src/penalty_calibration.rs` with the winners.

### 3.2 Add `--auto-threshold` calibration

Add a new CLI flag:

```rust
/// Automatically calibrate the model decision threshold per genome.
#[arg(long = "auto-threshold", value_enum, default_value = "none")]
auto_threshold: AutoThresholdMode,
```

Supported modes:

| Mode | Behavior |
|------|----------|
| `none` | Use the fixed `--model-threshold` (current behavior). |
| `length` | `target_genes = genome_bp / GENES_PER_KB`. Pick a score threshold so roughly `target_genes` ORFs receive a positive model multiplier. |
| `percentile` | Use a fixed percentile of all candidate ORF model scores as the threshold. |

The threshold is applied as a **floor**: `effective_threshold = max(per_genome_threshold, cli.model_threshold)`.

### 3.3 Compute per-genome threshold

After the ONNX model scores every ORF and before graph construction:

1. Collect raw model scores for all ORFs.
2. Sort scores descending.
3. For `length` mode, pick the score at index `min(target_genes, scores.len() - 1)`.
4. For `percentile` mode, pick the score at the chosen percentile.
5. Clamp the threshold to a sane range to avoid degenerate genomes.

Pass `effective_threshold` to `Orf::score` instead of the raw CLI threshold.

### 3.4 Learn the global constants

Create `scripts/tune_threshold.py` that grid-searches:
- `GENES_PER_KB`: [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
- Percentile: [70, 75, 80, 85, 90]

Run on the first 50 annotated genomes with `--detect-table --model` and the newly tuned penalties. Report the best mode/constant and F1 gain. Hard-code the winners as defaults in `src/main.rs`.

## 4. Fallbacks

- If an ORF set has fewer than 3 ORFs, fall back to `--model-threshold`.
- If the `length` target exceeds the number of ORFs, clamp to the last score.
- `--model-threshold` always acts as a floor.

## 5. Testing Plan

1. Re-tune penalties and confirm F1 does not regress.
2. Add unit tests for threshold computation helper.
3. Run `scripts/tune_threshold.py` and verify F1 improvement.
4. Run `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test --lib`, `cargo test --test cli_tests`, and `pytest tests/test_python_bindings.py`.
5. Final benchmark vs. prodigal-gv `-p meta`.

## 6. Files Affected

- `src/main.rs` — add `--auto-threshold` flag and calibration logic.
- `src/penalty_calibration.rs` — update learned gap/overlap target ratios.
- `src/orf.rs` — ensure `Orf::score` can accept a per-genome threshold (already accepts `model_threshold`).
- `scripts/tune_threshold.py` (new) — grid-search threshold constants.
- `tests/golden/orf_model.onnx` — unchanged, but all tuning uses it.

## 7. Open Questions / Future Work

- Should `length` mode be adjusted for detected translation table? Defer unless table-specific gains are obvious.
- Could combine length and percentile into a hybrid mode if either alone is unstable.
