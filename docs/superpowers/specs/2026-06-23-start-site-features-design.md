# Design: Relative Start-Site Features for the ONNX Model Path

**Date:** 2026-06-23  
**Status:** Approved  
**Scope:** PHANOTATE-rs `--model` (ONNX) path only

---

## 1. Problem

The current 29-feature ONNX model scores each ORF in isolation. It includes absolute start-site signals (`upstream_pwm_score`, `rbs_spacer`, `sd_rbs_score`), but it never sees the **alternative in-frame start codons** that share the same stop. Consequently, the model cannot learn that a weak-looking start is correct when every alternative upstream start is even weaker.

## 2. Goal

Improve start-site selection on the `--model` path by adding relative start-site features that compare the chosen start to the best alternative start for the same stop. Retrain the ONNX model and verify an F1 improvement over the current `--model` baseline of 0.6717 on the first 50 annotated genomes.

## 3. Proposed Solution

### 3.1 Train a PWM from annotated starts

- Input: GenBank reference genes in `tests/golden/annotgenomes/*.gb`.
- For each annotated start codon, extract the 20-bp upstream window.
- Build a 4 × 20 position weight matrix with pseudocount smoothing.
- Save the PWM (and optionally the spacer distribution) so both the training script and `src/ml_features.rs` can use the same matrix.

### 3.2 New ORF features

Add the following features to `src/ml_features.rs` (the existing `upstream_pwm_score` already plays the role of `chosen_pwm_score`):

| Feature | Description |
|---------|-------------|
| `best_alt_pwm_score` | Highest PWM score among alternative in-frame starts sharing the same stop. |
| `pwm_ratio` | `upstream_pwm_score / max(best_alt_pwm_score, eps)`. |
| `start_rank` | Rank of the chosen start by PWM among alternatives (1 = best). |
| `num_alt_starts` | Number of alternative in-frame starts considered. |
| `start_codon_log_freq` | Log frequency of the start codon among high-confidence ORFs in this genome. |

`NUM_FEATURES` increases from 29 to 34.

### 3.3 Training pipeline updates

- Update `scripts/train_orf_score_model.py` to expect 34 features.
- Regenerate training data from annotated genomes so the new features are populated and labels use the annotated start coordinates.
- Retrain the XGBoost model.
- Export to `tests/golden/orf_model.onnx`.

### 3.4 Runtime integration

- `src/ml_features.rs` computes the new features for every candidate ORF using the saved PWM and alternative-start enumeration.
- `src/onnx_scorer.rs` already accepts the feature vector length from the ONNX input shape; verify it matches 35.
- No changes to the graph/path algorithm are required.

## 4. Fallbacks

- If no annotated starts are available for a translation table, `start_codon_log_freq` defaults to 0.0 for all codons.
- If a stop has no alternative starts, `best_alt_pwm_score` = 0.0, `pwm_ratio` = 1.0, `start_rank` = 1.
- If the saved PWM file is missing, fall back to a uniform PWM (all scores 0.0).

## 5. Testing Plan

1. Train PWM on annotated genomes and visually inspect top motifs.
2. Regenerate training TSV and confirm the new columns are populated.
3. Retrain model, export ONNX, and run `cargo test --lib`.
4. Benchmark `--model` on the first 50 annotated genomes; target F1 > 0.6717.
5. Run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test --test cli_tests`.

## 6. Files Affected

- `src/ml_features.rs` — add new features and PWM loading.
- `src/lib.rs` — expose PWM helper if needed.
- `scripts/train_orf_score_model.py` — update feature count and training data generation.
- `scripts/build_start_pwm.py` (new) — train PWM from annotated genomes.
- `tests/golden/orf_model.onnx` — retrained model.
- `tests/golden/start_pwm.json` (new) — saved PWM.

## 7. Open Questions / Future Work

- Should the PWM be per translation table or global? Start with global; split per table if table-4 genomes show different motifs.
- Should we also add a separate start-site classifier head? Defer unless relative features alone are insufficient.
