# Design: Per-Genome Gap/Overlap Penalty Calibration for the `--model` Path

**Date:** 2026-06-23  
**Status:** Approved  
**Scope:** PHANOTATE-rs `--model` (ONNX) path only

---

## 1. Problem

The default heuristic path and the ONNX `--model` path produce ORF edge rewards with very different magnitude distributions. The current code uses a single `PHANOTATE_MODEL_PENALTY_SCALE` multiplier to align gap/overlap penalties with those rewards, but that multiplier is constant across genomes. This design removes that environment variable and replaces it with per-genome, auto-calibrated `gap_scale` and `overlap_scale` values.

## 2. Goal

Make gap and overlap edge weights on the `--model` path proportional to the per-genome distribution of ORF edge rewards, so that:

- Weak-scoring genomes are not dominated by fixed penalties.
- Strong-scoring genomes still pay enough penalty to avoid spurious merges/overlaps.

## 3. Proposed Solution

Replace the single `penalty_scale` passed into `Graph::from_orfs` with two separate scales: `gap_scale` and `overlap_scale`. For the default (non-model) path both scales remain `1.0`. For the `--model` path the scales are computed per genome from the scored ORFs.

### 3.1 Graph construction API change

```rust
pub fn from_orfs(
    orfs: &[Orf],
    contig_length: usize,
    pgap: f64,
    gap_scale: f64,
    overlap_scale: f64,
) -> (Self, Vec<usize>)
```

`score_gap` and `score_overlap` in `src/weights.rs` each receive only their own scale factor.

### 3.2 Per-genome scale computation

After the ONNX model scores all ORFs and before graph construction, `process_genome` computes:

1. `med_orf`: median of `|orf.weight|` over all candidate ORFs.
2. `med_gap_raw`: median of `score_gap(length, dir, pgap, 1.0)` over a representative grid:
   - lengths: `0, 10, 30, 100, 300`
   - directions: `"same"`, `"diff"`
3. `med_overlap_raw`: median of `score_overlap(length, dir, pstop_avg, 1.0)` over a representative grid:
   - lengths: `1, 5, 10, 20, 50`
   - directions: `"same"`, `"diff"`
   - `pstop_avg` is the mean `pstop` of all ORFs, falling back to `pgap` if no ORFs.

Then:

```rust
gap_scale      = TARGET_GAP_RATIO      * med_orf / med_gap_raw;
overlap_scale  = TARGET_OVERLAP_RATIO  * med_orf / med_overlap_raw;
```

Both scales are clamped to `[0.1, 10.0]`.

### 3.3 Learned global target ratios

`TARGET_GAP_RATIO` and `TARGET_OVERLAP_RATIO` are global constants hard-coded in `src/main.rs`. They are learned once by running a grid search over annotated genomes.

The search is performed by a new script `scripts/tune_penalty_balance.py` that:

- Iterates over a 2-D grid of `(TARGET_GAP_RATIO, TARGET_OVERLAP_RATIO)` on `tests/golden/annotgenomes`.
- Runs `target/release/phanotate-rs --model tests/golden/orf_model.onnx --detect-table --yes -f sco`.
- Computes aggregated F1 against GenBank reference CDS coordinates.
- Reports the best pair and the F1 improvement over the current default (`gap_scale = overlap_scale = 1.0`).

### 3.4 Fallbacks

- If there are fewer than 3 ORFs, use `gap_scale = overlap_scale = 1.0`.
- If any representative raw penalty is `0.0` or non-finite, use `1.0` for that scale.
- On the default heuristic path, always use `1.0` for both scales.

## 4. Error Handling

- All scale computations use finite checks (`is_finite()`).
- Clamping prevents extreme scales from producing pathological graphs.
- The existing graph algorithms and output formatting are unchanged.

## 5. Testing Plan

1. Unit tests in `src/weights.rs` continue to pass with scale factors of `1.0`.
2. Run `scripts/tune_penalty_balance.py` and confirm the learned constants improve aggregated F1 on the first 50 annotated genomes.
3. Run `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test --lib`, and `cargo test --test cli_tests`.
4. Spot-check that the default (non-model) path is bit-identical to before.

## 6. Files Affected

- `src/weights.rs` — split `penalty_scale` into `gap_scale` / `overlap_scale`.
- `src/graph.rs` — update `from_orfs` signature and all calls to `score_gap`/`score_overlap`.
- `src/main.rs` — compute per-genome scales for the `--model` path; add learned target-ratio constants.
- `src/weights.rs` tests — update call sites.
- `scripts/tune_penalty_balance.py` — new tuning script.

## 7. Open Questions / Future Work

- Should the representative grids be expanded to more lengths/directions? Start with the grid above; expand if F1 is noisy.
- Could the target ratios be table-specific (e.g., different for table 4)? Defer until per-table gains are demonstrated.
