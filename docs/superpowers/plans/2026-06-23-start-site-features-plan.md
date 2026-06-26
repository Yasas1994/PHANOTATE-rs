# Relative Start-Site Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add relative start-site PWM features to the ONNX ORF scorer, retrain the model, and verify F1 improvement on the first 50 annotated genomes.

**Architecture:** Extend `Orf` with four relative PWM fields and a per-genome start-codon frequency field. Compute these in `src/ml_features.rs::compute_extra_ml_features`, expose them through `Orf::extract_features`, update the training script to consume the new feature vector, retrain, and replace `tests/golden/orf_model.onnx`.

**Tech Stack:** Rust 2021, Python 3, XGBoost, ONNX.

---

### Task 1: Add new fields to `Orf`

**Files:**
- Modify: `src/orf.rs:6-34`
- Modify: every `Orf { ... }` literal in `src/orf.rs` (~6 places)
- Test: `cargo check`

- [ ] **Step 1: Add fields to the `Orf` struct**

Append to the struct definition in `src/orf.rs`:

```rust
    // Relative start-site features
    pub best_alt_pwm_score: f64,
    pub pwm_ratio: f64,
    pub start_rank: f64,
    pub num_alt_starts: f64,
    pub start_codon_log_freq: f64,
```

- [ ] **Step 2: Initialize the fields in every `Orf { ... }` literal**

Search `src/orf.rs` for all `Orf {` literals and add:

```rust
            best_alt_pwm_score: 0.0,
            pwm_ratio: 1.0,
            start_rank: 1.0,
            num_alt_starts: 1.0,
            start_codon_log_freq: 0.0,
```

There are approximately six literals.

- [ ] **Step 3: Compile**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/orf.rs
git commit -m "feat(orf): add relative start-site feature fields"
```

---

### Task 2: Compute relative start-site features in `ml_features.rs`

**Files:**
- Modify: `src/ml_features.rs` (around the upstream-PWM block and `Orf::extract_features`)
- Test: `cargo test --lib ml_features`

- [ ] **Step 1: Compute high-confidence start-codon frequencies**

After the line `let mean_len = ...` in `compute_extra_ml_features`, add:

```rust
    // Per-genome start-codon frequencies among high-confidence ORFs
    let mut start_counts: std::collections::HashMap<Vec<u8>, usize> = std::collections::HashMap::new();
    let mut total_confident = 0usize;
    for orf in orfs.iter() {
        if orf.seq.len() as f64 >= mean_len * 0.8
            && (orf.sd_rbs_score > 1.0 || orf.non_sd_rbs_score > 1.0)
        {
            *start_counts.entry(orf.start_codon().to_vec()).or_insert(0) += 1;
            total_confident += 1;
        }
    }
```

- [ ] **Step 2: Compute relative PWM features after upstream_pwm_score is set**

Replace the existing upstream-PWM scoring loop block (around lines 423-448) with this extended version:

```rust
    // -----------------------------------------------------------------
    // Upstream PWM score, RBS spacer, and relative start-site features
    // -----------------------------------------------------------------
    for orf in orfs.iter_mut() {
        let pwm_window = if orf.frame > 0 {
            upstream_window_forward(orf.start, dna)
        } else {
            upstream_window_reverse(orf.start, dna.len(), rc_dna)
        };
        orf.upstream_pwm_score = pwm_window.map_or(0.0, |w| {
            score_upstream_window(w, &gene_pwm, &bg_pwm, gene_total, bg_total)
        });

        let rbs_window = if orf.frame > 0 {
            if orf.start > 21 {
                Some(&dna[orf.start - 22..orf.start - 1])
            } else {
                None
            }
        } else {
            let end = dna.len().saturating_sub(orf.start.saturating_add(2));
            if end >= 21 {
                Some(&rc_dna[end - 21..end])
            } else {
                None
            }
        };
        orf.rbs_spacer = rbs_spacer(rbs_window.unwrap_or(&[]), orf.rbs_motif.as_deref());
    }

    // Group ORFs by (stop, frame) and compare chosen start to alternatives
    let mut by_stop_frame: std::collections::HashMap<(usize, i8), Vec<(f64, usize)>> =
        std::collections::HashMap::new();
    for (i, orf) in orfs.iter().enumerate() {
        by_stop_frame
            .entry((orf.stop, orf.frame))
            .or_default()
            .push((orf.upstream_pwm_score, i));
    }
    for group in by_stop_frame.values_mut() {
        group.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }

    let mut best_alt_pwm = vec![0.0f64; orfs.len()];
    let mut rank = vec![1.0f64; orfs.len()];
    let mut alt_count = vec![1.0f64; orfs.len()];
    for group in by_stop_frame.values() {
        let n = group.len();
        for pos in 0..n {
            let (score, idx) = group[pos];
            alt_count[idx] = n as f64;
            rank[idx] = (pos + 1) as f64;
            best_alt_pwm[idx] = if n > 1 {
                if pos == 0 {
                    group[1].0
                } else {
                    group[0].0
                }
            } else {
                0.0
            };
        }
    }

    for (i, orf) in orfs.iter_mut().enumerate() {
        orf.best_alt_pwm_score = best_alt_pwm[i];
        orf.pwm_ratio = if best_alt_pwm[i] != 0.0 {
            orf.upstream_pwm_score / best_alt_pwm[i]
        } else {
            1.0
        };
        orf.start_rank = rank[i];
        orf.num_alt_starts = alt_count[i];

        let codon = orf.start_codon().to_vec();
        let count = start_counts.get(&codon).copied().unwrap_or(0);
        orf.start_codon_log_freq = if total_confident > 0 {
            (count as f64 / total_confident as f64).max(1e-6).ln()
        } else {
            0.0
        };
    }
```

- [ ] **Step 3: Update `Orf::extract_features` to export the new fields**

Change `NUM_FEATURES` to `34` at the top of `src/ml_features.rs`.

Append to `FEATURE_NAMES`:

```rust
    "best_alt_pwm_score",
    "pwm_ratio",
    "start_rank",
    "num_alt_starts",
    "start_codon_log_freq",
```

In `Orf::extract_features`, after the existing `features[28] = heuristic_score` block, add:

```rust
        // 29. Highest PWM score among alternative starts for the same stop
        features[29] = self.best_alt_pwm_score as f32;

        // 30. chosen PWM / best alternative PWM
        features[30] = self.pwm_ratio as f32;

        // 31. Rank of chosen start by PWM among alternatives (1 = best)
        features[31] = self.start_rank as f32;

        // 32. Number of alternative starts considered
        features[32] = self.num_alt_starts as f32;

        // 33. Log frequency of start codon among high-confidence ORFs
        features[33] = self.start_codon_log_freq as f32;
```

- [ ] **Step 4: Update any hard-coded `NUM_FEATURES` assertions in tests**

Search `src/ml_features.rs` tests for `NUM_FEATURES` and update expected values from 29 to 34. There are usually 2-3 assertions.

- [ ] **Step 5: Run unit tests**

```bash
cargo test --lib ml_features
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ml_features.rs
git commit -m "feat(ml): compute and export relative start-site features"
```

---

### Task 3: Update the Python training script

**Files:**
- Modify: `scripts/train_orf_score_model.py`

- [ ] **Step 1: Update `NUM_FEATURES` and `FEATURE_NAMES`**

Change:

```python
NUM_FEATURES = 29
```

to:

```python
NUM_FEATURES = 34
```

Append these five names to `FEATURE_NAMES`:

```python
    "best_alt_pwm_score",
    "pwm_ratio",
    "start_rank",
    "num_alt_starts",
    "start_codon_log_freq",
```

- [ ] **Step 2: Verify the script imports still match**

No other changes are needed; the script already reads features by name from the TSV produced by `--export-features`.

- [ ] **Step 3: Run a quick Python syntax check**

```bash
.venv/bin/python -m py_compile scripts/train_orf_score_model.py
```

Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_orf_score_model.py
git commit -m "feat(train): expect 34 ML features including start-site features"
```

---

### Task 4: Build the binary and retrain the model

**Files:**
- Create: `tests/golden/orf_model.onnx` (overwrite)

- [ ] **Step 1: Build the release binary**

```bash
cargo build --release
```

- [ ] **Step 2: Retrain the XGBoost model**

```bash
.venv/bin/python scripts/train_orf_score_model.py \
    -i tests/golden/annotgenomes \
    --model-type xgboost \
    --hard-negatives \
    -o tests/golden/orf_model.onnx
```

Expected: script completes and writes `tests/golden/orf_model.onnx`.

- [ ] **Step 3: Verify ONNX input shape**

```bash
.venv/bin/python - <<'PY'
import onnx
m = onnx.load("tests/golden/orf_model.onnx")
print(m.graph.input[0].type.tensor_type.shape)
PY
```

Expected: the second dimension is `34`.

- [ ] **Step 4: Commit the new model**

```bash
git add tests/golden/orf_model.onnx
git commit -m "feat(model): retrain ORF scorer with relative start-site features"
```

---

### Task 5: Verification and benchmark

**Files:**
- All modified Rust files
- `tests/golden/orf_model.onnx`

- [ ] **Step 1: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean.

- [ ] **Step 2: Run Rust tests**

```bash
cargo test --lib
cargo test --test cli_tests
```

Expected: lib tests pass except the pre-existing external-file detect_table failures; cli_tests all pass.

- [ ] **Step 3: Benchmark on first 50 annotated genomes**

```bash
.venv/bin/python - <<'PY'
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from compare_predictions import _parse_genbank_cds, compare

paths = sorted(Path('tests/golden/annotgenomes').glob('*.gb'))[:50]
model = 'tests/golden/orf_model.onnx'
agg = {'tp':0,'fp':0,'fn':0}
for p in paths:
    r = subprocess.run(['target/release/phanotate-rs','-i',str(p),'--detect-table','--yes','-f','sco','--model',model],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    ref = _parse_genbank_cds(str(p))
    pred = set()
    for line in r.stdout.splitlines():
        if line.startswith('#') or not line.strip(): continue
        x = line.split('\t')
        if len(x) >= 2:
            try: pred.add((int(x[0]), int(x[1])))
            except: pass
    c = compare(pred, ref, 3)
    for k in ('tp','fp','fn'): agg[k] += c[k]
p = agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
r = agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
f1 = 2*p*r/(p+r) if p+r else 0
print(f"F1={f1:.4f} P={p:.4f} R={r:.4f}")
PY
```

Expected: F1 > 0.6717 (the previous `--model` baseline).

- [ ] **Step 4: Commit verification results**

If the benchmark passes:

```bash
git add -A
git commit -m "test(model): verify 34-feature model beats baseline"
```

If it does not improve, report the F1 and decide whether to iterate before committing.

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every design requirement maps to a task.
- [ ] **No placeholders:** Every task shows exact code/commands.
- [ ] **Type consistency:** `NUM_FEATURES` is 34 everywhere, feature indices line up with `FEATURE_NAMES`, and new `Orf` fields are initialized in every literal.
- [ ] **Test coverage:** Unit tests updated; benchmark validates the improvement.
