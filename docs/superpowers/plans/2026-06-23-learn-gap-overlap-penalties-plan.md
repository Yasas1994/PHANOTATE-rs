# Learn Gap/Overlap Penalty Scales Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-genome calibration of gap and overlap edge penalties on the `--model` path, replacing the single `PHANOTATE_MODEL_PENALTY_SCALE` environment variable.

**Architecture:** A new `src/penalty_calibration.rs` module computes `gap_scale` and `overlap_scale` from the median absolute ORF reward and representative raw gap/overlap scores. `Graph::from_orfs` accepts the two scales separately and applies them inside `weights.rs`. A tuning script searches the two global target ratios on annotated genomes.

**Tech Stack:** Rust 2021, Python 3 (scripts), existing `tests/golden/annotgenomes` reference genomes.

---

### Task 1: Split `penalty_scale` into `gap_scale` and `overlap_scale` in `weights.rs`

**Files:**
- Modify: `src/weights.rs:9`
- Modify: `src/weights.rs:123`
- Test: `src/weights.rs` unit tests

- [ ] **Step 1: Update `score_overlap` signature and docs**

```rust
/// Score an overlap edge.
/// `length` can be negative (the formula handles it as in the reference).
/// `direction` is "same" or "diff" (strand switch).
/// `pstop` is the average P(stop) of the two flanking ORFs.
/// `overlap_scale` is a multiplicative calibration factor used in the learned
/// `--model` path to align overlap penalties with the ORF score magnitude.
pub fn score_overlap(length: isize, direction: &str, pstop: f64, overlap_scale: f64) -> f64 {
    let o = 1.0 - pstop;
    let s = 0.05;
    let mut score = o.powi(length as i32);
    score = 1.0 / score;
    if direction == "diff" {
        score += 1.0 / s;
    }
    score * overlap_scale
}
```

- [ ] **Step 2: Update `score_gap` signature and docs**

```rust
/// Score a gap edge.
/// `length` is the gap length in nucleotides (can be negative for overlaps).
/// `direction` is "same" or "diff".
/// `pgap` is the genome-wide average P(not_stop).
/// `gap_scale` is a multiplicative calibration factor; see `score_overlap`.
pub fn score_gap(length: isize, direction: &str, pgap: f64, gap_scale: f64) -> f64 {
    let g = 1.0 - pgap;
    let s = 0.05;

    let score = if length > 300 {
        g.powi(100) + length as f64
    } else {
        1.0 / g.powf(length as f64 / 3.0)
    };
    let score = if direction == "diff" {
        score + 1.0 / s
    } else {
        score
    };
    score * gap_scale
}
```

- [ ] **Step 3: Update all unit-test calls to pass `1.0`**

Replace every `score_gap(..., 1.0)` and `score_overlap(..., 1.0)` in the `#[cfg(test)]` block. There are ~17 calls; each keeps `1.0` for both the old `penalty_scale` position.

- [ ] **Step 4: Run weight unit tests**

```bash
cargo test --lib weights
```

Expected: all `weights` tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/weights.rs
git commit -m "refactor(weights): split penalty_scale into gap_scale and overlap_scale"
```

---

### Task 2: Update `Graph::from_orfs` to accept two scales

**Files:**
- Modify: `src/graph.rs:161-166`
- Modify: all `score_gap`/`score_overlap` calls in `src/graph.rs`
- Test: `src/graph.rs` unit tests

- [ ] **Step 1: Change the `from_orfs` signature**

```rust
pub fn from_orfs(
    orfs: &[Orf],
    contig_length: usize,
    pgap: f64,
    gap_scale: f64,
    overlap_scale: f64,
) -> (Self, Vec<usize>)
```

- [ ] **Step 2: Replace every `penalty_scale` argument in calls**

Use the editor to replace all occurrences of `penalty_scale,` in `src/graph.rs` with either `gap_scale,` or `overlap_scale,` based on whether the call is `score_gap` or `score_overlap`. There are 15 calls total.

- [ ] **Step 3: Update graph unit-test call sites**

In `src/graph.rs` tests (~2 calls), change:

```rust
Graph::from_orfs(&orfs, 100, 0.05, 1.0)
```

to:

```rust
Graph::from_orfs(&orfs, 100, 0.05, 1.0, 1.0)
```

- [ ] **Step 4: Run graph unit tests**

```bash
cargo test --lib graph
```

Expected: all `graph` tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/graph.rs
git commit -m "refactor(graph): pass separate gap_scale and overlap_scale to from_orfs"
```

---

### Task 3: Create the penalty-calibration module

**Files:**
- Create: `src/penalty_calibration.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/penalty_calibration.rs`**

```rust
//! Per-genome calibration of gap/overlap edge penalties for the ONNX model path.

use crate::orf::Orf;
use crate::weights::{score_gap, score_overlap};

const MODEL_GAP_TARGET_RATIO: f64 = 1.0;
const MODEL_OVERLAP_TARGET_RATIO: f64 = 1.0;
const MIN_ORFS_FOR_CALIBRATION: usize = 3;
const SCALE_MIN: f64 = 0.1;
const SCALE_MAX: f64 = 10.0;

/// Compute per-genome gap and overlap scales for the `--model` path.
///
/// The scales are chosen so that representative raw gap/overlap penalties are
/// proportional to the median absolute ORF reward on this genome.
///
/// On the default (non-model) path, callers should simply use `(1.0, 1.0)`.
pub fn compute_model_penalty_scales(orfs: &[Orf], pgap: f64) -> (f64, f64) {
    if orfs.len() < MIN_ORFS_FOR_CALIBRATION {
        return (1.0, 1.0);
    }

    let mut orf_rewards: Vec<f64> = orfs.iter().map(|o| o.weight.abs()).collect();
    orf_rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_orf = orf_rewards[orf_rewards.len() / 2];
    if !med_orf.is_finite() || med_orf <= 0.0 {
        return (1.0, 1.0);
    }

    let gap_target: f64 = std::env::var("PHANOTATE_MODEL_GAP_TARGET_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MODEL_GAP_TARGET_RATIO);
    let overlap_target: f64 = std::env::var("PHANOTATE_MODEL_OVERLAP_TARGET_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MODEL_OVERLAP_TARGET_RATIO);

    let mut gap_raws = Vec::new();
    for &len in &[0_isize, 10, 30, 100, 300] {
        gap_raws.push(score_gap(len, "same", pgap, 1.0));
        gap_raws.push(score_gap(len, "diff", pgap, 1.0));
    }
    gap_raws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_gap = gap_raws[gap_raws.len() / 2];

    let pstop_avg = if orfs.is_empty() {
        pgap
    } else {
        orfs.iter().map(|o| o.pstop).sum::<f64>() / orfs.len() as f64
    };
    let mut overlap_raws = Vec::new();
    for &len in &[1_isize, 5, 10, 20, 50] {
        overlap_raws.push(score_overlap(len, "same", pstop_avg, 1.0));
        overlap_raws.push(score_overlap(len, "diff", pstop_avg, 1.0));
    }
    overlap_raws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_overlap = overlap_raws[overlap_raws.len() / 2];

    let gap_scale = if med_gap.is_finite() && med_gap > 0.0 {
        (gap_target * med_orf / med_gap).clamp(SCALE_MIN, SCALE_MAX)
    } else {
        1.0
    };
    let overlap_scale = if med_overlap.is_finite() && med_overlap > 0.0 {
        (overlap_target * med_orf / med_overlap).clamp(SCALE_MIN, SCALE_MAX)
    } else {
        1.0
    };

    (gap_scale, overlap_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_orfs_falls_back_to_one() {
        assert_eq!(compute_model_penalty_scales(&[], 0.05), (1.0, 1.0));
    }
}
```

- [ ] **Step 2: Register the module in `src/lib.rs`**

Add `pub mod penalty_calibration;` alongside the other `pub mod` declarations.

- [ ] **Step 3: Compile**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/penalty_calibration.rs src/lib.rs
git commit -m "feat(penalty): add per-genome gap/overlap calibration module"
```

---

### Task 4: Wire calibration into the CLI path

**Files:**
- Modify: `src/main.rs:285-299`

- [ ] **Step 1: Replace penalty-scale env var with calibration call**

Before the graph construction line, compute the scales:

```rust
    // --- Build graph ---
    let (gap_scale, overlap_scale) = if orf_model.is_some() {
        phanotate_rs::penalty_calibration::compute_model_penalty_scales(&orfs, pstop)
    } else {
        (1.0, 1.0)
    };
    let (graph, endpoints) = Graph::from_orfs(&orfs, contig_length, pstop, gap_scale, overlap_scale);
```

Remove the old `penalty_scale` variable and the `PHANOTATE_MODEL_PENALTY_SCALE` lookup.

- [ ] **Step 2: Compile**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): use per-genome gap/overlap scales on --model path"
```

---

### Task 5: Wire calibration into the Python bindings path

**Files:**
- Modify: `src/lib_python.rs:511-520`

- [ ] **Step 1: Replace penalty-scale env var with calibration call**

```rust
    // --- Build graph ---
    let (gap_scale, overlap_scale) = if orf_model.is_some() {
        crate::penalty_calibration::compute_model_penalty_scales(&orfs, pstop)
    } else {
        (1.0, 1.0)
    };
    let (graph, endpoints) = Graph::from_orfs(&orfs, contig_length, pstop, gap_scale, overlap_scale);
```

- [ ] **Step 2: Compile with python feature**

```bash
cargo check --features python
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/lib_python.rs
git commit -m "feat(python): use per-genome gap/overlap scales on --model path"
```

---

### Task 6: Create the target-ratio tuning script

**Files:**
- Create: `scripts/tune_penalty_balance.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Grid-search the global gap/overlap target ratios on annotated genomes."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_predictions import _parse_genbank_cds, compare

ANNOT_DIR = Path("tests/golden/annotgenomes")
BINARY = Path("target/release/phanotate-rs")
MODEL = Path("tests/golden/orf_model.onnx")


def paths(limit: int | None = None):
    p = sorted(ANNOT_DIR.glob("*.gb"))
    if limit:
        p = p[:limit]
    return p


def evaluate(gap_target: float, overlap_target: float, genomes: list[Path]) -> dict:
    env = os.environ.copy()
    env["PHANOTATE_MODEL_GAP_TARGET_RATIO"] = str(gap_target)
    env["PHANOTATE_MODEL_OVERLAP_TARGET_RATIO"] = str(overlap_target)
    agg = {"tp": 0, "fp": 0, "fn": 0}
    for genome in genomes:
        r = subprocess.run(
            [
                str(BINARY),
                "-i",
                str(genome),
                "--detect-table",
                "--yes",
                "-f",
                "sco",
                "--model",
                str(MODEL),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        ref = _parse_genbank_cds(str(genome))
        pred = set()
        for line in r.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                pred.add((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
        c = compare(pred, ref, 3)
        for k in ("tp", "fp", "fn"):
            agg[k] += c[k]
    p = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
    r = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, **agg}


def main():
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}; run cargo build --release --features ml", file=sys.stderr)
        sys.exit(1)
    genomes = paths(50)
    print(f"Tuning on {len(genomes)} genomes")
    best = None
    for gap in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        for overlap in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            res = evaluate(gap, overlap, genomes)
            print(f"gap={gap:.2f} overlap={overlap:.2f} -> F1={res['f1']:.4f}")
            if best is None or res["f1"] > best["f1"]:
                best = {"gap": gap, "overlap": overlap, **res}
    print("\nBest:")
    print(f"  gap_target={best['gap']:.2f}")
    print(f"  overlap_target={best['overlap']:.2f}")
    print(f"  F1={best['f1']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/tune_penalty_balance.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/tune_penalty_balance.py
git commit -m "feat(scripts): add grid-search tuner for gap/overlap target ratios"
```

---

### Task 7: Build with ML support and run the tuner

**Files:**
- Modify: `src/penalty_calibration.rs` constants (after tuning)

- [ ] **Step 1: Build release binary with ML feature**

```bash
cargo build --release --features ml
```

- [ ] **Step 2: Run coarse grid search**

```bash
.venv/bin/python scripts/tune_penalty_balance.py
```

Expected: prints a best `(gap_target, overlap_target)` pair and F1.

- [ ] **Step 3: Optionally refine around the best coarse value**

Edit the two `for` ranges in `scripts/tune_penalty_balance.py` to zoom in around the best pair, then rerun.

- [ ] **Step 4: Update constants in `src/penalty_calibration.rs`**

Replace:

```rust
const MODEL_GAP_TARGET_RATIO: f64 = 1.0;
const MODEL_OVERLAP_TARGET_RATIO: f64 = 1.0;
```

with the tuned values (e.g.):

```rust
const MODEL_GAP_TARGET_RATIO: f64 = 0.7;
const MODEL_OVERLAP_TARGET_RATIO: f64 = 1.3;
```

- [ ] **Step 5: Commit**

```bash
git add src/penalty_calibration.rs
git commit -m "feat(penalty): set learned gap/overlap target ratios"
```

---

### Task 8: Verification

**Files:**
- All modified Rust files

- [ ] **Step 1: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean (zero warnings/errors).

- [ ] **Step 2: Run Rust unit tests**

```bash
cargo test --lib
```

Expected: ~167 pass, only pre-existing external-file detect_table failures.

- [ ] **Step 3: Run CLI integration tests**

```bash
cargo test --test cli_tests
```

Expected: all pass.

- [ ] **Step 4: Verify model path improvement**

```bash
.venv/bin/python - <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from compare_predictions import _parse_genbank_cds, compare

paths = sorted(Path('tests/golden/annotgenomes').glob('*.gb'))[:50]
model = 'tests/golden/orf_model.onnx'

def f1(path):
    r = subprocess.run(['target/release/phanotate-rs','-i',str(path),'--detect-table','--yes','-f','sco','--model',model], stdout=subprocess.PIPE, text=True)
    ref = _parse_genbank_cds(str(path))
    pred = set()
    for line in r.stdout.splitlines():
        if line.startswith('#') or not line.strip(): continue
        p = line.split('\t')
        if len(p) >= 2:
            try: pred.add((int(p[0]), int(p[1])))
            except: pass
    c = compare(pred, ref, 3)
    p = c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0
    r = c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0
    return 2*p*r/(p+r) if p+r else 0

print('F1 =', sum(f1(p) for p in paths)/len(paths))
PY
```

Expected: F1 higher than the previous `--model` baseline of 0.675.

- [ ] **Step 5: Final commit if verification passed**

```bash
git add -A
git commit -m "test(penalty): verify learned gap/overlap scales with annotated genomes"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every section of the design doc maps to at least one task.
- [ ] **No placeholders:** Every task shows exact code/commands.
- [ ] **Type consistency:** `Graph::from_orfs`, `score_gap`, `score_overlap`, `compute_model_penalty_scales` signatures match across tasks.
- [ ] **Test coverage:** Unit tests updated for new signatures; integration/tuning scripts validate the improvement.
