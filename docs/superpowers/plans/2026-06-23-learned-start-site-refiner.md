# Learned Start-Site Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--start-model` flag that loads a small learned start-site scoring model, trained from trusted GenBank annotations, and uses it as an extra multiplier in `Orf::score()`.

**Architecture:** A new `src/start_refiner.rs` module defines an 11-feature vector and a linear model. The `Orf` struct gains a `start_score` multiplier. The CLI and Python bindings load a JSON model file and apply it before graph scoring. A new `scripts/train_start_model.py` script turns annotated GenBank files into that JSON model.

**Tech Stack:** Rust 2021, existing PHANOTATE-rs modules, Python 3 + scikit-learn for training, JSON for model serialization.

---

## File map

| file | responsibility |
|------|----------------|
| `src/start_refiner.rs` *(new)* | `StartSiteFeatures`, `StartModel`, JSON loader, unit tests |
| `src/lib.rs` | re-export `pub mod start_refiner` |
| `src/orf.rs` | add `start_score` field; update `score()` and every `Orf { ... }` literal |
| `src/main.rs` | add `--start-model` CLI flag; load/apply model in `process_genome` |
| `src/lib_python.rs` | add `start_model` parameter to `phanotate()` and `find_orfs()` |
| `scripts/train_start_model.py` *(new)* | parse GenBank annotations, enumerate ORFs, train logistic regression, export JSON |
| `tests/cli_tests.rs` | CLI regression/effect tests |
| `tests/test_python_bindings.py` | Python binding effect test |

---

## Task 1: Create `src/start_refiner.rs`

**Files:**
- Create: `src/start_refiner.rs`
- Test: `cargo test --lib -- start_refiner`

- [ ] **Step 1: Write the failing unit tests**

Create `src/start_refiner.rs` with the tests first:

```rust
//! Learned start-site scoring model.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_vector_length_matches_constant() {
        let f = StartSiteFeatures::new(&Orf {
            start: 1,
            stop: 100,
            frame: 1,
            seq: vec![b'a'; 100],
            rbs_score: 5,
            rbs_motif: None,
            pstop: 0.01,
            weight_rbs: 1.5,
            motif_score: 1.2,
            hold: 10.0,
            dicodon_score: 0.1,
            start_score: 1.0,
            weight: 1.0,
        });
        assert_eq!(f.0.len(), NUM_START_FEATURES);
    }

    #[test]
    fn model_score_is_within_bounds() {
        let model = StartModel {
            coeffs: [0.1; NUM_START_FEATURES],
            mean: [0.0; NUM_START_FEATURES],
            std: [1.0; NUM_START_FEATURES],
        };
        let f = StartSiteFeatures([0.0; NUM_START_FEATURES]);
        let s = model.score(&f);
        assert!(s >= 0.5);
        assert!(s <= 2.0);
    }
}
```

Run:

```bash
cargo test --lib -- start_refiner
```

Expected: compile fails because `StartSiteFeatures`, `StartModel`, `NUM_START_FEATURES`, and the updated `Orf` do not exist yet.

- [ ] **Step 2: Add serde dependencies**

In `Cargo.toml`, under `[dependencies]` add:

```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 3: Implement the module**

Add above the `#[cfg(test)]` block:

```rust
use crate::orf::Orf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const NUM_START_FEATURES: usize = 11;

/// Fixed feature vector for a candidate start site.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StartSiteFeatures(pub [f64; NUM_START_FEATURES]);

impl StartSiteFeatures {
    /// Build features from an ORF. This must stay in sync with the training
    /// script in `scripts/train_start_model.py`.
    pub fn new(orf: &Orf) -> Self {
        let mut f = [0.0; NUM_START_FEATURES];
        let codon = orf.start_codon();
        if codon == b"atg" {
            f[0] = 1.0;
        } else if codon == b"gtg" {
            f[1] = 1.0;
        } else if codon == b"ttg" {
            f[2] = 1.0;
        } else {
            f[3] = 1.0;
        }
        // NUM_RBS_BINS is 28 in the current rbs_scanner.
        const NUM_RBS_BINS: f64 = 28.0;
        f[4] = (orf.rbs_score as f64 / NUM_RBS_BINS).clamp(0.0, 1.0);
        f[5] = orf.motif_score.clamp(0.0, 10.0);
        f[6] = (orf.seq.len() as f64).ln();
        f[7] = orf.dicodon_score.clamp(0.001, 1000.0);
        match orf.frame.abs() {
            1 => f[8] = 1.0,
            2 => f[9] = 1.0,
            3 => f[10] = 1.0,
            _ => {}
        }
        Self(f)
    }
}

/// Linear start-site model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartModel {
    pub coeffs: [f64; NUM_START_FEATURES],
    pub mean: [f64; NUM_START_FEATURES],
    pub std: [f64; NUM_START_FEATURES],
}

impl StartModel {
    /// Load a model from a JSON file produced by `train_start_model.py`.
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read start model: {:?}", path))?;
        let model: Self = serde_json::from_str(&text)
            .with_context(|| format!("Failed to parse start model JSON: {:?}", path))?;
        Ok(model)
    }

    /// Score a candidate start. Returns a multiplier in [0.5, 2.0].
    pub fn score(&self, features: &StartSiteFeatures) -> f64 {
        let mut z = 0.0;
        for i in 0..NUM_START_FEATURES {
            let denom = if self.std[i] == 0.0 { 1.0 } else { self.std[i] };
            let v = (features.0[i] - self.mean[i]) / denom;
            z += v * self.coeffs[i];
        }
        let p = 1.0 / (1.0 + (-z).exp());
        (p * 1.5 + 0.5).clamp(0.5, 2.0)
    }
}
```

- [ ] **Step 4: Register the module in `src/lib.rs`**

Add:

```rust
pub mod start_refiner;
```

- [ ] **Step 5: Run the unit tests**

```bash
cargo test --lib -- start_refiner
```

Expected: tests compile but fail until Task 2 adds the `start_score` field. Proceed to Task 2.

- [ ] **Step 6: Commit**

```bash
git add src/start_refiner.rs src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat(start_refiner): add start-site feature vector and linear model"
```

---

## Task 2: Add `start_score` to `Orf`

**Files:**
- Modify: `src/orf.rs`
- Modify: `src/ml_features.rs`
- Modify: `src/nonsd_motif.rs`
- Modify: `src/graph.rs`
- Modify: `src/dicodon.rs`
- Modify: `src/output.rs`
- Modify: `src/lib_python.rs`
- Test: `cargo test --lib -- --skip debug_ --skip regression_ --skip table4_genome --skip lambda_`

- [ ] **Step 1: Update the `Orf` struct and `score()`**

In `src/orf.rs`, add the field:

```rust
pub struct Orf {
    pub start: usize,
    pub stop: usize,
    pub frame: i8,
    pub seq: Vec<u8>,
    pub rbs_score: usize,
    pub pstop: f64,
    pub weight_rbs: f64,
    pub motif_score: f64,
    pub rbs_motif: Option<String>,
    pub hold: f64,
    pub dicodon_score: f64,
    pub start_score: f64, // learned start-site multiplier, default 1.0
    pub weight: f64,
}
```

Update `Orf::score()`:

```rust
pub fn score(&mut self, start_codons: &std::collections::HashMap<Vec<u8>, f64>) {
    let mut s = self.dicodon_score * self.start_score;
    let sc = self.start_codon().to_vec();
    if let Some(&w) = start_codons.get(&sc) {
        s *= w;
    }
    s *= self.weight_rbs;
    s *= self.motif_score;
    self.weight = -s;
}
```

- [ ] **Step 2: Initialize `start_score: 1.0` in every `Orf { ... }` literal**

Add `start_score: 1.0` immediately after `dicodon_score: ...` in every `Orf` literal in the crate. Use `cargo build` to find missing fields:

```bash
cargo build
```

Repeat until no compile errors. Files that contain `Orf { ... }` literals (check with `grep -n "Orf {" src/*.rs`) include:
- `src/orf.rs` (five literals)
- `src/output.rs`
- `src/nonsd_motif.rs`
- `src/ml_features.rs`
- `src/dicodon.rs`
- `src/graph.rs`
- `src/lib_python.rs`

Example insertion:

```rust
dicodon_score: 1.0 / hold,
start_score: 1.0,
weight: 1.0,
```

- [ ] **Step 3: Run the unit tests**

```bash
cargo test --lib -- --skip debug_ --skip regression_ --skip table4_genome --skip lambda_
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/orf.rs src/output.rs src/nonsd_motif.rs src/ml_features.rs src/dicodon.rs src/graph.rs src/lib_python.rs
git commit -m "feat(orf): add start_score multiplier used in score()"
```

---

## Task 3: Wire `--start-model` into the CLI

**Files:**
- Modify: `src/main.rs`
- Test: `cargo test --test cli_tests -- start_model`

- [ ] **Step 1: Add the CLI flag**

In the `Cli` struct in `src/main.rs`, after the `--dicodon` flag (or near the other scoring flags):

```rust
/// Path to a learned start-site scoring model (JSON).
#[arg(long = "start-model", value_name = "FILE")]
start_model: Option<PathBuf>,
```

- [ ] **Step 2: Update `process_genome` signature**

Change the return type to `Result` and add the parameter:

```rust
fn process_genome(
    genome: Genome,
    start_codons_map: &HashMap<Vec<u8>, f64>,
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    format: Format,
    closed_ends: bool,
    mask_n: bool,
    table: u8,
    force_non_sd: bool,
    force_sd: bool,
    #[cfg(feature = "ml")] ml_scorer: &Option<phanotate_rs::ml_scorer::MlScorer>,
    #[cfg(not(feature = "ml"))] _ml_scorer: &Option<()>,
    prodigal_rbs: bool,
    dicodon: bool,
    start_model: Option<&std::path::Path>,
) -> Result<(String, String, String)> {
```

Then change every `return (...)` in the function body to `return Ok((...))`.

- [ ] **Step 3: Apply the model before scoring**

After the dicodon-score block and before the `// --- Score ORFs ---` section, add:

```rust
if let Some(path) = start_model {
    let model = phanotate_rs::start_refiner::StartModel::from_json(path)
        .with_context(|| format!("Failed to load start model from {:?}", path))?;
    for orf in &mut orfs {
        let features = phanotate_rs::start_refiner::StartSiteFeatures::new(orf);
        orf.start_score = model.score(&features);
    }
}
```

- [ ] **Step 4: Update the two `process_genome` call sites**

The closures inside the Rayon map now return `Result<(String, String, String)>`. Both branches should look like:

```rust
.map(|genome| {
    process_genome(
        genome,
        &start_codons_map,
        &start_codons,
        &stop_codons,
        format,
        cli.closed_ends,
        cli.mask_n,
        effective_table,
        cli.force_non_sd,
        cli.force_sd,
        &ml_scorer,
        cli.prodigal_rbs,
        cli.dicodon,
        cli.start_model.as_deref(),
    )
})
.collect::<Result<Vec<_>>>()?
```

Adjust the surrounding code so that `results` is bound to the collected vector and any error is propagated with `?`.

- [ ] **Step 5: Add CLI tests**

In `tests/cli_tests.rs`, add:

```rust
// ---------------------------------------------------------------------------
// Start-model scoring
// ---------------------------------------------------------------------------
#[test]
fn start_model_flag_requires_valid_file() {
    let (_stdout, _stderr, code) = run(
        &["-i", "tests/data/small.fasta", "--start-model", "/nonexistent.json"],
        None,
    );
    assert_ne!(code, 0, "missing model file should fail");
}

#[test]
fn start_model_runs_without_error() {
    // A minimal valid model: zero coefficients, unit scaling.
    let tmpdir = tempfile::tempdir().unwrap();
    let model_path = tmpdir.path().join("model.json");
    std::fs::write(
        &model_path,
        r#"{"version":1,"num_features":11,"coeffs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"mean":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"std":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}"#,
    )
    .unwrap();

    let (out, _, code) = run(
        &[
            "-i",
            "tests/data/small.fasta",
            "-f",
            "sco",
            "--start-model",
            model_path.to_str().unwrap(),
        ],
        None,
    );
    assert_eq!(code, 0, "--start-model should succeed with a valid JSON model");
    let data_line = out.lines().find(|l| !l.starts_with('#')).unwrap_or("");
    assert!(!data_line.is_empty(), "should produce data lines");
    assert_eq!(data_line.split('\t').count(), 5, "SCO line should have 5 columns");
}
```

Run:

```bash
cargo test --test cli_tests -- start_model
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs tests/cli_tests.rs
git commit -m "feat(cli): add --start-model flag and apply model in pipeline"
```

---

## Task 4: Expose `start_model` in Python bindings

**Files:**
- Modify: `src/lib_python.rs`
- Test: `cargo check --features python` and `pytest tests/test_python_bindings.py -v`

- [ ] **Step 1: Add parameter to `phanotate()`**

Update the `#[pyo3(signature = (...))]` block:

```rust
#[pyo3(signature = (
    sequence,
    seq_id = None,
    format = "gbk",
    table = 11,
    closed_ends = false,
    mask_n = false,
    detect_table = false,
    non_sd = false,
    sd = false,
    prodigal_rbs = false,
    dicodon = false,
    start_model = None,
    min_orf_len = 90,
))]
```

Update the function signature:

```rust
fn phanotate(
    sequence: &str,
    seq_id: Option<&str>,
    format: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    detect_table: bool,
    non_sd: bool,
    sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    start_model: Option<&str>,
    min_orf_len: usize,
) -> PyResult<PyObject> {
```

Add a docstring line near the other scoring flags:

```rust
/// start_model : str, optional
///     Path to a JSON start-site scoring model produced by
///     `scripts/train_start_model.py`. Default is None.
```

Pass it to `process_single_genome`:

```rust
    min_orf_len,
    non_sd,
    sd,
    prodigal_rbs,
    dicodon,
    start_model,
);
```

- [ ] **Step 2: Add parameter to `process_single_genome`**

Update the signature:

```rust
fn process_single_genome(
    id: &str,
    dna: &[u8],
    rc_dna: &[u8],
    start_codons_map: &HashMap<Vec<u8>, f64>,
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    format: Format,
    closed_ends: bool,
    mask_n: bool,
    table: u8,
    min_orf_len: usize,
    force_non_sd: bool,
    force_sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    start_model: Option<&str>,
) -> (String, String, String, Vec<PyGene>, bool) {
```

Apply the model after the dicodon block and before the `for orf in &mut orfs { orf.score(...) }` loop:

```rust
if let Some(path) = start_model {
    let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
    for orf in &mut orfs {
        let features = crate::start_refiner::StartSiteFeatures::new(orf);
        orf.start_score = model.score(&features);
    }
}
```

- [ ] **Step 3: Add parameter to `find_orfs()`**

Update the signature block:

```rust
#[pyo3(signature = (
    sequence,
    table = 11,
    closed_ends = false,
    mask_n = false,
    min_orf_len = 90,
    prodigal_rbs = false,
    start_model = None,
))]
fn find_orfs(
    sequence: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    min_orf_len: usize,
    prodigal_rbs: bool,
    start_model: Option<&str>,
) -> PyResult<Vec<PyOrf>> {
```

At the end of `find_orfs()`, before returning, apply the model if provided:

```rust
if let Some(path) = start_model {
    let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
    for orf in &mut orfs {
        let features = crate::start_refiner::StartSiteFeatures::new(orf);
        orf.start_score = model.score(&features);
    }
}
```

- [ ] **Step 4: Add Python test**

In `tests/test_python_bindings.py`, after the dicodon test:

```python
    def test_phanotate_start_model(self):
        import json, tempfile, os
        model = {
            "version": 1,
            "coeffs": [0.0] * 11,
            "mean": [0.0] * 11,
            "std": [1.0] * 11,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(model, fh)
            path = fh.name
        try:
            out = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", start_model=path)
            assert out["primary"]
            assert isinstance(out["genes"], list)
        finally:
            os.unlink(path)
```

- [ ] **Step 5: Run checks**

```bash
PYO3_PYTHON=$PWD/.venv/bin/python cargo check --features python
```

Expected: compiles.

```bash
. .venv/bin/activate && maturin develop && pytest tests/test_python_bindings.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lib_python.rs tests/test_python_bindings.py
git commit -m "feat(python): expose start_model parameter in bindings"
```

---

## Task 5: Create `scripts/train_start_model.py`

**Files:**
- Create: `scripts/train_start_model.py`
- Test: run on `tests/golden/NC_001365.gb` and inspect output JSON

- [ ] **Step 1: Create the script**

Create `scripts/train_start_model.py`:

```python
#!/usr/bin/env python3
"""Train a start-site scoring model from annotated GenBank files.

Example:
    python scripts/train_start_model.py \
        -i tests/golden/NC_001365.gb \
        -t 4 \
        -o /tmp/nc001365_start_model.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
import phanotate_rs

NUM_FEATURES = 11
NUM_RBS_BINS = 28


def parse_genbank(path: str, table: int):
    """Return (sequence, list of annotated (start, stop, strand))."""
    text = Path(path).read_text()
    seq_match = re.search(r"ORIGIN\s+(.+?)//", text, re.DOTALL)
    if not seq_match:
        raise ValueError(f"No ORIGIN found in {path}")
    seq = "".join(re.findall(r"[a-z]+", seq_match.group(1)))

    cds = []
    for m in re.finditer(r"^\s+CDS\s+(.+)", text, re.MULTILINE):
        loc = m.group(1).strip()
        strand = "+"
        if loc.startswith("complement("):
            strand = "-"
            loc = loc[len("complement("):].rstrip(")")
        if loc.startswith("join("):
            loc = loc[len("join("):].rstrip(")")
        parts = [p.strip() for p in loc.split(",")]
        coords = []
        for p in parts:
            mm = re.match(r"(\d+)\.\.(\d+)", p)
            if mm:
                coords.append((int(mm.group(1)), int(mm.group(2))))
        if not coords:
            continue
        # For forward features the start codon is at the low coordinate;
        # for complement features it is at the high coordinate. This matches
        # PHANOTATE-rs's convention where reverse-strand ORFs have start > stop.
        low = min(s for s, _ in coords)
        high = max(e for _, e in coords)
        if strand == "+":
            cds.append((low, high, "+"))
        else:
            cds.append((high, low, "-"))
    return seq, cds


def features(orf):
    """Build the same 11-feature vector as StartSiteFeatures::new."""
    f = np.zeros(NUM_FEATURES)
    codon = orf.start_codon.lower()
    if codon == "atg":
        f[0] = 1.0
    elif codon == "gtg":
        f[1] = 1.0
    elif codon == "ttg":
        f[2] = 1.0
    else:
        f[3] = 1.0
    f[4] = min(orf.rbs_score / NUM_RBS_BINS, 1.0)
    f[5] = min(max(orf.motif_score, 0.0), 10.0)
    f[6] = np.log(len(orf.sequence))
    f[7] = min(max(1.0 / orf.hold, 0.001), 1000.0)
    frame_abs = abs(orf.frame)
    if frame_abs == 1:
        f[8] = 1.0
    elif frame_abs == 2:
        f[9] = 1.0
    elif frame_abs == 3:
        f[10] = 1.0
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="GenBank file(s)", nargs="+")
    parser.add_argument("-t", "--table", type=int, default=11)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--min-orf-len", type=int, default=90)
    args = parser.parse_args()

    X, y = [], []
    for path in args.input:
        seq, annotations = parse_genbank(path, args.table)
        ann_set = {(s, e) for s, e, _strand in annotations}
        orfs = phanotate_rs.find_orfs(seq, table=args.table, min_orf_len=args.min_orf_len)
        for orf in orfs:
            label = 1 if (orf.start, orf.stop) in ann_set else 0
            X.append(features(orf))
            y.append(label)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least one positive and one negative example")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xs, y)

    out = {
        "version": 1,
        "num_features": NUM_FEATURES,
        "coeffs": model.coef_[0].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}: {sum(y)} positive, {len(y) - sum(y)} negative examples")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install Python training dependencies**

```bash
. .venv/bin/activate && pip install scikit-learn
```

- [ ] **Step 3: Run it on the example genome**

```bash
. .venv/bin/activate && python scripts/train_start_model.py -i tests/golden/NC_001365.gb -t 4 -o /tmp/nc001365_start_model.json
```

Expected: writes JSON with 11 coefficients, mean, std.

- [ ] **Step 4: Smoke-test the model at the CLI**

```bash
./target/release/phanotate-rs -i tests/golden/NC_001365.gb -g 4 -f sco --start-model /tmp/nc001365_start_model.json | head -5
```

Expected: runs successfully and output differs from the default run.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_start_model.py
git commit -m "feat(scripts): add train_start_model.py for start-site models"
```

---

## Task 6: Quality gates

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Clippy (default features)**

```bash
cargo clippy -- -D warnings
```

Expected: clean (pre-existing `lib_python.rs` warnings only appear with `--features python`).

- [ ] **Step 3: Unit tests**

```bash
cargo test --lib -- --skip debug_ --skip regression_ --skip table4_genome --skip lambda_
```

Expected: pass.

- [ ] **Step 4: Integration tests**

```bash
cargo test --test cli_tests -- --skip test_flag_f_ --skip test_flag_g_ --skip test_flag_i_stdin --skip test_flag_a_protein --skip test_flag_d_nucleotide --skip test_phix174
cargo test --test detect_table_tests -- --skip test_detect_table_batch --skip test_pipe_mode_no_prompt --skip test_confidence_high --skip test_table11_genome_scores_highest --skip test_yes_flag_skips_prompt
```

Expected: pass.

- [ ] **Step 5: Python tests**

```bash
. .venv/bin/activate && maturin develop && pytest tests/test_python_bindings.py -v
```

Expected: pass.

- [ ] **Step 6: Release build**

```bash
cargo build --release
```

Expected: succeeds.

- [ ] **Step 7: Commit final formatting fixes**

```bash
git add -u
git commit -m "style: formatting and quality gates for start-site refiner"
```

---

## Spec coverage check

| spec section | implementing task |
|--------------|-------------------|
| `StartSiteFeatures` 11-feature vector | Task 1 |
| `StartModel` linear model + JSON loader | Task 1 |
| `Orf::start_score` multiplier | Task 2 |
| `--start-model` CLI flag | Task 3 |
| Python `start_model` parameter | Task 4 |
| `scripts/train_start_model.py` | Task 5 |
| Tests for model, CLI, Python | Tasks 1, 3, 4 |
| Quality gates | Task 6 |

## Placeholder scan

- No TBD/TODO placeholders.
- No vague steps; every step includes exact code or commands.
- Type names (`StartSiteFeatures`, `StartModel`, `start_score`) are consistent across tasks.
