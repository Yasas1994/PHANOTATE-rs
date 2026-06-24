# Learned Start-Site Refiner — Design Spec

**Project:** PHANOTATE-rs  
**Date:** 2026-06-23  
**Feature:** Optional learned start-site scoring model trained from trusted GenBank annotations.

---

## 1. Goal

Reduce false-positive gene predictions that come from choosing an upstream in-frame start codon instead of the annotated start. Use trusted GenBank CDS coordinates as supervised labels to train a small, interpretable scoring model for candidate start sites.

The model score becomes an additional multiplier in `Orf::score()`. When the flag is not used, behavior is identical to today.

---

## 2. Motivation

On `tests/golden/NC_001365.gb` (table 4), PHANOTATE-rs recovers all 12 annotated genes but predicts 19 genes overall. prodigal-gv predicts only 13 genes and hits starts exactly (e.g., `5015`, `3555`, `5839`). PHANOTATE-rs often chooses an upstream start (`4985`, `3546`, `5836`), creating extra ORFs that lower precision.

A learned start-site score should push the shortest-path search toward annotated-like starts.

---

## 3. Feature vector

`StartSiteFeatures` is a fixed-length vector computed for each candidate ORF:

| index | feature | notes |
|-------|---------|-------|
| 0 | ATG start | 1.0 if start codon is ATG, else 0.0 |
| 1 | GTG start | 1.0 if GTG |
| 2 | TTG start | 1.0 if TTG |
| 3 | other start | 1.0 if none of the above |
| 4 | normalized RBS score | `rbs_score / max_bin` or Prodigal bin / max bin |
| 5 | motif score | `motif_score` already computed by non-SD / Prodigal path |
| 6 | log ORF length | `ln(seq.len() as f64)` |
| 7 | coding-potential multiplier | `dicodon_score` if `--dicodon`, else `1.0 / hold` |
| 8 | frame 1 | 1.0 if `frame == 1` |
| 9 | frame 2 | 1.0 if `frame == 2` |
| 10 | frame 3 | 1.0 if `frame == 3` |

`NUM_START_FEATURES = 11`.

---

## 4. Model

A simple linear model:

```rust
pub struct StartModel {
    coeffs: [f64; NUM_START_FEATURES],
    mean: [f64; NUM_START_FEATURES],
    std: [f64; NUM_START_FEATURES],
}

impl StartModel {
    pub fn from_json(path: &Path) -> Result<Self>;

    pub fn score(&self, features: &StartSiteFeatures) -> f64 {
        let mut z = 0.0;
        for i in 0..NUM_START_FEATURES {
            let v = (features.0[i] - self.mean[i]) / self.std[i];
            z += v * self.coeffs[i];
        }
        // Convert log-odds to a multiplier in [0.5, 2.0]
        (1.0 / (1.0 + (-z).exp())).mul_add(1.5, 0.5)
    }
}
```

The output is clamped to `[0.5, 2.0]` so the model cannot destabilize the shortest-path search.

---

## 5. Runtime integration

### 5.1 `Orf`

Add a field:

```rust
pub struct Orf {
    ...
    pub start_score: f64, // default 1.0
}
```

Update `Orf::score()`:

```rust
pub fn score(&mut self, start_codons: &HashMap<Vec<u8>, f64>) {
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

### 5.2 CLI

New flag in `src/main.rs`:

```rust
/// Path to a learned start-site scoring model (JSON).
#[arg(long = "start-model", value_name = "FILE")]
start_model: Option<PathBuf>,
```

Flow inside `process_genome`:

1. After RBS/motif scoring and GC-frame `hold` computation (and optional dicodon scoring), if `start_model` is provided:
   - Load `StartModel`.
   - For each ORF, extract `StartSiteFeatures` and set `orf.start_score = model.score(&features)`.
2. Otherwise, leave `start_score = 1.0` for every ORF.
3. Proceed to `Orf::score()` as usual.

### 5.3 Python bindings

Add optional parameter to `phanotate()` and `find_orfs()`:

```rust
start_model: Option<&str> = None,
```

If provided, load the model and apply it before scoring.

---

## 6. Training pipeline

New script: `scripts/train_start_model.py`

Inputs:
- One or more GenBank files with annotated CDS features.
- A PHANOTATE-rs binary (or the Rust library) to enumerate candidate ORFs.

Steps:
1. For each GenBank file, extract the full sequence and the set of annotated `(start, stop, strand, frame)` gene coordinates.
2. Enumerate all candidate ORFs on that sequence with the correct translation table.
3. Build labels:
   - Positive: candidate ORF whose start coordinate matches an annotated start (within 3 bp on the same strand).
   - Negative: all other candidate ORFs on the same strand.
4. Extract the feature vector for each candidate ORF.
5. Fit a logistic regression with L2 regularization (scikit-learn).
6. Export JSON:
   ```json
   {
     "version": 1,
     "num_features": 11,
     "coeffs": [...],
     "mean": [...],
     "std": [...]
   }
   ```

The script is independent of the Rust runtime; it only produces the model file.

---

## 7. Files changed

| file | change |
|------|--------|
| `src/start_refiner.rs` | new module: features, model, JSON loader |
| `src/lib.rs` | re-export `pub mod start_refiner` |
| `src/orf.rs` | add `start_score`; update `score()` and all constructors/tests |
| `src/main.rs` | add `--start-model` flag; apply model when provided |
| `src/lib_python.rs` | add `start_model` parameter to `phanotate()` and `find_orfs()` |
| `scripts/train_start_model.py` | new training script |
| `tests/cli_tests.rs` | add regression + effect tests |
| `tests/test_python_bindings.py` | add Python binding test |

---

## 8. Testing

- Unit tests in `src/start_refiner.rs`:
  - feature vector length and normalization
  - `score()` returns a value in `[0.5, 2.0]`
  - JSON round-trip
- CLI tests:
  - `--start-model` run succeeds and output differs from default on `NC_001365.gb`.
  - Default mode (no `--start-model`) is byte-identical to before.
- Python tests:
  - `phanotate(..., start_model=path)` succeeds and differs from default.
- Manual evaluation:
  - Compare precision/recall on `NC_001365.gb` with and without the model.

---

## 9. Backwards compatibility

The feature is opt-in. Without `--start-model`, `start_score` is always `1.0` and all outputs are identical to the current behavior.

---

## 10. Future extensions

- Add more features (e.g., distance to nearest upstream gene, GC content near start, upstream 6-mer composition).
- Support ONNX export/import for consistency with the existing `ml` feature.
- Combine the start-site model with the weak-ORF filter (approach 2) in a single optional ML pipeline.
