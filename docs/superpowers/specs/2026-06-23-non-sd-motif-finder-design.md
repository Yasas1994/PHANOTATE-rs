# Non-Shine–Dalgarno Upstream Motif Finder — Design Spec

**Project:** PHANOTATE-rs  
**Date:** 2026-06-23  
**Feature:** Add a Prodigal-style non-SD motif discovery mode for genomes that do not use canonical Shine–Dalgarno ribosome-binding sites.

---

## 1. Goals

- Detect and score arbitrary 3–6 bp upstream motifs enriched in front of real start codons when SD signals are weak or absent.
- Preserve existing SD-based behavior by default.
- Support automatic detection, manual non-SD override, and manual SD override.
- Integrate cleanly with the existing `Orf` scoring pipeline, graph construction, and output formatters.
- Expose the new signal to the optional ML feature extractor.

---

## 2. Background

PHANOTATE-rs currently scores every ORF with `score_rbs`, a byte-pattern matcher that looks for canonical SD motifs (`ggagga`, `ggagg`, `agga`, etc.) in the 21 nt upstream of a start codon. The resulting integer score (0–27) is converted into a `weight_rbs` multiplier from the ratio of training to background frequencies.

Some phages and bacteria lack strong SD motifs. Prodigal handles this with an unsupervised non-SD motif finder (`train_starts_nonsd` in `node.c`) that discovers the most enriched upstream 3–6 mer and uses it for start-codon scoring instead of the SD model.

---

## 3. High-Level Architecture

A new module, `src/nonsd_motif.rs`, owns all non-SD training and scoring logic. The rest of the pipeline is modified only where it must branch between SD and non-SD modes.

```text
Input genome
    │
    ▼
Find ORFs (existing)
    │
    ▼
Train SD model (existing background_rbs / training_rbs)
    │
    ▼
Mode decision: auto-detect, --non-sd, or --sd
    │
    ├── SD mode ──► apply existing weight_rbs, motif_score = 1.0
    │
    └── Non-SD mode ──► train NonSdModel
                       score each Orf.motif_score
    │
    ▼
Orf::score() = heuristic * weight_rbs * motif_score
    │
    ▼
Graph + shortest path (unchanged)
    │
    ▼
Output with uses_sd header comment
```

---

## 4. New Module: `src/nonsd_motif.rs`

### 4.1 Public API

```rust
pub struct NonSdModel {
    /// motif weights: [length 3-6][spacer group 0-3][encoded motif]
    pub mot_wt: [[[f64; 4096]; 4]; 4],
    /// weight for the "no motif" case
    pub no_mot: f64,
    /// ATG/GTG/TTG log weights
    pub type_wt: [f64; 3],
    /// upstream base-composition weights [position 0-31][base 0-3]
    pub ups_comp: [[f64; 4]; 32],
}

impl NonSdModel {
    /// Train the model from a set of putative ORFs.
    pub fn train(
        orfs: &[Orf],
        dna: &[u8],
        rc_dna: &[u8],
        start_weights: &HashMap<Vec<u8>, f64>,
    ) -> Self;

    /// Score a single ORF's upstream region.
    pub fn score_orf(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> f64;
}
```

### 4.2 Internal helpers

| Function | Purpose |
|----------|---------|
| `kmer_encode(seq, pos, len) -> usize` | 2-bit encode a DNA word of length 3–6. |
| `kmer_decode(index, len) -> Vec<u8>` | Decode for diagnostics/tests. |
| `find_best_motif(model, dna, rc_dna, orf) -> MotifHit` | Scan upstream 6–18 bp and return best (len, spacer, score). |
| `update_motif_counts(..., stage)` | Accumulate background / real motif counts per stage. |
| `build_coverage_map(real_counts, ngenes, stage) -> CoverageMap` | Mark motifs good if a 3-mer subset appears in ≥20% of genes. |
| `count_upstream_composition(...)` | Train `ups_comp`. |
| `score_upstream_composition(...)` | Score -1/-2 and -15/-44 composition. |

### 4.3 Training algorithm

Mirror Prodigal's `train_starts_nonsd`:

1. **Background type frequencies** — count ATG/GTG/TTG among all candidate starts.
2. **20 EM iterations** split into three stages:
   - **Stage 0** (iterations 0–3): count all 3–6 bp motifs in the upstream window for all spacer groups.
   - **Stage 1** (iterations 4–11): count only the best motif and all its sub-motifs.
   - **Stage 2** (iterations 12–19): count only the single best motif.
3. Each iteration:
   - **Background pass**: for every candidate `Orf`, find its best upstream motif and accumulate into background counts.
   - **Gene pass**: group `Orf`s by `(stop, frame)` and select the highest-scoring start in each group (mirroring Prodigal's per-stop start selection). Accumulate its motif into real counts.
   - Apply coverage filter (≥20% gene presence via 3-mer subset).
   - Update `mot_wt` as log-likelihood ratios, clamped to `[-4.0, 4.0]`.
   - Update `type_wt` as log-likelihood ratios, clamped to `[-4.0, 4.0]`.
4. After training, compute `ups_comp` from the final selected gene set.

### 4.4 Spacer groups

| Group | Spacer range |
|-------|--------------|
| 0 | 5–10 bp |
| 1 | 3–4 bp |
| 2 | 11–12 bp |
| 3 | 13–15 bp |

### 4.5 Scoring an ORF

```rust
let start_type = start_codon_index(orf.start_codon()); // ATG=0, GTG=1, TTG=2
let type_bonus = self.type_wt[start_type];
let motif = find_best_motif(self, dna, rc_dna, orf);
let motif_bonus = motif.score; // or self.no_mot
let comp_bonus = score_upstream_composition(self, dna, rc_dna, orf);
let log_score = type_bonus + motif_bonus + comp_bonus;
// Convert log-likelihood sum to a positive multiplier, then clamp
// to keep graph weights stable for the shortest-path solver.
log_score.exp().clamp(0.25, 4.0)
```

The returned multiplier is stored in `Orf::motif_score` and applied in `Orf::score()`. The clamp range (0.25–4.0) mirrors the `[-4, 4]` log-likelihood bounds and may be tuned during validation.

---

## 5. Changes to Existing Code

### 5.1 `src/orf.rs`

Add a field to `Orf`:

```rust
pub motif_score: f64,
```

Initialize to `1.0` in all constructors.

Modify `Orf::score`:

```rust
pub fn score(&mut self, start_codons: &HashMap<Vec<u8>, f64>) {
    let mut s = 1.0 / self.hold;
    let sc = self.start_codon().to_vec();
    if let Some(&w) = start_codons.get(&sc) {
        s *= w;
    }
    s *= self.weight_rbs;
    s *= self.motif_score; // <-- new
    self.weight = -s;
}
```

`score_hybrid` is unaffected because it calls `score()` first.

### 5.2 `src/main.rs`

Add CLI flags:

```rust
/// Force non-Shine-Dalgarno motif discovery for start-codon scoring.
#[arg(long = "non-sd")]
force_non_sd: bool,

/// Force Shine-Dalgarno scoring (default behavior).
#[arg(long = "sd")]
force_sd: bool,
```

Validate no conflict:

```rust
if cli.force_non_sd && cli.force_sd {
    anyhow::bail!("--non-sd and --sd are mutually exclusive");
}
```

Inside `process_genome`, after existing SD training:

```rust
let use_non_sd = if force_non_sd {
    true
} else if force_sd {
    false
} else {
    !detect_uses_sd(&training_rbs, &background_rbs)
};

if use_non_sd {
    let model = nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);
    for orf in &mut orfs {
        orf.motif_score = model.score_orf(orf, dna, rc_dna);
    }
}
```

### 5.3 `src/lib.rs`

Add:

```rust
pub mod nonsd_motif;
```

### 5.4 `src/output.rs`

Prepend a header comment to all primary outputs:

```text
# phanotate-rs v0.1.3
# uses_sd: 0
```

Use `uses_sd: 1` in SD mode. For GenBank output, place this in the `COMMENT` field; for GFF3/SCO, as a `#` comment line.

### 5.5 `src/ml_features.rs`

Expand the feature vector from 13 to 14 features by appending:

```rust
motif_score: f64,
```

In SD mode this value is `0.0`. In non-SD mode it is the raw best-motif log score.

---

## 6. Auto-Detection Heuristic

`detect_uses_sd` examines the trained SD weight distribution. A simple, conservative rule:

```rust
/// Returns true if the genome appears to use canonical SD motifs.
fn detect_uses_sd(training: &[f64; 28], background: &[f64; 28]) -> bool {
    // If the top canonical SD bins are not enriched over background,
    // assume the genome does not use SD.
    let top_bins = [27, 26, 25, 24, 22, 20];
    let signal: f64 = top_bins.iter().map(|&i| training[i] / background[i]).sum();
    signal >= 2.0 // threshold to be tuned on real genomes
}
```

The threshold will be tuned on a small set of phage genomes with known SD vs non-SD biology. The user can always override with `--sd` or `--non-sd`.

---

## 7. Testing Strategy

### 7.1 Unit tests in `nonsd_motif.rs`

- `kmer_encode` / `kmer_decode` round-trip for all 3–6-mers.
- `build_coverage_map` marks a synthetic motif good when its 3-mer is present in ≥20% of genes.
- Training converges on a planted 6-mer motif in a synthetic sequence with no SD signal.

### 7.2 Integration tests in `tests/cli_tests.rs`

- Run on a real phage FASTA with `--non-sd` and verify output is produced.
- Golden-file regression: default run (SD) output is byte-identical to before.
- Auto-detect smoke test: run without flags and check the `uses_sd` header line is present.

### 7.3 Quality gates

- `cargo test`
- `cargo clippy -- -D warnings`
- `cargo fmt --check`

---

## 8. Open Questions / Follow-ups

1. **Auto-detect threshold**: needs calibration on real SD and non-SD phage genomes.
2. **Output compatibility**: adding a header comment may break very strict golden-file parsers. Existing tests will be updated if necessary.
3. **ML model retraining**: the new `motif_score` feature is only useful after retraining an ONNX model; the heuristic path works immediately.

---

## 9. Decisions Made

| Decision | Rationale |
|----------|-----------|
| New module `nonsd_motif.rs` | Keeps `orf.rs` focused; follows existing modular style. |
| Full Prodigal-style 3-stage EM | Maximizes accuracy and biological fidelity. |
| `--non-sd` / `--sd` flags | Gives users full control plus auto-detect default. |
| `motif_score` replaces `weight_rbs` in non-SD mode | Avoids double-counting weak SD signal. |
| Add `uses_sd` header comment | Low-cost observability. |
| Add `motif_score` to ML features | Keeps ML path informed of the new signal. |
