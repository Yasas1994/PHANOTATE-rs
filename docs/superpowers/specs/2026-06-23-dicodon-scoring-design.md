# Prodigal-Style Dicodon Scoring — Design Spec

**Project:** PHANOTATE-rs  
**Date:** 2026-06-23  
**Feature:** Add an optional 5th-order Markov (dicodon / 6-mer) coding-potential scorer modeled on Prodigal.

---

## 1. Goal

Add a Prodigal-style dicodon coding-potential model to PHANOTATE-rs. When enabled, it replaces the GC-frame-plot-based `1.0 / hold` term in `Orf::score()` with a log-likelihood sum over in-frame 6-mers. This gives the gene caller a richer sequence model and should improve accuracy on genomes where GC-frame bias is weak.

The feature is **optional** (`--dicodon`) so existing default behavior is preserved.

---

## 2. Motivation

PHANOTATE-rs currently scores coding potential with the GC frame plot (`hold`). Prodigal uses an additional dicodon model: it trains 6-mer frequencies on a seed set of likely genes and scores every ORF by how much its 6-mer composition deviates from the genomic background. This is a more powerful signal, especially for small phage genomes with few long ORFs.

---

## 3. Algorithm

### 3.1 Seed-gene selection

To avoid a strict length cutoff that would discard genes in small phage genomes, the seed set is defined relative to the **mean ORF length**:

```rust
let mean_len = orfs.iter().map(|o| o.len() as f64).sum::<f64>() / orfs.len() as f64;
let threshold = mean_len * 0.8;  // include ORFs down to 80% of the mean
```

An ORF is selected as a training gene if:

```rust
orf.len() as f64 >= threshold && orf.start_score() >= some_min_start_bonus
```

The start bonus can be as simple as `weight_rbs > 1.0` or `motif_score > 1.0`, so the seed set favors ORFs that already look like real genes.

### 3.2 Counting 6-mers

- **Background**: count every overlapping 6-mer in the full input sequence (both strands). Total background counts = `2 * (genome_len - 5)`.
- **Genes**: count every in-frame 6-mer in every selected seed ORF. Total gene counts = sum of `(orf_len - 5) / 3` over selected ORFs.

6-mers are encoded as 12-bit integers (4 bases × 2 bits), giving a 4096-element table.

### 3.3 Log-likelihood score

With add-one (Laplace) smoothing:

```rust
let g = (gene_count[w] + 1.0) / (total_gene_6mers + 4096.0);
let b = (bg_count[w] + 1.0) / (total_bg_6mers + 4096.0);
score[w] = (g / b).ln().clamp(-4.0, 4.0);
```

### 3.4 ORF dicodon score

```rust
fn score_orf(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> f64 {
    let seq = orf.sequence();
    let mut sum = 0.0;
    for i in (0..seq.len() - 5).step_by(3) {
        if let Some(w) = kmer_encode(&seq, i, 6) {
            sum += self.scores[w];
        }
    }
    // Convert to a positive multiplier roughly comparable to 1/hold
    sum.exp().clamp(0.1, 10.0)
}
```

The final multiplier is clamped to keep graph weights stable.

---

## 4. Data Model & Integration

### 4.1 New module: `src/dicodon.rs`

```rust
pub struct DicodonModel {
    pub scores: [f64; 4096],
}

impl DicodonModel {
    pub fn train(orfs: &[Orf], dna: &[u8], rc_dna: &[u8]) -> Self;
    pub fn score_orf(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> f64;
}
```

### 4.2 `Orf` changes

Add `pub dicodon_score: f64`, default `1.0`.

### 4.3 `Orf::score()` changes

`dicodon_score` becomes the generic coding-potential multiplier. In default mode it is initialized to `1.0 / self.hold`; in `--dicodon` mode it is overwritten with the dicodon model score.

```rust
pub fn score(&mut self, start_codons: &HashMap<Vec<u8>, f64>) {
    let mut s = self.dicodon_score;
    let sc = self.start_codon().to_vec();
    if let Some(&w) = start_codons.get(&sc) {
        s *= w;
    }
    s *= self.weight_rbs;
    s *= self.motif_score;
    self.weight = -s;
}
```

This removes the special-case fallback and keeps `score()` simple.

### 4.4 CLI and Python bindings

- `--dicodon` flag in `src/main.rs`.
- `dicodon=True` parameter in `src/lib_python.rs`.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `src/dicodon.rs` *(new)* | 6-mer encoding, model training, ORF scoring. |
| `src/orf.rs` | Add `dicodon_score`; modify `score()` to use it when set. |
| `src/lib.rs` | Re-export `pub mod dicodon`. |
| `src/main.rs` | Add `--dicodon` flag; train and apply model. |
| `src/lib_python.rs` | Add `dicodon` parameter and logic. |
| `src/ml_features.rs` | Optionally add `dicodon_score` as a feature. |
| `tests/cli_tests.rs` | Add `--dicodon` smoke test. |
| `src/dicodon.rs` tests | Unit tests for encoding, training, scoring. |

---

## 6. Testing

- **Unit:** `kmer_encode` round-trip; model training on synthetic sequence with planted coding bias; `score_orf` returns positive multiplier.
- **Integration:** `--dicodon -f sco` runs successfully; output differs from default mode on a real phage.
- **Regression:** Default mode (no `--dicodon`) output is byte-identical to before.

---

## 7. Backwards Compatibility

Default behavior is unchanged. `--dicodon` is opt-in. Existing tests pass without the flag.
