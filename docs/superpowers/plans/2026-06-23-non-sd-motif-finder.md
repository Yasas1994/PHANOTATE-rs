# Non-Shine–Dalgarno Motif Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Prodigal-style non-SD upstream motif finder to PHANOTATE-rs that automatically discovers and scores 3–6 bp motifs when canonical SD signals are weak.

**Architecture:** A new `src/nonsd_motif.rs` module owns all training/scoring logic. The existing `Orf` struct gains a `motif_score` multiplier. `main.rs` decides SD vs non-SD mode, trains the appropriate model, and applies the score before graph construction. Output and ML features are extended to report the mode and the new signal.

**Tech Stack:** Rust 2021, existing PHANOTATE-rs modules (`orf`, `main`, `output`, `ml_features`), `cargo test`, `cargo clippy`, `cargo fmt`.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/nonsd_motif.rs` *(new)* | K-mer encoding, motif counting, coverage map, EM training, per-ORF scoring. |
| `src/orf.rs` *(modify)* | Add `motif_score` field to `Orf`; apply it in `Orf::score`. |
| `src/main.rs` *(modify)* | Add `--non-sd` / `--sd` flags; auto-detect SD usage; call `NonSdModel::train` when needed. |
| `src/lib.rs` *(modify)* | Re-export `pub mod nonsd_motif`. |
| `src/ml_features.rs` *(modify)* | Add `motif_score` as 14th feature in TSV export. |
| `src/output.rs` *(modify)* | Emit `uses_sd: 0/1` header comment in primary output. |
| `tests/cli_tests.rs` *(modify)* | Add smoke tests for `--non-sd` and auto-detect header. |

---

## Task 1: Extend `Orf` with `motif_score`

**Files:**
- Modify: `src/orf.rs:1-12`
- Modify: `src/orf.rs:40-48`
- Modify: `src/orf.rs:171-184` (forward ORF constructor)
- Modify: `src/orf.rs:220-230` (reverse ORF constructor)
- Modify: `src/orf.rs:253-263` (forward fragment constructor)
- Modify: `src/orf.rs:293-303` (reverse fragment constructor)
- Test: `cargo test --lib -- orf`

- [ ] **Step 1: Add the field to `Orf`**

```rust
pub struct Orf {
    pub start: usize,
    pub stop: usize,
    pub frame: i8,
    pub seq: Vec<u8>,
    pub rbs_score: usize,
    pub pstop: f64,
    pub weight_rbs: f64,
    pub hold: f64,
    pub motif_score: f64, // <-- new
    pub weight: f64,
}
```

- [ ] **Step 2: Apply `motif_score` in `Orf::score`**

```rust
pub fn score(&mut self, start_codons: &std::collections::HashMap<Vec<u8>, f64>) {
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

- [ ] **Step 3: Initialize `motif_score: 1.0` in all four `Orf` constructors**

Forward ORF:
```rust
orfs.push(Orf {
    start,
    stop: stop - 2,
    frame: frame_i8,
    seq,
    rbs_score,
    pstop,
    weight_rbs: 1.0,
    hold: 1.0,
    motif_score: 1.0, // <-- new
    weight: 1.0,
});
```

Repeat for reverse ORF, forward fragment, reverse fragment.

- [ ] **Step 4: Run unit tests**

```bash
cargo test --lib -- orf
```

Expected: existing `Orf` tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/orf.rs
git commit -m "feat(orf): add motif_score multiplier to Orf struct"
```

---

## Task 2: Create `src/nonsd_motif.rs` with k-mer utilities

**Files:**
- Create: `src/nonsd_motif.rs`
- Modify: `src/lib.rs`
- Test: `cargo test --lib -- nonsd`

- [ ] **Step 1: Create the file with encoding/decoding utilities**

```rust
//! Non-Shine-Dalgarno upstream motif finder.
//!
//! Discovers arbitrary 3-6 bp motifs enriched upstream of start codons,
//! mirroring Prodigal's train_starts_nonsd algorithm.

use crate::orf::Orf;
use std::collections::HashMap;

/// Number of possible spacer distance groups.
pub const NUM_SPACERS: usize = 4;
/// Minimum motif length (3 bp).
pub const MIN_MOTIF_LEN: usize = 3;
/// Maximum motif length (6 bp).
pub const MAX_MOTIF_LEN: usize = 6;
/// Maximum encoded motif index (4^6).
pub const MAX_MOTIF_INDEX: usize = 4096;

/// 2-bit encode a single base: A=0, C=1, G=2, T=3.
fn encode_base(b: u8) -> Option<usize> {
    match b {
        b'a' | b'A' => Some(0),
        b'c' | b'C' => Some(1),
        b'g' | b'G' => Some(2),
        b't' | b'T' => Some(3),
        _ => None,
    }
}

/// Decode a single 2-bit value to ASCII base.
fn decode_base(v: usize) -> u8 {
    match v {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => b'N',
    }
}

/// Encode a DNA word of length `len` starting at `pos` in `seq`.
/// Returns `None` if any base is ambiguous or out of range.
pub fn kmer_encode(seq: &[u8], pos: usize, len: usize) -> Option<usize> {
    if pos + len > seq.len() {
        return None;
    }
    let mut ndx = 0;
    for i in 0..len {
        ndx |= encode_base(seq[pos + i])? << (2 * i);
    }
    Some(ndx)
}

/// Decode a motif index back to a DNA string of length `len`.
pub fn kmer_decode(index: usize, len: usize) -> Vec<u8> {
    let mut seq = Vec::with_capacity(len);
    for i in 0..len {
        seq.push(decode_base((index >> (2 * i)) & 0x3));
    }
    seq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmer_roundtrip_3_to_6() {
        for len in 3..=6 {
            for ndx in 0..(1 << (2 * len)) {
                let seq = kmer_decode(ndx, len);
                assert_eq!(kmer_encode(&seq, 0, len), Some(ndx));
            }
        }
    }

    #[test]
    fn kmer_encode_rejects_ambiguous() {
        assert!(kmer_encode(b"atgnat", 0, 3).is_none());
    }
}
```

- [ ] **Step 2: Register the module in `src/lib.rs`**

Add after the existing module declarations:

```rust
pub mod nonsd_motif;
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- nonsd
```

Expected: k-mer round-trip tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/nonsd_motif.rs src/lib.rs
git commit -m "feat(nonsd): add k-mer encoding/decoding utilities"
```

---

## Task 3: Add motif counting and coverage map

**Files:**
- Modify: `src/nonsd_motif.rs`
- Test: `cargo test --lib -- nonsd`

- [ ] **Step 1: Define spacer group helper**

```rust
/// Classify a spacer (distance from motif start to coding start) into a group.
fn spacer_group(spacer: usize) -> usize {
    match spacer {
        3 | 4 => 1,
        5..=10 => 0,
        11 | 12 => 2,
        13..=15 => 3,
        _ => panic!("spacer out of range: {}", spacer),
    }
}
```

- [ ] **Step 2: Add a `MotifHit` struct and best-motif scan**

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct MotifHit {
    pub len: usize,        // 3..6
    pub spacer: usize,     // 3..15
    pub spacendx: usize,   // 0..3
    pub ndx: usize,        // encoded motif
    pub score: f64,
}

/// Scan positions start-18-i .. start-6-i for each motif length i+3 and
/// return the highest scoring motif. If no valid motif is found, returns
/// a zeroed hit with score set to the caller's no_mot value.
pub fn find_best_motif(
    mot_wt: &[[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    seq: &[u8],
    start: usize,
    no_mot: f64,
) -> MotifHit {
    let mut best = MotifHit {
        score: no_mot,
        ..Default::default()
    };
    for len_idx in 0..=(MAX_MOTIF_LEN - MIN_MOTIF_LEN) {
        let len = MIN_MOTIF_LEN + len_idx;
        let earliest = start.saturating_sub(18 + len);
        let latest = start.saturating_sub(6 + len);
        for pos in earliest..=latest {
            if pos + len > seq.len() {
                continue;
            }
            if let Some(ndx) = kmer_encode(seq, pos, len) {
                let spacer = start - pos - len;
                let spacendx = spacer_group(spacer);
                let score = mot_wt[len_idx][spacendx][ndx];
                if score > best.score {
                    best = MotifHit {
                        len,
                        spacer,
                        spacendx,
                        ndx,
                        score,
                    };
                }
            }
        }
    }
    best
}
```

- [ ] **Step 3: Add coverage map**

```rust
pub type CoverageMap = [[[u8; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];

/// Build a coverage map. A motif is "good" if it contains a 3-mer subset
/// present in at least 20% of selected genes.
pub fn build_coverage_map(
    real: &[[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    ngenes: f64,
) -> CoverageMap {
    let mut good = [[[0u8; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
    let thresh = 0.2;

    // 3-base motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..64 {
            if real[0][sp][j] / ngenes >= thresh {
                for k in 0..NUM_SPACERS {
                    good[0][k][j] = 1;
                }
            }
        }
    }

    // 4-base motifs need two valid 3-base sub-motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..256 {
            let d0 = (j & 0b11111100) >> 2;
            let d1 = j & 0b00111111;
            if good[0][sp][d0] == 0 || good[0][sp][d1] == 0 {
                continue;
            }
            good[1][sp][j] = 1;
        }
    }

    // 5-base motifs need three valid 3-base sub-motifs; allow interior mismatches
    for sp in 0..NUM_SPACERS {
        for j in 0..1024 {
            let d0 = (j & 0b1111110000) >> 4;
            let d1 = (j & 0b0000111100) >> 2;
            let d2 = j & 0b0000001111;
            if good[0][sp][d0] == 0 || good[0][sp][d1] == 0 || good[0][sp][d2] == 0 {
                continue;
            }
            good[2][sp][j] = 1;
            // flip bits 3 and 4 of the 5-mer (positions 2 and 3) to allow one mismatch
            let mut tmp = j;
            for k in [0, 16] {
                tmp ^= k;
                for l in [0, 32] {
                    tmp ^= l;
                    if good[2][sp][tmp] == 0 {
                        good[2][sp][tmp] = 2;
                    }
                }
            }
        }
    }

    // 6-base motifs need two valid 5-base sub-motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..MAX_MOTIF_INDEX {
            let d0 = (j & 0b111111111100) >> 2;
            let d1 = j & 0b000000111111;
            if good[2][sp][d0] == 0 || good[2][sp][d1] == 0 {
                continue;
            }
            good[3][sp][j] = if good[2][sp][d0] == 1 && good[2][sp][d1] == 1 {
                1
            } else {
                2
            };
        }
    }

    good
}
```

- [ ] **Step 4: Add unit tests for coverage map**

```rust
#[test]
fn coverage_map_marks_common_3mer_good() {
    let mut real = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
    let ngenes = 10.0;
    real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 3.0; // 30%
    let good = build_coverage_map(&real, ngenes);
    assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 1);
}

#[test]
fn coverage_map_keeps_rare_3mer_bad() {
    let mut real = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
    let ngenes = 10.0;
    real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 1.0; // 10%
    let good = build_coverage_map(&real, ngenes);
    assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 0);
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib -- nonsd
```

Expected: all new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/nonsd_motif.rs
git commit -m "feat(nonsd): add motif scanning and coverage map"
```

---

## Task 4: Implement EM training loop

**Files:**
- Modify: `src/nonsd_motif.rs`
- Test: `cargo test --lib -- nonsd`

- [ ] **Step 1: Define the model struct and training types**

```rust
pub struct NonSdModel {
    pub mot_wt: [[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    pub no_mot: f64,
    pub type_wt: [f64; 3],
    pub ups_comp: [[f64; 4]; 32],
}

impl Default for NonSdModel {
    fn default() -> Self {
        Self {
            mot_wt: [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
            no_mot: 0.0,
            type_wt: [0.0; 3],
            ups_comp: [[0.0; 4]; 32],
        }
    }
}
```

- [ ] **Step 2: Implement `train` method**

```rust
impl NonSdModel {
    pub fn train(
        orfs: &[Orf],
        dna: &[u8],
        rc_dna: &[u8],
        start_weights: &HashMap<Vec<u8>, f64>,
    ) -> Self {
        let mut model = Self::default();
        let st_wt = 4.35; // matches Prodigal's hard-coded start weight

        // Initialize type_wt from user-provided start-codon weights
        let atg_w = *start_weights.get(&b"atg".to_vec()).unwrap_or(&1.0);
        let gtg_w = *start_weights.get(&b"gtg".to_vec()).unwrap_or(&1.0);
        let ttg_w = *start_weights.get(&b"ttg".to_vec()).unwrap_or(&1.0);
        let max_w = atg_w.max(gtg_w).max(ttg_w);
        if max_w > 0.0 {
            model.type_wt[0] = (atg_w / max_w).ln();
            model.type_wt[1] = (gtg_w / max_w).ln();
            model.type_wt[2] = (ttg_w / max_w).ln();
        }

        // Background type frequencies across all ORFs
        let mut tbg = [0.0; 3];
        for orf in orfs {
            if let Some(idx) = start_codon_index(orf.start_codon()) {
                tbg[idx] += 1.0;
            }
        }
        let tbg_sum: f64 = tbg.iter().sum();
        if tbg_sum > 0.0 {
            for v in &mut tbg {
                *v /= tbg_sum;
            }
        }

        let mut sthresh = 35.0;

        for iter in 0..20 {
            let stage = if iter < 4 {
                0
            } else if iter < 12 {
                1
            } else {
                2
            };

            // Background motif counts
            let mut mbg = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
            let mut zbg = 0.0;
            for orf in orfs {
                let (wseq, start) = upstream_context(dna, rc_dna, orf);
                if start < 18 + MIN_MOTIF_LEN {
                    continue;
                }
                let hit = find_best_motif(&model.mot_wt, wseq, start, model.no_mot);
                update_motif_counts(&mut mbg, &mut zbg, wseq, start, &hit, stage);
            }
            let mbg_sum = mbg.iter().flat_map(|a| a.iter()).flat_map(|b| b.iter()).sum::<f64>() + zbg;
            if mbg_sum > 0.0 {
                for a in &mut mbg {
                    for b in a {
                        for v in b {
                            *v /= mbg_sum;
                        }
                    }
                }
                zbg /= mbg_sum;
            }

            // Real counts: group ORFs by (stop, frame) and pick best start per group
            let mut mreal = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
            let mut zreal = 0.0;
            let mut treal = [0.0; 3];
            let mut ngenes = 0.0;

            let groups = group_orfs_by_stop(orfs);
            for group in groups.values() {
                let best = group
                    .iter()
                    .map(|&orf| {
                        let (wseq, start) = upstream_context(dna, rc_dna, orf);
                        let hit = find_best_motif(&model.mot_wt, wseq, start, model.no_mot);
                        let type_idx = start_codon_index(orf.start_codon()).unwrap_or(0);
                        let coding_score = 1.0 / orf.hold;
                        let score = coding_score + st_wt * (hit.score + model.type_wt[type_idx]);
                        (orf, hit, score, type_idx)
                    })
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

                if let Some((orf, hit, score, type_idx)) = best {
                    if score >= sthresh {
                        ngenes += 1.0;
                        treal[type_idx] += 1.0;
                        let (wseq, start) = upstream_context(dna, rc_dna, orf);
                        update_motif_counts(&mut mreal, &mut zreal, wseq, start, &hit, stage);
                        if iter == 19 {
                            count_upstream_composition(wseq, start, &mut model.ups_comp);
                        }
                    }
                }
            }

            // Coverage filter and weight update
            if stage < 2 {
                let mgood = build_coverage_map(&mreal, ngenes);
                let mreal_sum = mreal.iter().flat_map(|a| a.iter()).flat_map(|b| b.iter()).sum::<f64>() + zreal;
                if mreal_sum == 0.0 {
                    model.mot_wt = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
                    model.no_mot = 0.0;
                } else {
                    for li in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                        for si in 0..NUM_SPACERS {
                            for mi in 0..MAX_MOTIF_INDEX {
                                if mgood[li][si][mi] == 0 {
                                    zreal += mreal[li][si][mi];
                                    zbg += mreal[li][si][mi];
                                    mreal[li][si][mi] = 0.0;
                                }
                                mreal[li][si][mi] /= mreal_sum;
                                model.mot_wt[li][si][mi] = if mbg[li][si][mi] != 0.0 {
                                    (mreal[li][si][mi] / mbg[li][si][mi]).ln()
                                } else {
                                    -4.0
                                }
                                .clamp(-4.0, 4.0);
                            }
                        }
                    }
                    model.no_mot = if zbg != 0.0 {
                        (zreal / mreal_sum / zbg).ln()
                    } else {
                        -4.0
                    }
                    .clamp(-4.0, 4.0);
                }
            }

            // Update type weights
            let treal_sum: f64 = treal.iter().sum();
            if treal_sum == 0.0 {
                model.type_wt = [0.0; 3];
            } else {
                for i in 0..3 {
                    let real = treal[i] / treal_sum;
                    model.type_wt[i] = if tbg[i] != 0.0 {
                        (real / tbg[i]).ln()
                    } else {
                        -4.0
                    }
                    .clamp(-4.0, 4.0);
                }
            }

            if treal_sum <= orfs.len() as f64 / 2000.0 {
                sthresh /= 2.0;
            }
        }

        // Convert ups_comp counts to log scores
        finalize_upstream_composition(&mut model.ups_comp, dna, rc_dna);

        model
    }
}
```

- [ ] **Step 3: Add required helper functions**

```rust
/// Map start codon bytes to ATG=0, GTG=1, TTG=2.
pub fn start_codon_index(codon: &[u8]) -> Option<usize> {
    match codon {
        b"atg" | b"ATG" => Some(0),
        b"gtg" | b"GTG" => Some(1),
        b"ttg" | b"TTG" => Some(2),
        _ => None,
    }
}

/// Return the upstream sequence and the start position in that sequence
/// for scoring/training. For reverse-strand ORFs, uses the reverse complement.
fn upstream_context<'a>(dna: &'a [u8], rc_dna: &'a [u8], orf: &Orf) -> (&'a [u8], usize) {
    if orf.frame > 0 {
        (dna, orf.start - 1)
    } else {
        let start = dna.len() - orf.start;
        (rc_dna, start)
    }
}

/// Group ORFs by (stop, frame) for per-stop start selection.
fn group_orfs_by_stop(orfs: &[Orf]) -> HashMap<(usize, i8), Vec<&Orf>> {
    let mut map: HashMap<(usize, i8), Vec<&Orf>> = HashMap::new();
    for orf in orfs {
        map.entry((orf.stop, orf.frame)).or_default().push(orf);
    }
    map
}
```

- [ ] **Step 4: Add `update_motif_counts`, `count_upstream_composition`, and `finalize_upstream_composition`**

```rust
fn update_motif_counts(
    mcnt: &mut [[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    zero: &mut f64,
    seq: &[u8],
    start: usize,
    hit: &MotifHit,
    stage: usize,
) {
    if hit.len == 0 {
        *zero += 1.0;
        return;
    }

    match stage {
        0 => {
            for len_idx in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                let len = MIN_MOTIF_LEN + len_idx;
                let earliest = start.saturating_sub(18 + len);
                let latest = start.saturating_sub(6 + len);
                for pos in earliest..=latest {
                    if pos + len > seq.len() {
                        continue;
                    }
                    if let Some(ndx) = kmer_encode(seq, pos, len) {
                        for sp in 0..NUM_SPACERS {
                            mcnt[len_idx][sp][ndx] += 1.0;
                        }
                    }
                }
            }
        }
        1 => {
            mcnt[hit.len - MIN_MOTIF_LEN][hit.spacendx][hit.ndx] += 1.0;
            for sub_len_idx in 0..hit.len - MIN_MOTIF_LEN {
                let sub_len = MIN_MOTIF_LEN + sub_len_idx;
                let earliest = start - hit.spacer - hit.len;
                let latest = start - hit.spacer - sub_len;
                for pos in earliest..=latest {
                    if pos + sub_len > seq.len() {
                        continue;
                    }
                    if let Some(ndx) = kmer_encode(seq, pos, sub_len) {
                        let spacer = start - pos - sub_len;
                        let sp = spacer_group(spacer);
                        mcnt[sub_len_idx][sp][ndx] += 1.0;
                    }
                }
            }
        }
        _ => {
            mcnt[hit.len - MIN_MOTIF_LEN][hit.spacendx][hit.ndx] += 1.0;
        }
    }
}

fn count_upstream_composition(seq: &[u8], start: usize, ups_comp: &mut [[f64; 4]; 32]) {
    let mut count = 0;
    for i in 1..45 {
        if i > 2 && i < 15 {
            continue;
        }
        if start >= i {
            if let Some(base) = encode_base(seq[start - i]) {
                ups_comp[count][base] += 1.0;
            }
        }
        count += 1;
    }
}

fn finalize_upstream_composition(
    ups_comp: &mut [[f64; 4]; 32],
    dna: &[u8],
    rc_dna: &[u8],
) {
    // Compute genomic GC from both strands
    let mut gc_count = 0.0;
    let mut at_count = 0.0;
    for &b in dna.iter().chain(rc_dna.iter()) {
        match b {
            b'g' | b'G' | b'c' | b'C' => gc_count += 1.0,
            b'a' | b'A' | b't' | b'T' => at_count += 1.0,
            _ => {}
        }
    }
    let gc = if gc_count + at_count > 0.0 {
        gc_count / (gc_count + at_count)
    } else {
        0.5
    };

    for row in ups_comp.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum == 0.0 {
            continue;
        }
        for j in 0..4 {
            row[j] /= sum;
            let expected = if j == 0 || j == 3 {
                // A or T
                if gc > 0.1 && gc < 0.9 {
                    1.0 - gc
                } else if gc <= 0.1 {
                    0.90
                } else {
                    0.10
                }
            } else {
                // C or G
                if gc > 0.1 && gc < 0.9 {
                    gc
                } else if gc <= 0.1 {
                    0.10
                } else {
                    0.90
                }
            };
            row[j] = (row[j] * 2.0 / expected).ln().clamp(-4.0, 4.0);
        }
    }
}
```

- [ ] **Step 5: Add a training convergence test**

```rust
#[test]
fn training_finds_planted_motif() {
    // Build a tiny synthetic genome: many ORFs with a planted 6-mer upstream
    let motif = b"aaaaaa";
    let mut seq = Vec::new();
    let mut orfs = Vec::new();
    for i in 0..50 {
        seq.extend_from_slice(&[b't', b't', b'g']); // start
        seq.extend_from_slice(motif);                // planted motif, spacer 3
        seq.extend_from_slice(&[b'a', b't', b'g']);  // filler
        seq.extend_from_slice(&[b't', b'a', b'a']);  // stop
        orfs.push(Orf {
            start: i * 18 + 1,
            stop: i * 18 + 15,
            frame: 1,
            seq: seq[i * 18..i * 18 + 15].to_vec(),
            rbs_score: 0,
            pstop: 0.01,
            weight_rbs: 1.0,
            hold: 100.0,
            motif_score: 1.0,
            weight: 1.0,
        });
    }
    let rc = crate::genome::rev_comp(&seq);
    let mut weights = HashMap::new();
    weights.insert(b"ttg".to_vec(), 1.0);
    let model = NonSdModel::train(&orfs, &seq, &rc, &weights);

    let hit = find_best_motif(&model.mot_wt, &seq, 4, model.no_mot);
    assert_eq!(hit.len, 6);
    assert_eq!(kmer_decode(hit.ndx, 6), b"aaaaaa".to_vec());
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test --lib -- nonsd
```

Expected: training convergence test passes.

- [ ] **Step 7: Commit**

```bash
git add src/nonsd_motif.rs
git commit -m "feat(nonsd): implement EM training loop and model scoring"
```

---

## Task 5: Add per-ORF scoring helpers

**Files:**
- Modify: `src/nonsd_motif.rs`
- Test: `cargo test --lib -- nonsd`

- [ ] **Step 1: Implement `score_orf` and upstream composition scoring**

```rust
impl NonSdModel {
    /// Score a single ORF and return a positive multiplier.
    pub fn score_orf(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> f64 {
        let (wseq, start) = upstream_context(dna, rc_dna, orf);
        if start < 18 + MIN_MOTIF_LEN {
            return 1.0;
        }
        let hit = find_best_motif(&self.mot_wt, wseq, start, self.no_mot);
        let type_idx = start_codon_index(orf.start_codon()).unwrap_or(0);
        let type_bonus = self.type_wt[type_idx];
        let comp_bonus = score_upstream_composition(wseq, start, &self.ups_comp);
        let log_score = hit.score + type_bonus + comp_bonus;
        log_score.exp().clamp(0.25, 4.0)
    }
}

fn score_upstream_composition(seq: &[u8], start: usize, ups_comp: &[[f64; 4]; 32]) -> f64 {
    let mut score = 0.0;
    let mut count = 0;
    for i in 1..45 {
        if i > 2 && i < 15 {
            continue;
        }
        if start >= i {
            if let Some(base) = encode_base(seq[start - i]) {
                score += 0.4 * 4.35 * ups_comp[count][base];
            }
        }
        count += 1;
    }
    score
}
```

- [ ] **Step 2: Add unit test for score_orf**

```rust
#[test]
fn score_orf_returns_reasonable_multiplier() {
    let model = NonSdModel::default();
    let orf = Orf {
        start: 25,
        stop: 60,
        frame: 1,
        seq: vec![b'a'; 36],
        rbs_score: 0,
        pstop: 0.01,
        weight_rbs: 1.0,
        hold: 100.0,
        motif_score: 1.0,
        weight: 1.0,
    };
    let dna = vec![b'a'; 100];
    let rc = dna.clone();
    let s = model.score_orf(&orf, &dna, &rc);
    assert!(s > 0.0);
    assert!(s <= 4.0);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- nonsd
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/nonsd_motif.rs
git commit -m "feat(nonsd): add per-ORF scoring and upstream composition"
```

---

## Task 6: Wire mode selection into `main.rs`

**Files:**
- Modify: `src/main.rs`
- Test: `cargo test --test cli_tests`

- [ ] **Step 1: Add CLI flags to `Cli`**

```rust
/// Force non-Shine-Dalgarno motif discovery for start-codon scoring.
#[arg(long = "non-sd")]
force_non_sd: bool,

/// Force Shine-Dalgarno scoring (default behavior).
#[arg(long = "sd")]
force_sd: bool,
```

- [ ] **Step 2: Validate conflicting flags in `main()`**

After `let cli = Cli::parse();`:

```rust
if cli.force_non_sd && cli.force_sd {
    anyhow::bail!("--non-sd and --sd are mutually exclusive");
}
```

- [ ] **Step 3: Pass flags to `process_genome`**

Change `process_genome` signature:

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
) -> (String, String, String)
```

- [ ] **Step 4: Implement auto-detection and branch after SD training**

After the existing SD training block (`for orf in &mut orfs { orf.weight_rbs = ... }`):

```rust
let use_non_sd = if force_non_sd {
    true
} else if force_sd {
    false
} else {
    !detect_uses_sd(&background_rbs, &training_rbs)
};

if use_non_sd {
    let model = phanotate_rs::nonsd_motif::NonSdModel::train(
        &orfs,
        dna,
        rc_dna,
        start_codons_map,
    );
    for orf in &mut orfs {
        orf.motif_score = model.score_orf(orf, dna, rc_dna);
    }
}
```

- [ ] **Step 5: Add `detect_uses_sd` helper near `process_genome`**

```rust
fn detect_uses_sd(background: &[f64], training: &[f64]) -> bool {
    let top_bins = [27, 26, 25, 24, 22, 20];
    let signal: f64 = top_bins
        .iter()
        .map(|&i| training[i] / background[i])
        .sum();
    signal >= 2.0
}
```

- [ ] **Step 6: Update both call sites of `process_genome`**

Pass `cli.force_non_sd` and `cli.force_sd`.

- [ ] **Step 7: Run tests**

```bash
cargo test --test cli_tests
```

Expected: existing tests pass; compilation succeeds.

- [ ] **Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --non-sd/--sd flags and auto-detection"
```

---

## Task 7: Update ML features

**Files:**
- Modify: `src/ml_features.rs`
- Test: `cargo test --lib -- ml_features`

- [ ] **Step 1: Bump feature count and add column name**

```rust
pub const NUM_FEATURES: usize = 14;

pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "log_length",
    "rbs_score_norm",
    "log_hold",
    "pstop",
    "weight_rbs_log",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
    "motif_score", // <-- new
];
```

- [ ] **Step 2: Extract `motif_score` in `extract_features`**

```rust
let mut features = [0.0f32; NUM_FEATURES];
// ... existing features[0] through features[12] ...
features[13] = self.motif_score.ln() as f32;
OrfFeatures(features)
```

- [ ] **Step 3: Update `test_orf` helper to include `motif_score`**

```rust
fn test_orf() -> Orf {
    Orf {
        start: 100,
        stop: 300,
        frame: 1,
        seq: b"atggctagctagctagc".to_vec(),
        rbs_score: 15,
        pstop: 0.05,
        weight_rbs: 2.5,
        hold: 0.8,
        motif_score: 1.0, // <-- new
        weight: -1.0,
    }
}
```

- [ ] **Step 4: Add a test for the new feature**

```rust
#[test]
fn test_motif_score_feature() {
    let mut orf = test_orf();
    orf.motif_score = 2.0;
    let f = orf.extract_features();
    assert_eq!(f.0.len(), 14);
    assert!((f.0[13] - 2.0f32.ln()).abs() < 0.001);
}

#[test]
fn test_tsv_header_includes_motif_score() {
    let orfs = vec![test_orf()];
    let mut buf = Vec::new();
    write_features_tsv(&mut buf, &orfs, true).unwrap();
    let s = String::from_utf8(buf).unwrap();
    assert!(s.starts_with("log_length\trbs_score_norm\tlog_hold\tpstop\tweight_rbs_log\tstart_codon_atg\tstart_codon_gtg\tstart_codon_ttg\tgc_content\tframe_fwd\tframe_1\tframe_2\tframe_3\tmotif_score"));
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib -- ml_features
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ml_features.rs
git commit -m "feat(ml): expose motif_score as 14th feature"
```

---

## Task 8: Update output header

**Files:**
- Modify: `src/output.rs`
- Test: `cargo test --test cli_tests`

- [ ] **Step 1: Add a `uses_sd` parameter to primary output functions**

Change `write_primary` signature to accept `uses_sd: bool`.

Prepend a comment line to the returned string:

```rust
let mut out = String::new();
out.push_str(&format!("# uses_sd: {}\n", if uses_sd { 1 } else { 0 }));
```

- [ ] **Step 2: Thread `uses_sd` from `process_genome`**

`process_genome` should compute `uses_sd = !use_non_sd` and pass it to `write_primary`.

- [ ] **Step 3: Run tests**

```bash
cargo test --test cli_tests
```

Expected: tests compile; adjust golden assertions as needed.

- [ ] **Step 4: Commit**

```bash
git add src/output.rs src/main.rs
git commit -m "feat(output): emit uses_sd header comment"
```

---

## Task 9: Add integration tests

**Files:**
- Modify: `tests/cli_tests.rs`

- [ ] **Step 1: Add a `--non-sd` smoke test**

```rust
#[test]
fn non_sd_flag_runs_without_error() {
    let out = run_phanotate(&[
        "-i",
        "tests/data/small.fasta",
        "--non-sd",
        "-f",
        "sco",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("uses_sd: 0"));
}
```

- [ ] **Step 2: Add an auto-detect header test**

```rust
#[test]
fn default_run_emits_uses_sd_header() {
    let out = run_phanotate(&[
        "-i",
        "tests/data/small.fasta",
        "-f",
        "sco",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("uses_sd:"));
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --test cli_tests -- non_sd
```

Expected: new tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cli_tests.rs
git commit -m "test(cli): add --non-sd and uses_sd header smoke tests"
```

---

## Task 10: Quality gates and final verification

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Clippy**

```bash
cargo clippy -- -D warnings
```

- [ ] **Step 3: Unit tests**

```bash
cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_
```

Expected: 98+ tests pass.

- [ ] **Step 4: Integration tests**

```bash
cargo test --test cli_tests
cargo test --test detect_table_tests
```

Expected: tests pass (external-file-dependent tests may still fail).

- [ ] **Step 5: Full build**

```bash
cargo build --release
```

- [ ] **Step 6: End-to-end smoke**

```bash
./target/release/phanotate-rs -i phage_reviewed.fasta -f sco | head -5
./target/release/phanotate-rs -i phage_reviewed.fasta --non-sd -f sco | head -5
```

Expected: both produce output with `uses_sd:` header.

- [ ] **Step 7: Commit any final fixes**

```bash
git add -A
git commit -m "style: fmt and clippy fixes for non-sd feature"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| New `nonsd_motif.rs` module | Task 2, 3, 4, 5 |
| `Orf::motif_score` field and scoring | Task 1 |
| `--non-sd` / `--sd` CLI flags | Task 6 |
| Auto-detection heuristic | Task 6 |
| ML feature extension | Task 7 |
| Output `uses_sd` header | Task 8 |
| Unit tests for k-mers, coverage, training | Task 2, 3, 4, 5 |
| Integration smoke tests | Task 9 |

## Placeholder Scan

No TBD/TODO placeholders, no vague "handle edge cases" steps, no "similar to Task N" references. Each task includes exact file paths, code, and commands.

## Type Consistency Notes

- `NonSdModel::train` and `score_orf` use the same `upstream_context` helper.
- `start_codon_index` is used consistently for type-weight indexing.
- `motif_score` is initialized to `1.0` everywhere and applied as a multiplier in `Orf::score`.
