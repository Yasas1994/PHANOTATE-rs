# Prodigal-Style RBS Scanner with Non-SD Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--prodigal-rbs` mode to PHANOTATE-rs that scores starts with a Prodigal-style SD scanner and falls back to the existing non-SD motif finder for ORFs without SD motifs.

**Architecture:** A new `src/rbs_scanner.rs` module hosts the legacy matcher and a Rust port of Prodigal's `shine_dalgarno_exact` / `shine_dalgarno_mm`. The CLI flag triggers per-genome training/background ratio scoring plus non-SD fallback in `process_genome`; the Python API mirrors it.

**Tech Stack:** Rust 2021, `clap`, `pyo3`, existing PHANOTATE-rs modules, `cargo test`, `pytest`.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/rbs_scanner.rs` | New module: legacy RBS matcher, Prodigal SD scanner, bin labels, helpers. |
| `src/orf.rs` | Remove inline `score_rbs`/`detect_rbs_motif`; re-export from `rbs_scanner.rs`; make `get_rbs` reusable from `main.rs`. |
| `src/lib.rs` | Re-export `rbs_scanner` module. |
| `src/main.rs` | Add `--prodigal-rbs` CLI flag; wire background/training ratios and fallback logic. |
| `src/lib_python.rs` | Add `prodigal_rbs` parameter to `phanotate`/`find_orfs`; mirror CLI logic. |
| `tests/cli_tests.rs` | Add integration tests for `--prodigal-rbs` and mutual exclusivity. |
| `tests/test_python_bindings.py` | Add Python binding tests for `prodigal_rbs=True`. |

---

## Task 1: Create `src/rbs_scanner.rs` and move the legacy matcher

**Files:**
- Create: `src/rbs_scanner.rs`
- Modify: `src/orf.rs:375-782` and `src/orf.rs:789-~1010` (delete legacy functions)
- Modify: `src/lib.rs`
- Test: `cargo test --lib -- rbs`

- [ ] **Step 1: Create `src/rbs_scanner.rs` with the legacy matcher**

```rust
//! RBS motif scanning.
//!
//! Contains the legacy PHANOTATE pattern matcher and a Prodigal-style
//! Shine-Dalgarno consensus scanner with mismatch support.

/// Number of RBS score bins (0 = no motif, 1-27 = increasing SD signal).
pub const NUM_RBS_BINS: usize = 28;

/// Legacy PHANOTATE Shine-Dalgarno likelihood score.
/// Replicates the Python reference implementation exactly.
/// `seq` is the 21-nt upstream window (original orientation).  The function
/// reverses it internally, matching the original convention.
pub fn score_rbs_legacy(seq: &[u8]) -> usize {
    // Copy the entire body of src/orf.rs score_rbs here, including the
    // in_range closure and final return 0; rename only the function.
}

/// Detect the matching Shine-Dalgarno motif in the upstream window.
/// Mirrors the priority order and position ranges of `score_rbs_legacy`.
pub fn detect_rbs_motif_legacy(seq: &[u8]) -> Option<String> {
    // Copy the entire body of src/orf.rs detect_rbs_motif here;
    // rename only the function.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_no_motif() {
        assert_eq!(score_rbs_legacy(b"aaaaaaaaaaaaaaaaaaaaa"), 0);
        assert_eq!(detect_rbs_motif_legacy(b"aaaaaaaaaaaaaaaaaaaaa"), None);
    }

    #[test]
    fn legacy_aggagg_detected() {
        // 21-nt window ending in AGGAGG 5-10 bp upstream
        let seq = b"aaaaaaaaaaaaggaggaaaa"; // reversed AGGAGG at positions 3-9
        assert!(score_rbs_legacy(seq) > 0);
        assert_eq!(detect_rbs_motif_legacy(seq), Some("AGGAGG".to_string()));
    }
}
```

- [ ] **Step 2: Re-export in `src/lib.rs`**

Add after the existing module declarations:

```rust
pub mod rbs_scanner;
```

- [ ] **Step 3: Remove legacy functions from `src/orf.rs` and alias them**

Delete the old `score_rbs` (lines ~388-782) and `detect_rbs_motif` (lines ~789-~1010).

Add at the top of `src/orf.rs`:

```rust
pub use crate::rbs_scanner::{
    detect_rbs_motif_legacy as detect_rbs_motif,
    score_rbs_legacy as score_rbs,
};
```

Existing call sites and tests in `src/orf.rs` keep using `score_rbs` and `detect_rbs_motif` unchanged.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib -- rbs
```

Expected: legacy tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rbs_scanner.rs src/orf.rs src/lib.rs
git commit -m "refactor(rbs): move legacy RBS matcher into src/rbs_scanner.rs"
```

---

## Task 2: Implement Prodigal-style exact SD scanner

**Files:**
- Modify: `src/rbs_scanner.rs`
- Test: `cargo test --lib -- rbs`

- [ ] **Step 1: Add exact-match scanner helper**

```rust
/// Helper: 2-bit-like base check.  Only A/C/G/T are valid.
fn is_base(b: u8, target: u8) -> bool {
    b.eq_ignore_ascii_case(&target)
}

/// Score a single position against the AGGAGG consensus.
/// Position indices are 0-based from the 5' end of the motif.
fn consensus_match_value(pos: usize, b: u8) -> f64 {
    if pos % 3 == 0 && is_base(b, b'A') {
        2.0
    } else if pos % 3 != 0 && is_base(b, b'G') {
        3.0
    } else {
        -10.0
    }
}

/// Convert spacer distance to Prodigal's dis_flag for exact matches.
fn exact_dis_flag(rdis: usize, motif_len: usize) -> usize {
    if rdis < 5 && motif_len < 5 {
        2
    } else if rdis < 5 && motif_len >= 5 {
        1
    } else if rdis > 10 && rdis <= 12 && motif_len < 5 {
        1
    } else if rdis > 10 && rdis <= 12 && motif_len >= 5 {
        2
    } else if rdis >= 13 {
        3
    } else {
        0
    }
}

/// Port of Prodigal's shine_dalgarno_exact.
/// `seq` is the upstream window in original orientation.
/// `pos` is the candidate motif start within `seq`.
/// `start` is the index one past the upstream window (start codon position).
fn shine_dalgarno_exact(seq: &[u8], pos: usize, start: usize) -> usize {
    let limit = (6usize).min(start.saturating_sub(4).saturating_sub(pos));
    if limit < 3 {
        return 0;
    }

    let mut match_values = [-10.0f64; 6];
    for i in 0..limit {
        match_values[i] = consensus_match_value(i, seq[pos + i]);
    }

    let mut max_val = 0usize;
    for len in (3..=limit).rev() {
        for offset in 0..=limit - len {
            let mut cur_ctr = -2.0;
            let mut mism = 0;
            for k in offset..offset + len {
                cur_ctr += match_values[k];
                if match_values[k] < 0.0 {
                    mism += 1;
                }
            }
            if mism > 0 {
                continue;
            }
            let rdis = start.saturating_sub(pos + offset + len);
            if rdis > 15 || cur_ctr < 6.0 {
                continue;
            }
            let dis_flag = exact_dis_flag(rdis, len);

            let cur_val = exact_bin(cur_ctr, dis_flag);
            if cur_val > max_val {
                max_val = cur_val;
            }
        }
    }
    max_val
}

fn exact_bin(cur_ctr: f64, dis_flag: usize) -> usize {
    // Matches Prodigal's exact-bin mapping.
    if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 2 {
        1
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 3 {
        2
    } else if ((cur_ctr - 8.0).abs() < f64::EPSILON || (cur_ctr - 9.0).abs() < f64::EPSILON) && dis_flag == 3 {
        3
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 1 {
        6
    } else if ((cur_ctr - 11.0).abs() < f64::EPSILON || (cur_ctr - 12.0).abs() < f64::EPSILON || (cur_ctr - 14.0).abs() < f64::EPSILON) && dis_flag == 3 {
        10
    } else if ((cur_ctr - 8.0).abs() < f64::EPSILON || (cur_ctr - 9.0).abs() < f64::EPSILON) && dis_flag == 2 {
        11
    } else if ((cur_ctr - 8.0).abs() < f64::EPSILON || (cur_ctr - 9.0).abs() < f64::EPSILON) && dis_flag == 1 {
        12
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 0 {
        13
    } else if (cur_ctr - 8.0).abs() < f64::EPSILON && dis_flag == 0 {
        15
    } else if (cur_ctr - 9.0).abs() < f64::EPSILON && dis_flag == 0 {
        16
    } else if (cur_ctr - 11.0).abs() < f64::EPSILON && dis_flag == 2 {
        20
    } else if (cur_ctr - 11.0).abs() < f64::EPSILON && dis_flag == 1 {
        21
    } else if (cur_ctr - 11.0).abs() < f64::EPSILON && dis_flag == 0 {
        22
    } else if (cur_ctr - 12.0).abs() < f64::EPSILON && dis_flag == 2 {
        20
    } else if (cur_ctr - 12.0).abs() < f64::EPSILON && dis_flag == 1 {
        23
    } else if (cur_ctr - 12.0).abs() < f64::EPSILON && dis_flag == 0 {
        24
    } else if (cur_ctr - 14.0).abs() < f64::EPSILON && dis_flag == 2 {
        25
    } else if (cur_ctr - 14.0).abs() < f64::EPSILON && dis_flag == 1 {
        26
    } else if (cur_ctr - 14.0).abs() < f64::EPSILON && dis_flag == 0 {
        27
    } else {
        0
    }
}

/// Initial version: exact matches only. Will be extended in Task 3.
pub fn score_rbs_prodigal(seq: &[u8]) -> usize {
    let start = seq.len();
    let mut best = 0usize;
    for pos in 0..=start.saturating_sub(4) {
        best = best.max(shine_dalgarno_exact(seq, pos, start));
    }
    best
}
```

- [ ] **Step 2: Add unit tests for exact scanner**

```rust
#[cfg(test)]
mod prodigal_tests {
    use super::*;

    fn upstream_with_motif(motif: &[u8], spacer: usize) -> Vec<u8> {
        let mut seq = vec![b'a'; 21];
        let motif_start = 21 - spacer - motif.len();
        seq[motif_start..motif_start + motif.len()].copy_from_slice(motif);
        seq
    }

    #[test]
    fn exact_aggagg_5_10bp_is_bin_27() {
        let seq = upstream_with_motif(b"AGGAGG", 6);
        assert_eq!(score_rbs_prodigal(&seq), 27);
    }

    #[test]
    fn exact_aggagg_11_12bp_is_bin_25() {
        let seq = upstream_with_motif(b"AGGAGG", 11);
        assert_eq!(score_rbs_prodigal(&seq), 25);
    }

    #[test]
    fn exact_aggagg_3_4bp_is_bin_26() {
        let seq = upstream_with_motif(b"AGGAGG", 4);
        assert_eq!(score_rbs_prodigal(&seq), 26);
    }

    #[test]
    fn exact_aggagg_13_15bp_is_bin_10() {
        let seq = upstream_with_motif(b"AGGAGG", 14);
        assert_eq!(score_rbs_prodigal(&seq), 10);
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- rbs
```

Expected: new exact-scanner tests pass; legacy tests still pass.

- [ ] **Step 4: Commit**

```bash
git add src/rbs_scanner.rs
git commit -m "feat(rbs): add Prodigal exact SD scanner"
```

---

## Task 3: Implement Prodigal-style mismatch SD scanner

**Files:**
- Modify: `src/rbs_scanner.rs`
- Test: `cargo test --lib -- rbs`

- [ ] **Step 1: Add mismatch scanner helper**

```rust
fn consensus_match_value_mm(pos: usize, b: u8) -> f64 {
    if pos % 3 == 0 {
        if is_base(b, b'A') {
            2.0
        } else {
            -3.0
        }
    } else if is_base(b, b'G') {
        3.0
    } else {
        -2.0
    }
}

fn mm_dis_flag(rdis: usize) -> usize {
    if rdis < 5 {
        1
    } else if rdis > 10 && rdis <= 12 {
        2
    } else if rdis >= 13 {
        3
    } else {
        0
    }
}

fn shine_dalgarno_mm(seq: &[u8], pos: usize, start: usize) -> usize {
    let limit = (6usize).min(start.saturating_sub(4).saturating_sub(pos));
    if limit < 5 {
        return 0;
    }

    let mut match_values = [-10.0f64; 6];
    for i in 0..limit {
        match_values[i] = consensus_match_value_mm(i, seq[pos + i]);
    }

    let mut max_val = 0usize;
    for len in (5..=limit).rev() {
        for offset in 0..=limit - len {
            let mut cur_ctr = -2.0;
            let mut mism = 0;
            for k in offset..offset + len {
                cur_ctr += match_values[k];
                if match_values[k] < 0.0 {
                    mism += 1;
                    // Penalize mismatches in first/last two positions
                    if k <= offset + 1 || k >= offset + len - 2 {
                        cur_ctr -= 10.0;
                    }
                }
            }
            if mism != 1 {
                continue;
            }
            let rdis = start.saturating_sub(pos + offset + len);
            if rdis > 15 || cur_ctr < 6.0 {
                continue;
            }
            let dis_flag = mm_dis_flag(rdis);

            let cur_val = mm_bin(cur_ctr, dis_flag);
            if cur_val > max_val {
                max_val = cur_val;
            }
        }
    }
    max_val
}

fn mm_bin(cur_ctr: f64, dis_flag: usize) -> usize {
    if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 3 {
        2
    } else if ((cur_ctr - 6.0).abs() < f64::EPSILON || (cur_ctr - 7.0).abs() < f64::EPSILON) && dis_flag == 3 {
        2
    } else if (cur_ctr - 9.0).abs() < f64::EPSILON && dis_flag == 3 {
        3
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 2 {
        4
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 1 {
        5
    } else if (cur_ctr - 6.0).abs() < f64::EPSILON && dis_flag == 0 {
        9
    } else if (cur_ctr - 7.0).abs() < f64::EPSILON && dis_flag == 2 {
        7
    } else if (cur_ctr - 7.0).abs() < f64::EPSILON && dis_flag == 1 {
        8
    } else if (cur_ctr - 7.0).abs() < f64::EPSILON && dis_flag == 0 {
        14
    } else if (cur_ctr - 9.0).abs() < f64::EPSILON && dis_flag == 2 {
        17
    } else if (cur_ctr - 9.0).abs() < f64::EPSILON && dis_flag == 1 {
        18
    } else if (cur_ctr - 9.0).abs() < f64::EPSILON && dis_flag == 0 {
        19
    } else {
        0
    }
}
```

- [ ] **Step 2: Extend `score_rbs_prodigal` to include mismatch matches**

Replace the initial `score_rbs_prodigal` from Task 2 with:

```rust
pub fn score_rbs_prodigal(seq: &[u8]) -> usize {
    let start = seq.len();
    let mut best = 0usize;
    for pos in 0..=start.saturating_sub(4) {
        let exact = shine_dalgarno_exact(seq, pos, start);
        let mm = shine_dalgarno_mm(seq, pos, start);
        best = best.max(exact).max(mm);
    }
    best
}
```

- [ ] **Step 3: Add mismatch scanner tests**

```rust
#[test]
fn single_mismatch_6mer_5_10bp_is_bin_19() {
    // AGGAGG with one mismatch in a middle position -> 6-base 1-mm
    let mut seq = vec![b'a'; 21];
    // Place "AGGAGT" (last G -> T) with 6 bp spacer
    let spacer = 6;
    let motif_start = 21 - spacer - 6;
    seq[motif_start..motif_start + 6].copy_from_slice(b"AGGAGT");
    assert_eq!(score_rbs_prodigal(&seq), 19);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- rbs
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rbs_scanner.rs
git commit -m "feat(rbs): add Prodigal mismatch SD scanner"
```

---

## Task 4: Add Prodigal bin → motif/spacer labels

**Files:**
- Modify: `src/rbs_scanner.rs`
- Test: `cargo test --lib -- rbs`

- [ ] **Step 1: Add label tables and formatter**

```rust
const PRODIGAL_SD_STRING: [&str; NUM_RBS_BINS] = [
    "None", "GGA/GAG/AGG", "3Base/5BMM", "4Base/6BMM", "AGxAG", "AGxAG",
    "GGA/GAG/AGG", "GGxGG", "GGxGG", "AGxAG", "AGGAG(G)/GGAGG",
    "AGGA/GGAG/GAGG", "AGGA/GGAG/GAGG", "GGA/GAG/AGG", "GGxGG", "AGGA",
    "GGAG/GAGG", "AGxAGG/AGGxGG", "AGxAGG/AGGxGG", "AGxAGG/AGGxGG",
    "AGGAG/GGAGG", "AGGAG", "AGGAG", "GGAGG", "GGAGG", "AGGAGG",
    "AGGAGG", "AGGAGG",
];

const PRODIGAL_SD_SPACER: [&str; NUM_RBS_BINS] = [
    "None", "3-4bp", "13-15bp", "13-15bp", "11-12bp", "3-4bp",
    "11-12bp", "11-12bp", "3-4bp", "5-10bp", "13-15bp", "3-4bp",
    "11-12bp", "5-10bp", "5-10bp", "5-10bp", "5-10bp", "11-12bp",
    "3-4bp", "5-10bp", "11-12bp", "3-4bp", "5-10bp", "3-4bp",
    "5-10bp", "11-12bp", "3-4bp", "5-10bp",
];

pub fn format_prodigal_rbs_motif(bin: usize) -> Option<String> {
    if bin == 0 || bin >= NUM_RBS_BINS {
        return None;
    }
    Some(format!(
        "{} ({})",
        PRODIGAL_SD_STRING[bin], PRODIGAL_SD_SPACER[bin]
    ))
}

pub fn detect_rbs_motif_prodigal(seq: &[u8]) -> Option<String> {
    format_prodigal_rbs_motif(score_rbs_prodigal(seq))
}
```

- [ ] **Step 2: Add formatter tests**

```rust
#[test]
fn format_prodigal_bin_27() {
    assert_eq!(
        format_prodigal_rbs_motif(27),
        Some("AGGAGG (5-10bp)".to_string())
    );
}

#[test]
fn format_prodigal_bin_0_is_none() {
    assert_eq!(format_prodigal_rbs_motif(0), None);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- rbs
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rbs_scanner.rs
git commit -m "feat(rbs): add Prodigal RBS motif labels"
```

---

## Task 5: Make `get_rbs` reusable from `main.rs`

**Files:**
- Modify: `src/orf.rs:375`
- Modify: `src/rbs_scanner.rs`
- Test: `cargo check`

- [ ] **Step 1: Move `get_rbs` to `src/rbs_scanner.rs` and export it**

In `src/rbs_scanner.rs`:

```rust
/// Extract the 21-nt upstream window of a start codon.
/// `start` is 1-based and points to the first base of the start codon.
/// The returned window is in original (5'→3') orientation.
pub fn get_rbs(dna: &[u8], start: usize) -> Vec<u8> {
    if start >= 21 {
        dna[start - 21..start].to_vec()
    } else {
        let mut pad = vec![b'a'; 21 - start];
        pad.extend_from_slice(&dna[..start]);
        pad
    }
}
```

- [ ] **Step 2: Remove `get_rbs` from `src/orf.rs` and update call sites**

Replace `get_rbs(dna, start, true)` with `crate::rbs_scanner::get_rbs(dna, start)`.

- [ ] **Step 3: Run check**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/orf.rs src/rbs_scanner.rs
git commit -m "refactor(rbs): make get_rbs reusable from main.rs"
```

---

## Task 6: Add `--prodigal-rbs` CLI flag

**Files:**
- Modify: `src/main.rs:95-105`
- Modify: `src/main.rs:529-532`
- Test: `cargo check`

- [ ] **Step 1: Add the flag to `Cli`**

```rust
#[arg(long = "prodigal-rbs", help = "Use Prodigal-style SD scanner with non-SD fallback")]
prodigal_rbs: bool,
```

- [ ] **Step 2: Add mutual exclusivity check**

In the existing `--sd` / `--non-sd` conflict block (around line 529):

```rust
if cli.force_non_sd && cli.force_sd {
    anyhow::bail!("--non-sd and --sd are mutually exclusive");
}
if cli.prodigal_rbs && (cli.force_sd || cli.force_non_sd) {
    anyhow::bail!("--prodigal-rbs is mutually exclusive with --sd and --non-sd");
}
```

- [ ] **Step 3: Thread the flag into `process_genome` calls**

Find the call sites of `process_genome` and add `cli.prodigal_rbs` as the last argument.

- [ ] **Step 4: Update `process_genome` signature**

```rust
fn process_genome(
    genome: &Genome,
    ...existing args...,
    prodigal_rbs: bool,
) -> (String, String, String) {
```

- [ ] **Step 5: Run check**

```bash
cargo check
```

Expected: no errors (the flag is not yet used).

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --prodigal-rbs flag"
```

---

## Task 7: Wire Prodigal RBS + non-SD fallback in `process_genome`

**Files:**
- Modify: `src/main.rs:190-310`
- Modify: `src/nonsd_motif.rs` (add a helper to expose the best motif label)
- Test: `cargo test --lib -- rbs`, `cargo test --test cli_tests`

- [ ] **Step 1: Compute background with the right scanner**

In the sliding-window background loop (around line 222-234):

```rust
let score = if prodigal_rbs {
    crate::rbs_scanner::score_rbs_prodigal(window)
} else {
    crate::rbs_scanner::score_rbs_legacy(window)
};
background_rbs[score] += 1.0;

let rc_score = if prodigal_rbs {
    crate::rbs_scanner::score_rbs_prodigal(rc_window)
} else {
    crate::rbs_scanner::score_rbs_legacy(rc_window)
};
background_rbs[rc_score] += 1.0;
```

- [ ] **Step 2: Add a helper to expose the best non-SD motif label**

In `src/nonsd_motif.rs`, add to `NonSdModel`:

```rust
impl NonSdModel {
    /// Return the best non-SD motif label for an ORF, if one exists.
    pub fn best_motif_label(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> Option<String> {
        let (wseq, start) = upstream_context(dna, rc_dna, orf);
        if start < 18 + MIN_MOTIF_LEN {
            return None;
        }
        let hit = find_best_motif(&self.mot_wt, wseq, start, self.no_mot);
        format_motif_hit(&hit)
    }
}
```

- [ ] **Step 3: Implement the hybrid scoring block**

Replace the existing training/weight_rbs block (lines ~268-308) with:

```rust
if prodigal_rbs {
    // Train non-SD model once for the genome.
    let non_sd_model =
        phanotate_rs::nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);

    // Training distribution from actual ORF upstream windows (Prodigal bins).
    let mut training_rbs = [1.0f64; crate::rbs_scanner::NUM_RBS_BINS];
    for orf in &orfs {
        let upstream = if orf.frame > 0 {
            crate::rbs_scanner::get_rbs(dna, orf.start)
        } else {
            let rbs_start = dna.len().saturating_sub(orf.start + 21);
            let rbs_end = dna.len() - orf.start;
            rc_dna[rbs_start..rbs_end].to_vec()
        };
        training_rbs[crate::rbs_scanner::score_rbs_prodigal(&upstream)] += 1.0;
    }
    let tr_sum: f64 = training_rbs.iter().sum();
    for v in &mut training_rbs {
        *v /= tr_sum;
    }

    for orf in &mut orfs {
        let upstream = if orf.frame > 0 {
            crate::rbs_scanner::get_rbs(dna, orf.start)
        } else {
            let rbs_start = dna.len().saturating_sub(orf.start + 21);
            let rbs_end = dna.len() - orf.start;
            rc_dna[rbs_start..rbs_end].to_vec()
        };
        let bin = crate::rbs_scanner::score_rbs_prodigal(&upstream);
        if bin > 0 {
            orf.rbs_score = bin;
            orf.weight_rbs = training_rbs[bin] / background_rbs[bin];
            orf.motif_score = 1.0;
            orf.rbs_motif = crate::rbs_scanner::format_prodigal_rbs_motif(bin);
        } else {
            orf.rbs_score = 0;
            orf.weight_rbs = 1.0;
            orf.motif_score = non_sd_model.score_orf(orf, dna, rc_dna);
            orf.rbs_motif = non_sd_model
                .best_motif_label(orf, dna, rc_dna)
                .map(|m| format!("nonSD:{m}"));
        }
    }
} else {
    // --- Existing legacy RBS training block ---
    let mut training_rbs = [1.0f64; 28];
    for orf in &orfs {
        training_rbs[orf.rbs_score] += 1.0;
    }
    let tr_sum: f64 = training_rbs.iter().sum();
    for v in &mut training_rbs {
        *v /= tr_sum;
    }
    for orf in &mut orfs {
        orf.weight_rbs = training_rbs[orf.rbs_score] / background_rbs[orf.rbs_score];
    }

    // --- Existing non-SD auto-detect block ---
    let use_non_sd = if force_non_sd {
        true
    } else if force_sd {
        false
    } else {
        !detect_uses_sd(&background_rbs, &training_rbs)
    };

    if use_non_sd {
        for orf in &mut orfs {
            orf.weight_rbs = 1.0;
        }
        let model =
            phanotate_rs::nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);
        for orf in &mut orfs {
            orf.motif_score = model.score_orf(orf, dna, rc_dna);
            let (wseq, start) = phanotate_rs::nonsd_motif::upstream_context(dna, rc_dna, orf);
            let hit = if start >= 18 + phanotate_rs::nonsd_motif::MIN_MOTIF_LEN {
                phanotate_rs::nonsd_motif::find_best_motif(&model.mot_wt, wseq, start, model.no_mot)
            } else {
                phanotate_rs::nonsd_motif::MotifHit::default()
            };
            orf.rbs_motif = phanotate_rs::nonsd_motif::format_motif_hit(&hit);
        }
    }
}
```

- [ ] **Step 4: Run unit tests**

```bash
cargo test --lib -- rbs
```

Expected: all tests pass.

- [ ] **Step 5: Run CLI tests**

```bash
cargo test --test cli_tests -- --skip lambda_ --skip regression_ --skip table4_
```

Expected: existing tests pass (default mode unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/main.rs src/nonsd_motif.rs
git commit -m "feat(rbs): wire Prodigal scanner and non-SD fallback in main.rs"
```

---

## Task 8: Expose `prodigal_rbs` in Python bindings

**Files:**
- Modify: `src/lib_python.rs:260-330`
- Modify: `src/lib_python.rs:370-400`
- Modify: `src/lib_python.rs:495-535`
- Modify: `src/lib_python.rs:760-775`
- Test: `pytest tests/test_python_bindings.py -v`

- [ ] **Step 1: Add `prodigal_rbs` parameter to `phanotate`**

Update the Python signature:

```rust
#[pyfunction]
#[pyo3(signature = (
    input,
    format = "tabular",
    output = None,
    protein = None,
    nucleotide = None,
    table = 11,
    closed_ends = false,
    min_orf_len = 90,
    mask_n = false,
    sd = false,
    non_sd = false,
    prodigal_rbs = false,
))]
fn phanotate(
    ...,
    prodigal_rbs: bool,
) -> PyResult<PyObject> {
    if non_sd && sd {
        return Err(PyValueError::new_err("non_sd and sd are mutually exclusive"));
    }
    if prodigal_rbs && (sd || non_sd) {
        return Err(PyValueError::new_err(
            "prodigal_rbs is mutually exclusive with sd and non_sd",
        ));
    }
    ...
}
```

- [ ] **Step 2: Thread `prodigal_rbs` through `process_single_genome` and scoring**

Update `process_single_genome` signature and the call into `process_genome`.

- [ ] **Step 3: Add `prodigal_rbs` to `find_orfs`**

Update the `#[pyo3(signature = ...)]` and function signature, then pass the flag to the internal ORF-finding call. For `find_orfs`, the function returns ORF records; if `prodigal_rbs` is true, apply the Prodigal + fallback scoring before returning.

- [ ] **Step 4: Add Python tests**

In `tests/test_python_bindings.py`:

```python
def test_phanotate_prodigal_rbs():
    result = phanotate_rs.phanotate(phi_x174_path, prodigal_rbs=True)
    assert result["uses_sd"] is True
    # At least one gene should have an RBS motif string.
    motifs = [g.get("rbs_motif") for g in result["genes"]]
    assert any(m is not None for m in motifs)

def test_phanotate_prodigal_rbs_conflicts_with_non_sd():
    with pytest.raises(ValueError):
        phanotate_rs.phanotate(phi_x174_path, prodigal_rbs=True, non_sd=True)
```

- [ ] **Step 5: Run Python tests**

```bash
maturin develop
pytest tests/test_python_bindings.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lib_python.rs tests/test_python_bindings.py
git commit -m "feat(python): expose prodigal_rbs parameter"
```

---

## Task 9: Add CLI integration tests

**Files:**
- Modify: `tests/cli_tests.rs`
- Test: `cargo test --test cli_tests -- prodigal_rbs`

- [ ] **Step 1: Add test for `--prodigal-rbs` SCO output**

```rust
#[test]
fn prodigal_rbs_produces_sco_with_mixed_motifs() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("input.fa");
    fs::write(&input, include_str!("../phage_reviewed.fasta")).unwrap();

    let output = temp.path().join("out.sco");
    let mut cmd = Command::cargo_bin("phanotate-rs").unwrap();
    cmd.arg("-i").arg(&input)
       .arg("-o").arg(&output)
       .arg("-f").arg("sco")
       .arg("--prodigal-rbs");
    cmd.assert().success();

    let text = fs::read_to_string(&output).unwrap();
    assert!(text.starts_with("# uses_sd: 1"));
    // The motif column (5th tab field after header) should contain at least
    // one SD-looking or nonSD-looking entry.
    assert!(text.contains("AGGAGG") || text.contains("nonSD:"));
}

#[test]
fn prodigal_rbs_conflicts_with_non_sd() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("input.fa");
    fs::write(&input, ">t\nATGCATGCATGCATGCATGCATGC\n").unwrap();

    let output = temp.path().join("out.sco");
    let mut cmd = Command::cargo_bin("phanotate-rs").unwrap();
    cmd.arg("-i").arg(&input)
       .arg("-o").arg(&output)
       .arg("--prodigal-rbs")
       .arg("--non-sd");
    cmd.assert().failure().stderr(predicate::str::contains("mutually exclusive"));
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test --test cli_tests -- prodigal_rbs
```

Expected: new tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/cli_tests.rs
git commit -m "test(cli): add --prodigal-rbs integration tests"
```

---

## Task 10: Quality gates

**Files:** all touched files
- Test: full test suite, clippy, fmt

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Clippy**

```bash
cargo clippy -- -D warnings
```

If there are pre-existing warnings in `src/lib_python.rs` (11 known lints), do not fix them as part of this work. Document them in the commit message if they appear.

- [ ] **Step 3: Rust tests**

```bash
cargo test --lib -- rbs
cargo test --test cli_tests -- --skip lambda_ --skip regression_ --skip table4_
```

- [ ] **Step 4: Python tests**

```bash
maturin develop
pytest tests/test_python_bindings.py -v
```

- [ ] **Step 5: Commit final fixes**

```bash
git add -A
git commit -m "style: format and clippy fixes for prodigal-rbs"
```

---

## Self-Review Checklist

- [ ] Spec coverage: every section of `docs/superpowers/specs/2026-06-24-prodigal-rbs-scanner-design.md` maps to at least one task above.
- [ ] Placeholder scan: no TBD/TODO/fill-in-details remain.
- [ ] Type consistency: `score_rbs_prodigal`, `detect_rbs_motif_prodigal`, `format_prodigal_rbs_motif`, `get_rbs`, `NonSdModel::best_motif_label`, and `prodigal_rbs` parameter names are consistent across tasks.
- [ ] Test coverage: unit tests for scanners, hybrid-mode behavior, CLI integration, and Python bindings are all present.
- [ ] Default behavior unchanged: legacy path remains the default; `--prodigal-rbs` is opt-in.
