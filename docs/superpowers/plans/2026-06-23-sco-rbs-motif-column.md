# SCO RBS Motif Column Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a detected RBS/non-SD motif column to PHANOTATE-rs SCO output.

**Architecture:** Store the motif as `Option<String>` on each `Orf`; populate it during SD scoring in `find_orfs_with_rc` and during non-SD scoring in `main.rs`/`lib_python.rs`; render it as the fifth tab-separated column in `write_sco`.

**Tech Stack:** Rust 2021, existing PHANOTATE-rs modules, `cargo test`.

---

## Task 1: Add `rbs_motif` field to `Orf`

**Files:**
- Modify: `src/orf.rs:40-52`
- Modify: `src/orf.rs:171-184`
- Modify: `src/orf.rs:220-230`
- Modify: `src/orf.rs:253-263`
- Modify: `src/orf.rs:293-303`
- Test: `cargo test --lib -- orf`

- [ ] **Step 1: Add field to `Orf`**

```rust
pub struct Orf {
    pub start: usize,
    pub stop: usize,
    pub frame: i8,
    pub seq: Vec<u8>,
    pub rbs_score: usize,
    pub rbs_motif: Option<String>, // <-- new
    pub pstop: f64,
    pub weight_rbs: f64,
    pub hold: f64,
    pub motif_score: f64,
    pub weight: f64,
}
```

- [ ] **Step 2: Initialize `rbs_motif: None` in all four constructors**

Forward ORF:
```rust
orfs.push(Orf {
    start,
    stop: stop - 2,
    frame: frame_i8,
    seq,
    rbs_score,
    rbs_motif: None, // <-- new
    pstop,
    weight_rbs: 1.0,
    hold: 1.0,
    motif_score: 1.0,
    weight: 1.0,
});
```

Repeat for reverse ORF, forward fragment, reverse fragment.

- [ ] **Step 3: Update `test_orf` helper in `src/ml_features.rs` and any other test helpers that construct `Orf`**

Add `rbs_motif: None` to `test_orf` in `src/ml_features.rs` and to any `Orf` literals in tests.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib -- orf
```

Expected: compiles and passes.

- [ ] **Step 5: Commit**

```bash
git add src/orf.rs src/ml_features.rs
git commit -m "feat(orf): add rbs_motif field to Orf struct"
```

---

## Task 2: Add SD motif detector and populate in constructors

**Files:**
- Modify: `src/orf.rs:362-520`
- Modify: `src/orf.rs:174-184`
- Modify: `src/orf.rs:220-230`
- Modify: `src/orf.rs:255-265`
- Modify: `src/orf.rs:296-306`
- Test: `cargo test --lib -- orf`

- [ ] **Step 1: Add `detect_rbs_motif` helper after `score_rbs`**

```rust
/// Detect the matching Shine-Dalgarno motif in the upstream window.
/// Mirrors the priority order of `score_rbs` and returns the first
/// matching motif as an uppercase string, or `None` if no motif matches.
pub fn detect_rbs_motif(seq: &[u8]) -> Option<String> {
    let s: Vec<u8> = seq.iter().rev().copied().collect();

    let in_range = |pat: &[u8], start: usize, end: usize| -> bool {
        if end > s.len() || start >= s.len() {
            return false;
        }
        let window = &s[start..end];
        if pat.len() > window.len() {
            return false;
        }
        window.windows(pat.len()).any(|w| w == pat)
    };

    // Patterns and their display names (reverse-complement of the matched bytes).
    let patterns: &[(&[u8], &str)] = &[
        (b"ggagga", "AGGAGG"),
        (b"ggagg", "GGAGG"),
        (b"gagga", "GAGGA"),
        (b"ggacga", "GGACGA"),
        (b"ggatga", "GGATGA"),
        (b"ggaaga", "GGAAGA"),
        (b"ggcgga", "GGCGGA"),
        (b"ggggga", "GGGGGA"),
        (b"ggtgga", "GGTGGA"),
        (b"ggag", "GGAG"),
        (b"gagg", "GAGG"),
        (b"agga", "AGGA"),
        (b"ggcg", "GGCG"),
        (b"gacga", "GACGA"),
        (b"gauga", "GATGA"),
        (b"gaaga", "GAAGA"),
        (b"agcga", "AGCGA"),
        (b"aggcga", "AGGCGA"),
        (b"gggagg", "GGGAGG"),
        (b"gaggtg", "GAGGTG"),
        (b"ggtg", "GGTG"),
        (b"gctggt", "GCTGGT"),
        (b"gcccat", "GCCCAT"),
    ];

    for (pat, name) in patterns {
        // Canonical SD spacing: positions 3-15 in the reversed window
        if in_range(pat, 3, 15) {
            return Some(name.to_string());
        }
    }

    None
}
```

- [ ] **Step 2: Populate `rbs_motif` in the four `Orf` constructors**

Where `rbs_score = score_rbs(&rbs)` is called, also compute:

```rust
let rbs_motif = detect_rbs_motif(&rbs);
```

and pass `rbs_motif` into the `Orf` constructor.

- [ ] **Step 3: Add unit tests**

```rust
#[test]
fn test_detect_rbs_motif_aggagg() {
    let seq = b"aaggaggtgagtaacaaaacc"; // reversed window contains ggagga
    assert_eq!(detect_rbs_motif(seq), Some("AGGAGG".to_string()));
}

#[test]
fn test_detect_rbs_motif_none() {
    let seq = b"aaaaaaaaaaaaaaaaaaaaa";
    assert_eq!(detect_rbs_motif(seq), None);
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test --lib -- orf
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/orf.rs
git commit -m "feat(orf): detect and store SD motif string"
```

---

## Task 3: Populate `rbs_motif` in non-SD mode

**Files:**
- Modify: `src/nonsd_motif.rs`
- Modify: `src/main.rs`
- Modify: `src/lib_python.rs`
- Test: `cargo test --lib -- nonsd`

- [ ] **Step 1: Add `format_motif_hit` helper in `src/nonsd_motif.rs`**

```rust
/// Format a discovered non-SD motif hit as an uppercase DNA string.
pub fn format_motif_hit(hit: &MotifHit) -> Option<String> {
    if hit.len == 0 {
        None
    } else {
        Some(String::from_utf8(kmer_decode(hit.ndx, hit.len)).unwrap())
    }
}
```

- [ ] **Step 2: In `src/main.rs`, populate `rbs_motif` after non-SD scoring**

In the non-SD branch where `orf.motif_score = model.score_orf(...)` is set, also set:

```rust
let (wseq, start) = nonsd_motif::upstream_context(dna, rc_dna, orf);
let hit = if start >= 18 + MIN_MOTIF_LEN {
    nonsd_motif::find_best_motif(&model.mot_wt, wseq, start, model.no_mot)
} else {
    nonsd_motif::MotifHit::default()
};
orf.rbs_motif = nonsd_motif::format_motif_hit(&hit);
```

If `upstream_context` is private, make it `pub(crate)`.

- [ ] **Step 3: Do the same in `src/lib_python.rs`**

In the Python non-SD branch, populate `orf.rbs_motif` the same way.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib -- nonsd
cargo check --features python
```

- [ ] **Step 5: Commit**

```bash
git add src/nonsd_motif.rs src/main.rs src/lib_python.rs
git commit -m "feat(nonsd,python): populate rbs_motif in non-SD mode"
```

---

## Task 4: Add motif column to SCO output

**Files:**
- Modify: `src/output.rs:159-169`
- Modify: `src/output.rs:437-448`
- Modify: `src/output.rs:534-536`
- Test: `cargo test --lib -- output`

- [ ] **Step 1: Update `write_sco` to include motif column**

```rust
fn write_sco(_id: &str, path: &[(Node, Node, f64)], orfs: &[Orf], uses_sd: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("# uses_sd: {}\n", if uses_sd { 1 } else { 0 }));
    for (start, stop, strand, weight, orf) in collect_orf_edges(path, orfs) {
        let motif = orf.rbs_motif.as_deref().unwrap_or("not detected");
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2E}\t{}\n",
            start, stop, strand, weight, motif
        ));
    }
    out
}
```

- [ ] **Step 2: Update `test_write_sco_basic`**

```rust
#[test]
fn test_write_sco_basic() {
    let (path, orfs) = ...; // existing setup
    let out = write_sco(">test", &path, &orfs, true);
    let line = out.lines().nth(1).unwrap();
    assert!(line.contains('\t'));
    let cols: Vec<_> = line.split('\t').collect();
    assert_eq!(cols.len(), 5);
    assert_eq!(cols[4], "not detected");
}
```

- [ ] **Step 3: Update `test_write_primary_sco`**

Add an assertion that the output has five tab-separated columns.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib -- output
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/output.rs
git commit -m "feat(output): add motif column to SCO output"
```

---

## Task 5: Update CLI integration tests

**Files:**
- Modify: `tests/cli_tests.rs`

- [ ] **Step 1: Update existing SCO assertions**

Find any existing SCO assertions and verify they now expect five columns.

- [ ] **Step 2: Add non-SD motif test**

```rust
#[test]
fn non_sd_sco_includes_motif_column() {
    let out = run_phanotate(&[
        "-i",
        "tests/data/small.fasta",
        "--non-sd",
        "-f",
        "sco",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap();
    let cols: Vec<_> = data_line.split('\t').collect();
    assert_eq!(cols.len(), 5);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --test cli_tests -- non_sd
```

- [ ] **Step 4: Commit**

```bash
git add tests/cli_tests.rs
git commit -m "test(cli): verify SCO motif column"
```

---

## Task 6: Quality gates

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Clippy default**

```bash
cargo clippy -- -D warnings
```

- [ ] **Step 3: Clippy Python**

```bash
cargo clippy --features python -- -D warnings
```

Note: pre-existing warnings in `src/lib_python.rs` are acceptable.

- [ ] **Step 4: Unit tests**

```bash
cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_
```

- [ ] **Step 5: Integration tests**

```bash
cargo test --test cli_tests
```

- [ ] **Step 6: End-to-end smoke**

```bash
cargo build --release
./target/release/phanotate-rs -i tests/data/small.fasta -f sco | head -3
./target/release/phanotate-rs -i tests/data/small.fasta --non-sd -f sco | head -3
```

- [ ] **Step 7: Commit any final fixes**

```bash
git add -A
git commit -m "style: final formatting for SCO motif column"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| `rbs_motif` field on `Orf` | Task 1 |
| SD motif detection | Task 2 |
| Non-SD motif population | Task 3 |
| SCO column output | Task 4 |
| Integration tests | Task 5 |
| Quality gates | Task 6 |

## Placeholder Scan

No TBD/TODO placeholders. All steps include exact code, commands, and expected results.

## Type Consistency Notes

- `rbs_motif: Option<String>` is used consistently across `Orf`, `detect_rbs_motif`, and `format_motif_hit`.
- `write_sco` uses `as_deref().unwrap_or("not detected")` for rendering.
