# Dicodon Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--dicodon-filter` mode to PHANOTATE-rs that trains a 6-mer dicodon model from annotated CDS (GenBank input) or heuristic seeds (FASTA input) and drops low-coding-potential ORFs before the shortest-path step.

**Architecture:** Reuse the existing `DicodonModel` in `src/dicodon.rs`. Extend `Genome` to carry annotated CDS coordinates parsed from GenBank. Add a new `DicodonModel::from_annotated_orfs` training path, expose `--dicodon-filter` / `--dicodon-filter-threshold` on the CLI, and mirror the parameters in the Python bindings.

**Tech Stack:** Rust 2021, existing PHANOTATE-rs modules, `cargo test`, `pytest` for Python binding tests.

---

## Task 1: Parse GenBank CDS features and store them in `Genome`

**Files:**
- Modify: `src/genome.rs:5-9`
- Modify: `src/genome.rs:35-39`
- Modify: `src/genome.rs:52-56`
- Modify: `src/genome.rs:77-81`
- Modify: `src/genome.rs:103-107`
- Modify: `src/genome.rs:62-111`
- Test: `cargo test --lib -- genome`

- [ ] **Step 1: Add `cds` field to `Genome`**

```rust
#[derive(Debug, Clone)]
pub struct Genome {
    pub id: String,
    pub seq: Vec<u8>,    // lowercase ASCII nucleotides
    pub rc_seq: Vec<u8>, // reverse complement, pre-computed
    pub cds: Vec<(usize, usize, i8)>, // annotated CDS: (start, end, strand)
}
```

- [ ] **Step 2: Initialize `cds: Vec::new()` in every `Genome` constructor**

Locations:

```rust
// read_fasta_data, first push
genomes.push(Genome {
    id: current_id.clone(),
    seq,
    rc_seq,
    cds: Vec::new(),
});

// read_fasta_data, second push
genomes.push(Genome {
    id: current_id,
    seq,
    rc_seq,
    cds: Vec::new(),
});

// read_genbank, first push
genomes.push(Genome {
    id: current_id.clone(),
    seq,
    rc_seq,
    cds: Vec::new(),
});

// read_genbank, second push
genomes.push(Genome {
    id: current_id,
    seq,
    rc_seq,
    cds: Vec::new(),
});
```

- [ ] **Step 3: Add a CDS location parser**

Insert after `read_genbank`:

```rust
/// Parse a simple GenBank CDS location.
/// Handles `start..end`, `<start..end`, `start..>end`, and
/// `complement(start..end)`. Returns `(start, end, strand)` with strand `1`
/// for forward and `-1` for reverse. Joins and complex locations return None.
fn parse_cds_location(loc: &str) -> Option<(usize, usize, i8)> {
    let loc = loc.trim();
    let (strand, inner) = if loc.starts_with("complement(") {
        let inner = loc.strip_prefix("complement(")?.strip_suffix(")")?;
        (-1i8, inner)
    } else {
        (1i8, loc)
    };
    let (start_str, end_str) = inner.split_once("..")?;
    let start: usize = start_str.trim_start_matches('<').parse().ok()?;
    let end: usize = end_str.trim_end_matches('>').parse().ok()?;
    Some((start, end, strand))
}
```

- [ ] **Step 4: Parse FEATURES in `read_genbank`**

Change the loop body in `read_genbank` to track the FEATURES section and accumulate CDS locations:

```rust
pub fn read_genbank(data: &str) -> anyhow::Result<Vec<Genome>> {
    let mut genomes: Vec<Genome> = Vec::new();
    let mut current_id = String::new();
    let mut current_seq = String::new();
    let mut current_cds: Vec<(usize, usize, i8)> = Vec::new();
    let mut in_origin = false;
    let mut in_features = false;

    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("LOCUS") {
            if !current_id.is_empty() {
                let seq = normalize_seq(&current_seq);
                let rc_seq = rev_comp(&seq);
                genomes.push(Genome {
                    id: current_id.clone(),
                    seq,
                    rc_seq,
                    cds: std::mem::take(&mut current_cds),
                });
            }
            current_id = trimmed.split_whitespace().nth(1).unwrap_or("").to_string();
            current_seq.clear();
            in_origin = false;
            in_features = false;
        } else if trimmed.starts_with("FEATURES") {
            in_features = true;
        } else if trimmed.starts_with("ORIGIN") {
            in_features = false;
            in_origin = true;
        } else if trimmed.starts_with("//") {
            in_origin = false;
        } else if in_origin {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            for part in parts.iter().skip(1) {
                current_seq.push_str(part);
            }
        } else if in_features && trimmed.starts_with("CDS") {
            // Line format: "CDS             1..100"
            // or "CDS             complement(1..100)"
            if let Some(loc) = trimmed.strip_prefix("CDS") {
                if let Some(cds) = parse_cds_location(loc) {
                    current_cds.push(cds);
                }
            }
        }
    }

    if !current_id.is_empty() {
        let seq = normalize_seq(&current_seq);
        let rc_seq = rev_comp(&seq);
        genomes.push(Genome {
            id: current_id,
            seq,
            rc_seq,
            cds: current_cds,
        });
    }

    Ok(genomes)
}
```

- [ ] **Step 5: Add unit tests for the parser**

Append to the `tests` module in `src/genome.rs`:

```rust
#[test]
fn test_parse_cds_location_forward() {
    assert_eq!(parse_cds_location("1..100"), Some((1, 100, 1)));
}

#[test]
fn test_parse_cds_location_complement() {
    assert_eq!(parse_cds_location("complement(1..100)"), Some((1, 100, -1)));
}

#[test]
fn test_parse_cds_location_partial() {
    assert_eq!(parse_cds_location("<1..100"), Some((1, 100, 1)));
    assert_eq!(parse_cds_location("1..>100"), Some((1, 100, 1)));
}

#[test]
fn test_parse_cds_location_join_skipped() {
    assert_eq!(parse_cds_location("join(1..50,60..100)"), None);
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test --lib -- genome
```

Expected: all new and existing `genome` tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/genome.rs
git commit -m "feat(genome): parse GenBank CDS features into Genome.cds"
```

---

## Task 2: Add annotation-driven dicodon training

**Files:**
- Modify: `src/dicodon.rs:37-99`
- Test: `cargo test --lib -- dicodon`

- [ ] **Step 1: Add `from_annotated_orfs`**

Insert inside `impl DicodonModel` after `train`:

```rust
/// Train a dicodon model from a supplied set of trusted ORFs (e.g. annotated
/// CDS). Gene dicodon counts are taken from the ORF sequences in frame;
/// background counts are collected from both genome strands in all frames.
pub fn from_annotated_orfs(orfs: &[&Orf], dna: &[u8], rc_dna: &[u8]) -> Self {
    let mut model = Self {
        scores: [0.0; NUM_DICODONS],
    };

    let mut gene_counts = [0.0f64; NUM_DICODONS];
    let mut bg_counts = [0.0f64; NUM_DICODONS];
    let mut total_gene = 0.0;
    let mut total_bg = 0.0;

    // Background counts on both strands.
    for seq in [dna, rc_dna] {
        for i in 0..seq.len().saturating_sub(5) {
            if let Some(ndx) = kmer_encode(seq, i, 6) {
                bg_counts[ndx] += 1.0;
                total_bg += 1.0;
            }
        }
    }

    // Gene counts from annotated ORFs.
    for orf in orfs {
        let seq = orf.sequence();
        for i in (0..seq.len().saturating_sub(5)).step_by(3) {
            if let Some(ndx) = kmer_encode(seq, i, 6) {
                gene_counts[ndx] += 1.0;
                total_gene += 1.0;
            }
        }
    }

    // Smooth and convert to log-likelihoods.
    for ndx in 0..NUM_DICODONS {
        let g = (gene_counts[ndx] + 1.0) / (total_gene + NUM_DICODONS as f64);
        let b = (bg_counts[ndx] + 1.0) / (total_bg + NUM_DICODONS as f64);
        model.scores[ndx] = (g / b).ln().clamp(-4.0, 4.0);
    }

    model
}
```

- [ ] **Step 2: Add a unit test for annotation-driven training**

Append to the `tests` module in `src/dicodon.rs`:

```rust
#[test]
fn from_annotated_orfs_produces_scores() {
    let unit = b"atgaaaaaaaatgaaaaaaatgaaaaaaa";
    let seq = unit
        .iter()
        .cycle()
        .take(unit.len() * 20)
        .copied()
        .collect::<Vec<u8>>();
    let rc = crate::genome::rev_comp(&seq);
    let orf = Orf {
        start: 1,
        stop: seq.len() - 2,
        frame: 1,
        seq: seq.clone(),
        rbs_score: 0,
        rbs_motif: None,
        pstop: 0.01,
        weight_rbs: 2.0,
        hold: 100.0,
        motif_score: 1.0,
        dicodon_score: 1.0,
        start_score: 1.0,
        weight: 1.0,
    };
    let model = DicodonModel::from_annotated_orfs(&[&orf], &seq, &rc);
    assert!(model.scores.iter().all(|&s| s.is_finite()));
    let s = model.score_orf(&orf);
    assert!(s > 0.0);
    assert!(s <= 10.0);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib -- dicodon
```

Expected: all dicodon tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/dicodon.rs
git commit -m "feat(dicodon): add from_annotated_orfs training path"
```

---

## Task 3: Add CLI flags

**Files:**
- Modify: `src/main.rs:110-120`

- [ ] **Step 1: Add `--dicodon-filter` and `--dicodon-filter-threshold` to `Cli`**

Insert after the existing `--dicodon` flag:

```rust
/// Use a Prodigal-style 6-mer dicodon coding-potential model.
#[arg(long = "dicodon")]
dicodon: bool,

/// Train a dicodon model from annotated CDS and drop low-scoring ORFs.
#[arg(long = "dicodon-filter")]
dicodon_filter: bool,

/// Minimum dicodon score to keep an ORF when --dicodon-filter is enabled.
#[arg(long = "dicodon-filter-threshold", value_name = "FLOAT", default_value_t = 0.5)]
dicodon_filter_threshold: f64,

/// Path to a learned start-site scoring model (JSON).
#[arg(long = "start-model", value_name = "FILE")]
start_model: Option<PathBuf>,
```

- [ ] **Step 2: Validate mutual exclusivity with `--dicodon`**

Insert in `main()` after the existing mutual-exclusivity checks (around line 634):

```rust
if cli.dicodon && cli.dicodon_filter {
    anyhow::bail!("--dicodon and --dicodon-filter are mutually exclusive");
}
```

- [ ] **Step 3: Build to verify flag parsing**

```bash
cargo build
```

Expected: builds without errors.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --dicodon-filter and --dicodon-filter-threshold flags"
```

---

## Task 4: Wire the filter into the annotation pipeline

**Files:**
- Modify: `src/main.rs:203-219` (`process_genome` signature)
- Modify: `src/main.rs:530-548` (dicodon/start-model scoring block)
- Modify: `src/main.rs:842-858` and `src/main.rs:864-879` (call sites)

- [ ] **Step 1: Extend `process_genome` signature**

```rust
#[allow(clippy::too_many_arguments)]
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
    dicodon_filter: bool,
    dicodon_filter_threshold: f64,
    start_model: Option<&phanotate_rs::start_refiner::StartModel>,
) -> Result<(String, String, String)> {
```

- [ ] **Step 2: Add the filter block after start-model scoring**

Replace the existing block at lines 536-548:

```rust
    if dicodon {
        let model = phanotate_rs::dicodon::DicodonModel::train(&orfs, dna, rc_dna);
        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
    }

    if let Some(model) = start_model {
        for orf in &mut orfs {
            let features = phanotate_rs::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }
```

with:

```rust
    if dicodon {
        let model = phanotate_rs::dicodon::DicodonModel::train(&orfs, dna, rc_dna);
        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
    }

    if let Some(model) = start_model {
        for orf in &mut orfs {
            let features = phanotate_rs::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }

    if dicodon_filter {
        let annotated: Vec<&Orf> = if genome.cds.is_empty() {
            Vec::new()
        } else {
            orfs.iter()
                .filter(|o| {
                    genome.cds.iter().any(|(s, e, strand)| {
                        if *strand > 0 {
                            // Forward: ORF start == CDS start, ORF stop == CDS end.
                            o.frame > 0 && o.start == *s && o.stop == *e
                        } else {
                            // Reverse: ORF start == CDS end, ORF stop == CDS start.
                            o.frame < 0 && o.start == *e && o.stop == *s
                        }
                    })
                })
                .collect()
        };

        let model = if annotated.is_empty() {
            eprintln!(
                "Warning: --dicodon-filter enabled for '{}' but no annotated CDS found; \
                 falling back to heuristic seed training.",
                genome.id
            );
            phanotate_rs::dicodon::DicodonModel::train(&orfs, dna, rc_dna)
        } else {
            phanotate_rs::dicodon::DicodonModel::from_annotated_orfs(&annotated, dna, rc_dna)
        };

        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
        orfs.retain(|o| o.dicodon_score >= dicodon_filter_threshold);
    }
```

- [ ] **Step 3: Update the two `process_genome` call sites**

Add the two new arguments after `cli.dicodon`:

```rust
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
    cli.dicodon_filter,
    cli.dicodon_filter_threshold,
    start_model.as_ref(),
)
```

Update both the `cli.progress` branch and the `else` branch.

- [ ] **Step 4: Build**

```bash
cargo build
```

Expected: builds without errors.

- [ ] **Step 5: Smoke-test on NC_001365.gb**

```bash
./target/debug/phanotate-rs -i tests/golden/NC_001365.gb -g 4 -f sco --start-model /tmp/start_model_cv1000.json --dicodon-filter --dicodon-filter-threshold 0.5 | grep -v '^#' | wc -l
```

Expected: runs and prints a gene count (value will be verified later).

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): apply annotation-driven dicodon filter before graph build"
```

---

## Task 5: Expose the filter in Python bindings

**Files:**
- Modify: `src/lib_python.rs:311-317` (docstring)
- Modify: `src/lib_python.rs:348-353` (pyfunction defaults)
- Modify: `src/lib_python.rs:355-369` (phanotate signature)
- Modify: `src/lib_python.rs:435-440` (process_single_genome call)
- Modify: `src/lib_python.rs:465-476` (process_single_genome signature)
- Modify: `src/lib_python.rs:784-789` (dicodon scoring block)
- Modify: `src/lib_python.rs:936-941` (find_orfs docstring)
- Modify: `src/lib_python.rs:960-964` (find_orfs defaults)
- Modify: `src/lib_python.rs:965-973` (find_orfs signature)
- Modify: `src/lib_python.rs:1058-1063` (find_orfs start-model block)
- Test: `cargo check --features python`

- [ ] **Step 1: Update `phanotate` defaults and signature**

Change defaults block:

```rust
    dicodon = false,
    dicodon_filter = false,
    dicodon_filter_threshold = 0.5,
    start_model = None,
    min_orf_len = 90,
))]
fn phanotate(
    sequence: &str,
    seq_id: Option<&str>,
    format: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    non_sd: bool,
    sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    dicodon_filter: bool,
    dicodon_filter_threshold: f64,
    start_model: Option<&str>,
    min_orf_len: usize,
) -> PyResult<PyObject> {
```

- [ ] **Step 2: Pass new parameters to `process_single_genome`**

```rust
    let (primary, protein, nucleotide, genes, _use_non_sd) = process_single_genome(
        &dna,
        &rc_dna,
        seq_id.unwrap_or("unknown"),
        format,
        table,
        closed_ends,
        mask_n,
        non_sd,
        sd,
        prodigal_rbs,
        dicodon,
        dicodon_filter,
        dicodon_filter_threshold,
        start_model,
        min_orf_len,
    )?;
```

- [ ] **Step 3: Update `process_single_genome` signature**

```rust
fn process_single_genome(
    dna: &[u8],
    rc_dna: &[u8],
    seq_id: &str,
    format: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    force_non_sd: bool,
    force_sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    dicodon_filter: bool,
    dicodon_filter_threshold: f64,
    start_model: Option<&str>,
    min_orf_len: usize,
) -> PyResult<(String, String, String, Vec<PyGene>, bool)> {
```

- [ ] **Step 4: Add the filter block in `process_single_genome`**

After the existing dicodon/start-model block:

```rust
    if let Some(path) = start_model {
        let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
        for orf in &mut orfs {
            let features = crate::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }

    if dicodon_filter {
        // Python binding has no annotated CDS; always use heuristic seeds.
        let model = crate::dicodon::DicodonModel::train(&orfs, dna, rc_dna);
        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
        orfs.retain(|o| o.dicodon_score >= dicodon_filter_threshold);
    }
```

- [ ] **Step 5: Update `find_orfs` defaults and signature**

Defaults block:

```rust
    prodigal_rbs = false,
    dicodon_filter = false,
    dicodon_filter_threshold = 0.5,
    start_model = None,
))]
fn find_orfs(
    sequence: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    min_orf_len: usize,
    prodigal_rbs: bool,
    dicodon_filter: bool,
    dicodon_filter_threshold: f64,
    start_model: Option<&str>,
) -> PyResult<Vec<PyOrf>> {
```

- [ ] **Step 6: Add the filter block in `find_orfs`**

After the existing start-model block:

```rust
    if let Some(path) = start_model {
        let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
        for orf in &mut orfs {
            let features = crate::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }

    if dicodon_filter {
        let model = crate::dicodon::DicodonModel::train(&orfs, dna, rc_dna);
        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
        orfs.retain(|o| o.dicodon_score >= dicodon_filter_threshold);
    }
```

- [ ] **Step 7: Update docstrings**

Add to `phanotate` docstring after the `dicodon` paragraph:

```text
dicodon_filter : bool, optional
    If True, train a dicodon model from the ORF set and drop ORFs whose
    dicodon score is below `dicodon_filter_threshold`. Default is False.
dicodon_filter_threshold : float, optional
    Minimum dicodon score kept by `dicodon_filter`. Default is 0.5.
```

Add a similar paragraph to `find_orfs` docstring.

- [ ] **Step 8: Check Python build**

```bash
PYO3_PYTHON=$PWD/.venv/bin/python cargo check --features python
```

Expected: compiles without new errors (pre-existing `lib_python.rs` warnings are acceptable).

- [ ] **Step 9: Commit**

```bash
git add src/lib_python.rs
git commit -m "feat(python): expose dicodon_filter parameters in bindings"
```

---

## Task 6: Add integration tests

**Files:**
- Modify: `tests/cli_tests.rs`
- Test: `cargo test --test cli_tests -- dicodon_filter`

- [ ] **Step 1: Add a smoke test for `--dicodon-filter`**

Append to `tests/cli_tests.rs`:

```rust
#[test]
fn dicodon_filter_runs_on_genbank() {
    let out = run_phanotate(&[
        "-i",
        "tests/golden/NC_001365.gb",
        "-g",
        "4",
        "--dicodon-filter",
        "--dicodon-filter-threshold",
        "0.5",
        "-f",
        "sco",
    ]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap();
    let cols: Vec<_> = data_line.split('\t').collect();
    assert_eq!(cols.len(), 5);
}
```

- [ ] **Step 2: Add a regression test that filter changes output**

```rust
#[test]
fn dicodon_filter_changes_output() {
    let out_default = run_phanotate(&[
        "-i",
        "tests/data/small.fasta",
        "-f",
        "sco",
    ]);
    let out_filter = run_phanotate(&[
        "-i",
        "tests/data/small.fasta",
        "--dicodon-filter",
        "-f",
        "sco",
    ]);
    assert!(out_default.status.success());
    assert!(out_filter.status.success());
    assert_ne!(
        String::from_utf8(out_default.stdout).unwrap(),
        String::from_utf8(out_filter.stdout).unwrap()
    );
}
```

- [ ] **Step 3: Run the new tests**

```bash
cargo test --test cli_tests -- dicodon_filter
```

Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cli_tests.rs
git commit -m "test(cli): add --dicodon-filter smoke and effect tests"
```

---

## Task 7: Quality gates

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Clippy default**

```bash
cargo clippy -- -D warnings
```

Expected: zero warnings.

- [ ] **Step 3: Clippy Python**

```bash
PYO3_PYTHON=$PWD/.venv/bin/python cargo clippy --features python -- -D warnings
```

Expected: only pre-existing `lib_python.rs` warnings.

- [ ] **Step 4: Unit tests**

```bash
cargo test --lib -- --skip debug_
```

Expected: all unit tests pass.

- [ ] **Step 5: Integration tests**

```bash
cargo test --test cli_tests
```

Expected: all integration tests pass.

- [ ] **Step 6: Release build**

```bash
cargo build --release
```

- [ ] **Step 7: Manual smoke on NC_001365.gb**

```bash
./target/release/phanotate-rs -i tests/golden/NC_001365.gb -g 4 -f sco --start-model /tmp/start_model_cv1000.json --dicodon-filter --dicodon-filter-threshold 0.5 > /tmp/nc001365_filter.sco
```

Then compare against the start-model-only run and verify precision improves.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "style: final formatting for dicodon filter feature"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|---|---|
| Parse GenBank CDS into `Genome.cds` | Task 1 |
| `DicodonModel::from_annotated_orfs` | Task 2 |
| `--dicodon-filter` CLI flag | Task 3 |
| `--dicodon-filter-threshold` CLI flag | Task 3 |
| Filter logic in `process_genome` | Task 4 |
| Python binding parameters | Task 5 |
| CLI integration tests | Task 6 |
| Quality gates | Task 7 |

## Placeholder Scan

No TBD/TODO placeholders. Every step contains exact file paths, code, commands, and expected results.

## Type Consistency Notes

- `Genome.cds` is `Vec<(usize, usize, i8)>` everywhere.
- `DicodonModel::from_annotated_orfs` takes `&[&Orf]` and returns `Self`.
- `process_genome` and `process_single_genome` receive `dicodon_filter: bool` and `dicodon_filter_threshold: f64`.
- The filter is applied after start-model scoring and before graph construction in both CLI and Python paths.
