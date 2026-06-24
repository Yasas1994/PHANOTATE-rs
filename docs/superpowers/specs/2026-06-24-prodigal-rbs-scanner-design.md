# Prodigal-style RBS scanner with non-SD fallback

## Status

Approved for implementation.

## Goal

Add an optional `--prodigal-rbs` mode to PHANOTATE-rs that replaces the legacy hand-written Shine-Dalgarno pattern table with a direct port of Prodigal’s spacer- and mismatch-aware SD scanner, then falls back to the existing non-SD motif finder for ORFs that have no detectable SD motif.

The default RBS behavior remains unchanged.

## Background

PHANOTATE-rs currently detects SD motifs with a large fixed pattern table (`src/orf.rs` `score_rbs` / `detect_rbs_motif`). The table matches specific strings like `AGGAGG`, `GGAGG`, etc., at fixed upstream ranges and assigns integer scores 0-27.

Prodigal takes a different approach (`sequence.c` `shine_dalgarno_exact` and `shine_dalgarno_mm`):
- It scans a 6-base window upstream of each start codon.
- It scores each position against the `AGGAGG` consensus (A at positions 0/3 = +2, G elsewhere = +3).
- It considers every sub-motif of length 3-6 bp.
- It assigns one of 28 bins based on the motif score **and** the spacer distance to the start codon.
- It also supports single-mismatch 5/6-mers with position-dependent penalties.

This design ports that scanner to Rust and layers the existing non-SD motif finder underneath it.

## Design

### CLI / Python API

- Add `--prodigal-rbs` to `clap::Parser` `Cli` in `src/main.rs`.
- Add `prodigal_rbs: bool = false` to the Python `phanotate()` and `find_orfs()` signatures in `src/lib_python.rs`.
- `--prodigal-rbs` is mutually exclusive with `--sd` and `--non-sd`.
- Default behavior is unchanged (legacy SD matcher).

### New module: `src/rbs_scanner.rs`

Responsibilities:
1. Define the public `RbsMode` enum.
2. Host the legacy matcher (moved from `src/orf.rs`).
3. Implement the Prodigal-style exact and 1-mismatch SD scanners.
4. Provide bin → human-readable motif/spacer labels.

Public API:

```rust
pub const NUM_RBS_BINS: usize = 28;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbsMode {
    Legacy,
    Prodigal,
}

/// Legacy PHANOTATE pattern matcher (default).
pub fn score_rbs_legacy(seq: &[u8]) -> usize;
pub fn detect_rbs_motif_legacy(seq: &[u8]) -> Option<String>;

/// Prodigal-style consensus scanner (opt-in).
pub fn score_rbs_prodigal(seq: &[u8]) -> usize;
pub fn detect_rbs_motif_prodigal(seq: &[u8]) -> Option<String>;

/// Format a Prodigal bin as "MOTIF (SPACER)".
pub fn format_prodigal_rbs_motif(bin: usize) -> Option<String>;
```

Implementation notes:
- The input `seq` is the same 21-nt upstream window used by the current code.
- The function reverses the window once, then scans motif candidates exactly like Prodigal.
- Default RBS weights are identity weights (`rwt[i] = i as f64`) because we are not doing per-genome training in this iteration. Higher bins are therefore preferred.
- Bases other than A/C/G/T are treated as mismatches (match value stays negative).

### Changes to `src/orf.rs`

- Remove the inline `score_rbs` and `detect_rbs_motif` functions.
- Re-export the relevant functions from `src/rbs_scanner.rs` so existing call sites compile.
- Ensure `Orf::score()` continues to multiply `dicodon_score * start_codon_weight * weight_rbs * motif_score`.

### Changes to `src/main.rs`

The flag changes two places in `process_genome`:

1. **Background RBS computation** — the sliding-window background must use the same scanner as the foreground:
   - If `--prodigal-rbs`: use `score_rbs_prodigal(window)` for both forward and reverse windows.
   - Else: keep `score_rbs(window)` (legacy).

2. **ORF scoring** — after ORF enumeration, re-score each ORF:

```rust
if cli.prodigal_rbs {
    // Train non-SD model once for the genome.
    let non_sd_model = crate::nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);

    // Training distribution from the actual ORF upstream windows (Prodigal bins).
    let mut training_rbs = [1.0f64; NUM_RBS_BINS];
    for orf in &orfs {
        let upstream = upstream_window(dna, rc_dna, orf);
        training_rbs[score_rbs_prodigal(&upstream)] += 1.0;
    }
    let tr_sum: f64 = training_rbs.iter().sum();
    for v in &mut training_rbs { *v /= tr_sum; }

    for orf in &mut orfs {
        let upstream = upstream_window(dna, rc_dna, orf);
        let bin = score_rbs_prodigal(&upstream);
        if bin > 0 {
            orf.rbs_score = bin;
            orf.weight_rbs = training_rbs[bin] / background_rbs[bin];
            orf.motif_score = 1.0;
            orf.rbs_motif = format_prodigal_rbs_motif(bin);
        } else {
            orf.rbs_score = 0;
            orf.weight_rbs = 1.0;
            orf.motif_score = non_sd_model.score_orf(orf, dna, rc_dna);
            orf.rbs_motif = non_sd_model.best_motif_label(orf, dna, rc_dna)
                .map(|m| format!("nonSD:{m}"));
        }
    }
} else {
    // existing legacy path
}
```

Notes:
- `upstream_window(dna, rc_dna, orf)` reconstructs the 21-nt window using the same rule as ORF construction (forward uses `dna[orf.start-21..orf.start]`, reverse uses the corresponding slice of `rc_dna`).
- `get_rbs` must be made `pub(crate)` or moved to `src/rbs_scanner.rs` so `process_genome` can reuse it.
- The non-SD fallback label is prefixed with `nonSD:` to distinguish it from SD labels.

### Changes to `src/lib_python.rs`

- Add `prodigal_rbs: bool = false` to `phanotate()` and `find_orfs()`.
- Error if `prodigal_rbs` is combined with `sd=True` or `non_sd=True`.
- Mirror the CLI hybrid logic.

### Output

- SCO/GFF/Gbk already contain an `rbs_motif` column/attribute.
- Prodigal SD motifs are formatted as `AGGAGG (5-10bp)`.
- Non-SD fallback motifs are formatted as `nonSD:AAAAAA` so users can distinguish the source.
- The global `# uses_sd: 1` header is retained because SD scanning is the primary mode.

## Testing

1. **Unit tests in `src/rbs_scanner.rs`**
   - Perfect `AGGAGG` at canonical spacer → bin 27.
   - Perfect `AGGAGG` at 13-15 bp spacer → bin 25.
   - Single-mismatch 6-mer at 5-10 bp spacer → bin 19.
   - No motif → bin 0.
   - Legacy matcher still returns its existing scores for representative windows.

2. **Hybrid-mode unit test**
   - Synthetic genome with one ORF carrying a strong SD and one ORF carrying only a non-SD motif.
   - In `--prodigal-rbs` mode, the first ORF gets a high `weight_rbs` and `motif_score == 1.0`.
   - The second ORF gets `weight_rbs == 1.0` and `motif_score > 1.0`.

3. **CLI integration test**
   - `phanotate-rs -i genome.fasta --prodigal-rbs -f sco` runs without panic.
   - Output contains a `# uses_sd: 1` header and mixed SD/nonSD motif labels.

4. **Regression test**
   - Default run (no `--prodigal-rbs`) produces identical output to the pre-change binary on the test genomes.

## Scope exclusions

- No per-genome SD weight training (Approach B from brainstorming).
- No change to `--sd` or `--non-sd` default paths.
- No dicodon model changes; this feature is orthogonal to the in-progress dicodon work.

## Open questions

None remaining — design approved by user.
