//! Detect the most likely NCBI translation table from a DNA sequence.
//!
//! Uses two heuristics:
//! 1. Mean ORF length *ratio* relative to table-11 baseline — tables with
//!    fewer stop codons structurally produce longer ORFs on any genome, so we
//!    normalise by the table-11 baseline to remove that bias.
//! 2. Reassigned-codon signal — codons that are stops in table 11 but sense
//!    codons in an alternative table.  Their frequency inside candidate-table
//!    ORFs (compared to sibling codons or background) tells us whether the
//!    alternative code is actually in use.

use crate::codon_table::{stop_codons, table_name};

/// Translation tables tested by --detect-table.
/// Only tables with published evidence of use in phage genomes are included.
///
/// Excluded:
///   Table 6  (Ciliate nuclear; TAA/TAG → Gln) — no phage genome has been
///            reported to use this code in any large-scale survey
///            (INPHARED 20 000+ genomes; UHGV hundreds of thousands of vOTUs).
///            Available as a manual override via -g 6.
///
/// Included:
///   Table  1 — Standard; baseline for eukaryotic viruses.
///   Table  4 — TGA→Trp; all Mycoplasma/Spiroplasma phages (e.g. SpV4 NC_003438).
///   Table 11 — Default for all bacteriophages.
///   Table 15 — TAG→Gln; experimentally confirmed in crAss-like gut phages
///              (Peters et al., Nat Commun 2022).
///   Table 25 — TGA→Gly; Gracilibacteria/SR1 phages in oral and gut microbiomes
///              (Liu et al., mBio 2023; doi:10.5281/zenodo.8422333).
pub const CANDIDATE_TABLES: &[u8] = &[1, 4, 11, 15, 25];

/// Minimum sequence length for detection to be reliable.
const MIN_SEQ_LEN: usize = 300;

// ---------------------------------------------------------------------------
// 1. Mean ORF length
// ---------------------------------------------------------------------------

/// Enumerate all open regions ≥ `min_len` nt on both strands using the
/// stop-codon set for `table`, then return the mean length in nucleotides.
pub fn mean_orf_length(seq: &[u8], table: u8, min_len: usize) -> f64 {
    let stops = stop_codons(table);
    let rc = rev_comp(seq);

    let mut total_len: usize = 0;
    let mut count: usize = 0;

    for frame in 0..3 {
        let mut region_start = frame;
        let mut pos = frame;
        while pos + 3 <= seq.len() {
            let codon = &seq[pos..pos + 3];
            if is_stop(codon, stops) {
                let len = pos.saturating_sub(region_start);
                if len >= min_len {
                    total_len += len;
                    count += 1;
                }
                region_start = pos + 3;
            }
            pos += 3;
        }
        let len = seq.len().saturating_sub(region_start);
        if len >= min_len {
            total_len += len;
            count += 1;
        }
    }

    for frame in 0..3 {
        let mut region_start = frame;
        let mut pos = frame;
        while pos + 3 <= rc.len() {
            let codon = &rc[pos..pos + 3];
            if is_stop(codon, stops) {
                let len = pos.saturating_sub(region_start);
                if len >= min_len {
                    total_len += len;
                    count += 1;
                }
                region_start = pos + 3;
            }
            pos += 3;
        }
        let len = rc.len().saturating_sub(region_start);
        if len >= min_len {
            total_len += len;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_len as f64 / count as f64
    }
}

/// Maximum ORF length on both strands.
fn max_orf_length(seq: &[u8], table: u8, min_len: usize) -> usize {
    let stops = stop_codons(table);
    let rc = rev_comp(seq);
    let mut max_len: usize = 0;

    for strand in [seq, &rc] {
        for frame in 0..3 {
            let mut region_start = frame;
            let mut pos = frame;
            while pos + 3 <= strand.len() {
                let codon = &strand[pos..pos + 3];
                if is_stop(codon, stops) {
                    let len = pos.saturating_sub(region_start);
                    if len >= min_len {
                        max_len = max_len.max(len);
                    }
                    region_start = pos + 3;
                }
                pos += 3;
            }
            let len = strand.len().saturating_sub(region_start);
            if len >= min_len {
                max_len = max_len.max(len);
            }
        }
    }
    max_len
}

fn is_stop(codon: &[u8], stops: &[&[u8]]) -> bool {
    stops.iter().any(|&s| s == codon)
}

fn rev_comp(seq: &[u8]) -> Vec<u8> {
    let mut rc = Vec::with_capacity(seq.len());
    for &b in seq.iter().rev() {
        rc.push(match b {
            b'a' => b't',
            b't' => b'a',
            b'c' => b'g',
            b'g' => b'c',
            _ => b'n',
        });
    }
    rc
}

// ---------------------------------------------------------------------------
// 2. Reassigned-codon detection
// ---------------------------------------------------------------------------

/// Codons that are stop codons under table 11 but sense codons under
/// the given alternative table.  These are the signal codons for detection.
pub fn reassigned_codons(table: u8) -> &'static [&'static [u8]] {
    match table {
        1 | 11 => &[],          // baseline — nothing reassigned
        4 => &[b"tga"],         // TGA → Trp
        6 => &[b"taa", b"tag"], // TAA → Gln, TAG → Gln
        15 => &[b"tag"],        // TAG → Gln
        25 => &[b"tga"],        // TGA → Gly
        _ => &[],
    }
}

/// Result of checking one reassigned codon.
#[derive(Clone)]
pub struct ReassignmentResult {
    pub codon: [u8; 3],
    pub background_freq: f64,
    pub orf_freq: f64,
    pub ratio: f64,
}

/// Collect the longest ORFs under `table` (forward strand only) and
/// count target codons inside them.
///
/// Returns (total_codons_scanned, counts_per_target).
fn count_in_candidate_orfs(
    seq: &[u8],
    table: u8,
    targets: &[[u8; 3]],
    top_k: usize,
) -> (usize, Vec<usize>) {
    let stops = stop_codons(table);
    let mut regions: Vec<(usize, usize)> = Vec::new();

    // Forward strand only
    for frame in 0..3 {
        let mut region_start = frame;
        let mut pos = frame;
        while pos + 3 <= seq.len() {
            let codon = &seq[pos..pos + 3];
            if is_stop(codon, stops) {
                let len = pos.saturating_sub(region_start);
                if len >= 90 {
                    regions.push((region_start, pos));
                }
                region_start = pos + 3;
            }
            pos += 3;
        }
        let len = seq.len().saturating_sub(region_start);
        if len >= 90 {
            regions.push((region_start, seq.len()));
        }
    }

    regions.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then_with(|| a.0.cmp(&b.0)));

    let mut counts = vec![0usize; targets.len()];
    let mut total = 0usize;
    for &(start, end) in regions.iter().take(top_k) {
        let mut pos = start;
        while pos + 6 <= end {
            let codon = [seq[pos], seq[pos + 1], seq[pos + 2]];
            for (i, target) in targets.iter().enumerate() {
                if codon == *target {
                    counts[i] += 1;
                }
            }
            total += 1;
            pos += 3;
        }
    }
    (total, counts)
}

/// Compute background frequency of each target codon (frame 0 only).
fn background_freqs(seq: &[u8], targets: &[[u8; 3]]) -> (Vec<usize>, usize) {
    let mut counts = vec![0usize; targets.len()];
    let mut total = 0usize;
    for pos in (0..seq.len().saturating_sub(2)).step_by(3) {
        let codon = [seq[pos], seq[pos + 1], seq[pos + 2]];
        for (i, target) in targets.iter().enumerate() {
            if codon == *target {
                counts[i] += 1;
            }
        }
        total += 1;
    }
    (counts, total)
}

/// Scan a sliding window across the sequence and score each window.
///
/// For mosaic genomes (tables 15, 6), only a segment may use the alternative
/// code.  We scan windows and return the mean of the top-3 window scores.
/// This is more robust than taking the single best window (which can be a
/// false positive due to random variation).
///
/// Each window must have at least `min_codons` interior codons in its top
/// ORFs for the score to be trusted; windows with too few codons are
/// down-weighted.
fn sliding_window_signal(
    seq: &[u8],
    table: u8,
    targets: &[[u8; 3]],
    window_size: usize,
    step: usize,
) -> (f64, Vec<ReassignmentResult>) {
    let stops = stop_codons(table);
    const MIN_CODONS_PER_WINDOW: usize = 300;

    // Pre-compute background counts for the whole sequence
    let (bg_counts, bg_total) = background_freqs(seq, targets);
    let _bg_freqs: Vec<f64> = bg_counts
        .iter()
        .map(|&c| {
            if bg_total > 0 {
                c as f64 / bg_total as f64
            } else {
                0.0
            }
        })
        .collect();

    let mut window_scores: Vec<(f64, Vec<ReassignmentResult>)> = Vec::new();

    let mut window_start = 0usize;
    while window_start + window_size <= seq.len() {
        let window_end = window_start + window_size;

        // Find candidate-table ORFs inside this window (forward strand only)
        let mut regions: Vec<(usize, usize)> = Vec::new();
        for frame in 0..3 {
            let mut region_start = window_start + frame;
            let mut pos = region_start;
            while pos + 3 <= window_end {
                let codon = &seq[pos..pos + 3];
                if is_stop(codon, stops) {
                    let len = pos.saturating_sub(region_start);
                    if len >= 90 {
                        regions.push((region_start, pos));
                    }
                    region_start = pos + 3;
                }
                pos += 3;
            }
            let len = window_end.saturating_sub(region_start);
            if len >= 90 {
                regions.push((region_start, window_end));
            }
        }

        regions.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then_with(|| a.0.cmp(&b.0)));

        // Count targets in top-10 ORFs of this window
        let mut orf_counts = vec![0usize; targets.len()];
        let mut orf_total = 0usize;
        for &(start, end) in regions.iter().take(10) {
            let mut pos = start;
            while pos + 6 <= end {
                let codon = [seq[pos], seq[pos + 1], seq[pos + 2]];
                for (i, target) in targets.iter().enumerate() {
                    if codon == *target {
                        orf_counts[i] += 1;
                    }
                }
                orf_total += 1;
                pos += 3;
            }
        }

        // Skip windows with too few codons — the ratio is noisy
        if orf_total < MIN_CODONS_PER_WINDOW {
            window_start += step;
            continue;
        }

        // Compute per-window background frequencies
        let mut window_bg_counts = vec![0usize; targets.len()];
        let mut window_bg_total = 0usize;
        for pos in (window_start..window_end.saturating_sub(2)).step_by(3) {
            let codon = [seq[pos], seq[pos + 1], seq[pos + 2]];
            for (i, target) in targets.iter().enumerate() {
                if codon == *target {
                    window_bg_counts[i] += 1;
                }
            }
            window_bg_total += 1;
        }
        let window_bg_freqs: Vec<f64> = window_bg_counts
            .iter()
            .map(|&c| {
                if window_bg_total > 0 {
                    c as f64 / window_bg_total as f64
                } else {
                    0.0
                }
            })
            .collect();

        // Compute signal for this window
        let window_signal = if table == 15 {
            let tag_idx = targets.iter().position(|&t| t == *b"tag").unwrap_or(0);
            let orf_freq = orf_counts[tag_idx] as f64 / orf_total as f64;
            let bg_freq = window_bg_freqs[tag_idx];
            if bg_freq > 0.0 {
                (orf_freq / bg_freq).min(3.0)
            } else {
                0.0
            }
        } else if table == 6 {
            let taa_idx = targets.iter().position(|&t| t == *b"taa");
            let tag_idx = targets.iter().position(|&t| t == *b"tag");
            let taa_ratio = taa_idx.map_or(0.0, |i| {
                let orf_freq = orf_counts[i] as f64 / orf_total as f64;
                if window_bg_freqs[i] > 0.0 {
                    orf_freq / window_bg_freqs[i]
                } else {
                    0.0
                }
            });
            let tag_ratio = tag_idx.map_or(0.0, |i| {
                let orf_freq = orf_counts[i] as f64 / orf_total as f64;
                if window_bg_freqs[i] > 0.0 {
                    orf_freq / window_bg_freqs[i]
                } else {
                    0.0
                }
            });
            taa_ratio.min(tag_ratio).min(3.0)
        } else {
            let mut sum_ratio = 0.0;
            for i in 0..targets.len() {
                let orf_freq = orf_counts[i] as f64 / orf_total as f64;
                let ratio = if window_bg_freqs[i] > 0.0 {
                    orf_freq / window_bg_freqs[i]
                } else {
                    0.0
                };
                sum_ratio += ratio;
            }
            (sum_ratio / targets.len() as f64).min(3.0)
        };

        let mut details = Vec::with_capacity(targets.len());
        for i in 0..targets.len() {
            let orf_freq = orf_counts[i] as f64 / orf_total as f64;
            let ratio = if window_bg_freqs[i] > 0.0 {
                orf_freq / window_bg_freqs[i]
            } else {
                0.0
            };
            details.push(ReassignmentResult {
                codon: targets[i],
                background_freq: window_bg_freqs[i],
                orf_freq,
                ratio,
            });
        }
        window_scores.push((window_signal, details));

        window_start += step;
    }

    if window_scores.is_empty() {
        return (0.0, vec![]);
    }

    // Use the mean of the top 25% of windows — this captures the strongest
    // local signal while ignoring background noise.  We then subtract 0.5
    // to center the baseline (a standard-code genome has top25_mean ≈ 0.5).
    window_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top_n = (window_scores.len() / 4).max(1);
    let top_mean: f64 = window_scores[..top_n].iter().map(|(s, _)| s).sum::<f64>() / top_n as f64;
    let final_signal = (top_mean - 0.5).clamp(0.0, 3.0);

    // Return details from the best window (for reporting)
    let best_details = window_scores[0].1.clone();
    (final_signal, best_details)
}

/// Compute the reassignment signal for a candidate table.
///
/// The core idea: on the correct alternative table, reassigned codons appear
/// inside candidate-table ORFs at roughly the same frequency as they appear
/// in the whole genome (they are being read as sense codons).  On a
/// standard-code genome they are strongly depleted inside ORFs because they
/// are stops.
///
/// For tables 15 and 6 (which may be mosaic), we use a sliding window to
/// find the strongest local signal.  For tables 4 and 25 we use the whole
/// genome because TGA readthrough is typically genome-wide.
fn compute_reassignment_signal(seq: &[u8], table: u8) -> (f64, Vec<ReassignmentResult>) {
    let targets: Vec<[u8; 3]> = reassigned_codons(table)
        .iter()
        .map(|&c| [c[0], c[1], c[2]])
        .collect();
    if targets.is_empty() {
        return (1.0, vec![]);
    }

    // For mosaic-capable tables (15, 6), use sliding windows.
    // Window size scales with sequence length (3–9 kb) to capture local signals
    // in mosaic genomes like crAssphage while maintaining enough codons per
    // window for stable statistics.
    if table == 15 || table == 6 {
        let window_size = seq.len().clamp(3000, 9000);
        let step = window_size / 5;
        return sliding_window_signal(seq, table, &targets, window_size, step);
    }

    // For tables 4 and 25, use whole-genome ORFs
    let (orf_total, orf_counts) = count_in_candidate_orfs(seq, table, &targets, 20);
    let (bg_counts, bg_total) = background_freqs(seq, &targets);

    let mut details = Vec::with_capacity(targets.len());
    let mut ratios = Vec::with_capacity(targets.len());
    for (i, _target) in targets.iter().enumerate() {
        let bg_freq = if bg_total > 0 {
            bg_counts[i] as f64 / bg_total as f64
        } else {
            0.0
        };
        let orf_freq = if orf_total > 0 {
            orf_counts[i] as f64 / orf_total as f64
        } else {
            0.0
        };
        let ratio = if bg_freq > 0.0 {
            orf_freq / bg_freq
        } else {
            0.0
        };
        details.push(ReassignmentResult {
            codon: targets[i],
            background_freq: bg_freq,
            orf_freq,
            ratio,
        });
        ratios.push(ratio);
    }

    let signal = match table {
        4 => {
            // TGA → Trp: compare TGA to TGG in candidate ORFs
            let tga_idx = targets.iter().position(|&t| t == *b"tga").unwrap_or(0);
            let tga_count = orf_counts[tga_idx];
            let mut tgg_count = 0usize;
            for &(start, end) in {
                let stops = stop_codons(4);
                let mut regions: Vec<(usize, usize)> = Vec::new();
                for frame in 0..3 {
                    let mut region_start = frame;
                    let mut pos = frame;
                    while pos + 3 <= seq.len() {
                        let codon = &seq[pos..pos + 3];
                        if is_stop(codon, stops) {
                            let len = pos.saturating_sub(region_start);
                            if len >= 90 {
                                regions.push((region_start, pos));
                            }
                            region_start = pos + 3;
                        }
                        pos += 3;
                    }
                    let len = seq.len().saturating_sub(region_start);
                    if len >= 90 {
                        regions.push((region_start, seq.len()));
                    }
                }
                regions.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then_with(|| a.0.cmp(&b.0)));
                let mut v = Vec::new();
                for &(s, e) in regions.iter().take(20) {
                    v.push((s, e));
                }
                v
            }
            .iter()
            {
                let mut pos = start;
                while pos + 6 <= end {
                    if &seq[pos..pos + 3] == b"tgg" {
                        tgg_count += 1;
                    }
                    pos += 3;
                }
            }
            if tgg_count > 0 {
                (tga_count as f64 / tgg_count as f64).min(3.0)
            } else {
                ratios.get(tga_idx).copied().unwrap_or(0.0).min(3.0)
            }
        }
        25 => {
            // TGA → Gly: TGA / (GGA+GGC+GGG+GGT) in candidate ORFs
            let tga_idx = targets.iter().position(|&t| t == *b"tga").unwrap_or(0);
            let tga_count = orf_counts[tga_idx];
            let gly_codons: &[[u8; 3]] = &[*b"gga", *b"ggc", *b"ggg", *b"ggt"];
            let mut gly_count = 0usize;
            for &(start, end) in {
                let stops = stop_codons(25);
                let mut regions: Vec<(usize, usize)> = Vec::new();
                for frame in 0..3 {
                    let mut region_start = frame;
                    let mut pos = frame;
                    while pos + 3 <= seq.len() {
                        let codon = &seq[pos..pos + 3];
                        if is_stop(codon, stops) {
                            let len = pos.saturating_sub(region_start);
                            if len >= 90 {
                                regions.push((region_start, pos));
                            }
                            region_start = pos + 3;
                        }
                        pos += 3;
                    }
                    let len = seq.len().saturating_sub(region_start);
                    if len >= 90 {
                        regions.push((region_start, seq.len()));
                    }
                }
                regions.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then_with(|| a.0.cmp(&b.0)));
                let mut v = Vec::new();
                for &(s, e) in regions.iter().take(20) {
                    v.push((s, e));
                }
                v
            }
            .iter()
            {
                let mut pos = start;
                while pos + 6 <= end {
                    let codon = [seq[pos], seq[pos + 1], seq[pos + 2]];
                    if gly_codons.contains(&codon) {
                        gly_count += 1;
                    }
                    pos += 3;
                }
            }
            if gly_count > 0 {
                (tga_count as f64 / gly_count as f64).min(3.0)
            } else {
                ratios.get(tga_idx).copied().unwrap_or(0.0).min(3.0)
            }
        }
        _ => 1.0,
    };

    (signal, details)
}

// ---------------------------------------------------------------------------
// 3. Tie-breaker between tables 1 and 11
// ---------------------------------------------------------------------------

/// Count how many table-11 seed ORFs begin with a start codon that is
/// exclusive to table 11 (ATT, ATC, ATA, GTG).
fn table11_exclusive_start_count(seq: &[u8]) -> usize {
    let stops11 = stop_codons(11);
    let exclusive: &[[u8; 3]] = &[*b"att", *b"atc", *b"ata", *b"gtg"];

    let mut count = 0usize;
    for frame in 0..3 {
        let mut region_start = frame;
        let mut pos = frame;
        while pos + 3 <= seq.len() {
            let codon = &seq[pos..pos + 3];
            if is_stop(codon, stops11) {
                let len = pos.saturating_sub(region_start);
                if len >= 90 {
                    let start_codon = [
                        seq[region_start],
                        seq[region_start + 1],
                        seq[region_start + 2],
                    ];
                    if exclusive.contains(&start_codon) {
                        count += 1;
                    }
                }
                region_start = pos + 3;
            }
            pos += 3;
        }
        let len = seq.len().saturating_sub(region_start);
        if len >= 90 {
            let start_codon = [
                seq[region_start],
                seq[region_start + 1],
                seq[region_start + 2],
            ];
            if exclusive.contains(&start_codon) {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// 4. Composite scoring and ranking
// ---------------------------------------------------------------------------

/// Score for a single candidate table.
#[allow(dead_code)]
pub struct TableScore {
    pub table: u8,
    pub mean_orf_len: f64,        // informational only (raw nt)
    pub mol_ratio: f64,           // mean_orf_len / baseline_mol
    pub reassignment_signal: f64, // table-specific signal
    pub composite: f64,
    /// Per-codon breakdown for the report.
    pub codon_details: Vec<ReassignmentResult>,
}

/// Score all candidate tables and return them sorted by composite (descending).
pub fn score_tables(seq: &[u8], min_orf_len: usize) -> Vec<TableScore> {
    if seq.len() < MIN_SEQ_LEN {
        return Vec::new();
    }

    let baseline_mol = mean_orf_length(seq, 11, min_orf_len).max(1.0);
    let baseline_max = max_orf_length(seq, 11, min_orf_len).max(1);
    let t11_exclusive_count = table11_exclusive_start_count(seq);

    let mut scores = Vec::with_capacity(CANDIDATE_TABLES.len());
    for &table in CANDIDATE_TABLES {
        let mol = mean_orf_length(seq, table, min_orf_len);
        let max_len = max_orf_length(seq, table, min_orf_len);
        let mol_ratio = mol / baseline_mol;
        let max_ratio = max_len as f64 / baseline_max as f64;

        let (signal, details) = compute_reassignment_signal(seq, table);

        // Composite formula:
        // - mol_ratio captures the structural advantage of having fewer stops.
        // - reassignment_signal tells us whether reassigned codons appear
        //   freely inside candidate-table ORFs (high = supports alternative).
        // - max_ratio adds a bonus when the candidate table produces
        //   dramatically longer max ORFs than table 11.
        // - For tables 1 and 11 (no reassigned codons), signal is always 1.0.
        let composite = if table == 1 || table == 11 {
            mol_ratio * signal
        } else {
            // Boost tables that have both elevated max_ratio AND strong signal.
            // The boost is less aggressive for synthetic/short sequences where
            // max_ratio may not be elevated even when the alternative code is correct.
            let boost = if signal > 0.6 {
                1.0
            } else if signal > 0.3 {
                if max_ratio > 2.0 {
                    1.0
                } else {
                    0.7
                }
            } else if max_ratio > 2.5 {
                0.7
            } else {
                0.4
            };
            mol_ratio * signal * boost
        };

        scores.push(TableScore {
            table,
            mean_orf_len: mol,
            mol_ratio,
            reassignment_signal: signal,
            composite,
            codon_details: details,
        });
    }

    // Sort by composite descending, with tie-breaker for tables 1 vs 11
    scores.sort_by(|a, b| {
        let cmp = b.composite.partial_cmp(&a.composite).unwrap();
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
        // Composites are equal — apply tie-breaker if both are tables 1/11
        let a_is_baseline = a.table == 1 || a.table == 11;
        let b_is_baseline = b.table == 1 || b.table == 11;
        if a_is_baseline && b_is_baseline && t11_exclusive_count > 0 {
            if a.table == 11 {
                std::cmp::Ordering::Less
            } else if b.table == 11 {
                std::cmp::Ordering::Greater
            } else {
                a.table.cmp(&b.table)
            }
        } else {
            a.table.cmp(&b.table)
        }
    });

    scores
}

// ---------------------------------------------------------------------------
// 5. Human-readable report
// ---------------------------------------------------------------------------

/// Format a detection report for stderr.
pub fn format_report(scores: &[TableScore], seq_id: &str, seq_len: usize) -> String {
    let mut lines = Vec::new();
    let sep = "─".repeat(78);
    lines.push(sep.clone());
    lines.push(format!(
        "Codon table detection: seq={} len={} nt (first record)",
        seq_id, seq_len
    ));
    lines.push(sep.clone());
    lines.push(format!(
        "{:>4}  {:>5}  {:<40}  {:>10}  {:>15}  {:>9}",
        "Rank", "Table", "Name", "ORF ratio", "Reass. signal", "Composite"
    ));

    for (rank, score) in scores.iter().enumerate() {
        let name = table_name(score.table);
        let display_name = if name.len() > 38 {
            format!("{}..", &name[..36])
        } else {
            name.to_string()
        };
        lines.push(format!(
            "{:>4}  {:>5}  {:<40}  {:>10.2}  {:>15.3}  {:>9.2}",
            rank + 1,
            score.table,
            display_name,
            score.mol_ratio,
            score.reassignment_signal,
            score.composite
        ));

        for detail in &score.codon_details {
            lines.push(format!(
                "              └─ {}: bg={:.1}%  orf={:.1}%  ratio={:.2}",
                std::str::from_utf8(&detail.codon).unwrap_or("???"),
                detail.background_freq * 100.0,
                detail.orf_freq * 100.0,
                detail.ratio
            ));
        }
    }

    let confidence = if scores.len() >= 2 && scores[1].composite > 0.0 {
        let ratio = scores[0].composite / scores[1].composite;
        if ratio > 2.0 {
            "high"
        } else if ratio > 1.2 {
            "medium"
        } else {
            "low"
        }
    } else if scores.len() >= 2 && scores[1].composite == 0.0 && scores[0].composite > 0.0 {
        "high"
    } else {
        "low"
    };

    let rec = scores.first().map(|s| s.table).unwrap_or(11);
    let rec_name = table_name(rec);
    lines.push(format!(
        "\nRecommended table: {}  ({})  [confidence: {}]",
        rec, rec_name, confidence
    ));
    lines.push(sep);
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// 6. Batch detection
// ---------------------------------------------------------------------------

/// Summary of table detection for a single genome in a batch run.
#[allow(dead_code)]
pub struct BatchResult {
    pub seq_id: String,
    pub seq_len: usize,
    pub recommended_table: u8,
    pub confidence: String,
    pub top_score: f64,
    pub runner_up_table: u8,
    pub runner_up_score: f64,
    /// All table scores for this genome, sorted by composite.
    pub all_scores: Vec<TableScore>,
}

/// Detect the translation table for every genome in `genomes` and return a
/// summary per genome.  Runs in parallel via Rayon.
pub fn detect_tables_batch(genomes: &[(String, Vec<u8>)], min_orf_len: usize) -> Vec<BatchResult> {
    use rayon::prelude::*;

    genomes
        .par_iter()
        .map(|(id, seq)| {
            let scores = score_tables(seq, min_orf_len);
            if scores.is_empty() {
                // Sequence too short — return a fallback result
                return BatchResult {
                    seq_id: id.clone(),
                    seq_len: seq.len(),
                    recommended_table: 11,
                    confidence: "too_short".to_string(),
                    top_score: 0.0,
                    runner_up_table: 11,
                    runner_up_score: 0.0,
                    all_scores: vec![],
                };
            }

            let confidence = if scores.len() >= 2 && scores[1].composite > 0.0 {
                let ratio = scores[0].composite / scores[1].composite;
                if ratio > 2.0 {
                    "high"
                } else if ratio > 1.2 {
                    "medium"
                } else {
                    "low"
                }
            } else if scores.len() >= 2 && scores[1].composite == 0.0 && scores[0].composite > 0.0 {
                "high"
            } else {
                "low"
            }
            .to_string();

            let runner_up = scores
                .get(1)
                .map(|s| (s.table, s.composite))
                .unwrap_or((11, 0.0));

            BatchResult {
                seq_id: id.clone(),
                seq_len: seq.len(),
                recommended_table: scores[0].table,
                confidence,
                top_score: scores[0].composite,
                runner_up_table: runner_up.0,
                runner_up_score: runner_up.1,
                all_scores: scores,
            }
        })
        .collect()
}

/// Format batch results as a TSV table.
///
/// Columns:
///   seq_id  len  recommended  confidence  top_score  runner_up  runner_up_score
pub fn format_batch_tsv(results: &[BatchResult]) -> String {
    let mut lines = Vec::new();
    lines.push(
        "seq_id\tlen\trecommended\tconfidence\ttop_score\trunner_up\trunner_up_score".to_string(),
    );
    for r in results {
        lines.push(format!(
            "{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}",
            r.seq_id,
            r.seq_len,
            r.recommended_table,
            r.confidence,
            r.top_score,
            r.runner_up_table,
            r.runner_up_score
        ));
    }
    lines.join("\n") + "\n"
}

/// Format batch results with per-table composite scores as a TSV table.
///
/// Columns:
///   seq_id  len  recommended  confidence  t1  t4  t6  t11  t15  t25
#[allow(dead_code)]
pub fn format_batch_matrix_tsv(results: &[BatchResult]) -> String {
    let mut lines = Vec::new();
    lines.push("seq_id\tlen\trecommended\tconfidence\tt1\tt4\tt6\tt11\tt15\tt25".to_string());
    for r in results {
        // Build a map from table -> composite for quick lookup
        let mut composite_by_table = std::collections::HashMap::new();
        for s in &r.all_scores {
            composite_by_table.insert(s.table, s.composite);
        }
        let get = |t: u8| composite_by_table.get(&t).copied().unwrap_or(0.0);
        lines.push(format!(
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            r.seq_id,
            r.seq_len,
            r.recommended_table,
            r.confidence,
            get(1),
            get(4),
            get(6),
            get(11),
            get(15),
            get(25)
        ));
    }
    lines.join("\n") + "\n"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_seq(len: usize, seed: u64) -> Vec<u8> {
        let bases = [b'a', b't', b'c', b'g'];
        let mut seq = Vec::with_capacity(len);
        let mut state = seed;
        for _ in 0..len {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            seq.push(bases[(state % 4) as usize]);
        }
        seq
    }

    fn inject_tga(seq: &mut [u8], fraction: f64, seed: u64) {
        let mut state = seed;
        for pos in (0..seq.len().saturating_sub(2)).step_by(3) {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            if (state % 1000) as f64 / 1000.0 < fraction {
                seq[pos] = b't';
                seq[pos + 1] = b'g';
                seq[pos + 2] = b'a';
            }
        }
    }

    #[test]
    fn test_mean_orf_length_basic() {
        let seq = vec![b'a'; 300];
        let mol = mean_orf_length(&seq, 11, 90);
        assert!(mol > 0.0, "mean ORF length should be positive");
    }

    #[test]
    fn test_mean_orf_length_table4_longer_than_11_with_tga() {
        let mut seq = random_seq(3000, 42);
        inject_tga(&mut seq, 0.05, 99);
        let mol4 = mean_orf_length(&seq, 4, 90);
        let mol11 = mean_orf_length(&seq, 11, 90);
        assert!(
            mol4 > mol11,
            "table 4 should give longer ORFs than 11 when TGA is frequent: 4={} 11={}",
            mol4,
            mol11
        );
    }

    #[test]
    fn test_short_sequence_returns_empty() {
        let seq = vec![b'a'; 200];
        let scores = score_tables(&seq, 90);
        assert!(
            scores.is_empty(),
            "short sequence should return empty scores"
        );
    }

    #[test]
    fn test_confidence_high() {
        let scores = vec![
            TableScore {
                table: 11,
                mean_orf_len: 600.0,
                mol_ratio: 1.0,
                reassignment_signal: 1.0,
                composite: 1.0,
                codon_details: vec![],
            },
            TableScore {
                table: 4,
                mean_orf_len: 200.0,
                mol_ratio: 1.1,
                reassignment_signal: 0.02,
                composite: 0.02,
                codon_details: vec![],
            },
        ];
        let report = format_report(&scores, "test", 1000);
        assert!(report.contains("confidence: high"), "report: {}", report);
    }

    #[test]
    fn test_confidence_low() {
        let scores = vec![
            TableScore {
                table: 11,
                mean_orf_len: 600.0,
                mol_ratio: 1.0,
                reassignment_signal: 1.0,
                composite: 1.0,
                codon_details: vec![],
            },
            TableScore {
                table: 4,
                mean_orf_len: 550.0,
                mol_ratio: 1.05,
                reassignment_signal: 0.9,
                composite: 0.95,
                codon_details: vec![],
            },
        ];
        let report = format_report(&scores, "test", 1000);
        assert!(report.contains("confidence: low"), "report: {}", report);
    }

    #[test]
    fn test_score_tables_ranks_by_composite() {
        let seq = random_seq(3000, 42);
        let scores = score_tables(&seq, 90);
        assert!(!scores.is_empty());
        for i in 1..scores.len() {
            assert!(
                scores[i - 1].composite >= scores[i].composite,
                "scores should be sorted descending"
            );
        }
    }

    #[test]
    fn test_lambda_recommends_table11_or_1() {
        use crate::genome::read_fasta;
        let genomes = read_fasta("../PHANOTATE/tests/NC_001416.1.fasta").unwrap();
        for genome in &genomes {
            let scores = score_tables(&genome.seq, 90);
            assert!(!scores.is_empty());
            let top = scores[0].table;
            assert!(
                top == 11 || top == 1,
                "Lambda (table 11) should score 11 or 1 first, got {}: {:?}",
                top,
                scores
                    .iter()
                    .map(|s| (s.table, s.composite))
                    .collect::<Vec<_>>()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for the bug fix
    // -----------------------------------------------------------------------

    #[test]
    fn regression_spv4_table4_wins() {
        use crate::genome::read_fasta;
        let genomes = read_fasta("../test_genomes/NC_003438.1.fna").unwrap();
        for genome in &genomes {
            let scores = score_tables(&genome.seq, 90);
            assert!(!scores.is_empty());
            assert_eq!(
                scores[0].table,
                4,
                "SpV4 (NC_003438) should recommend table 4, got {:?}",
                scores
                    .iter()
                    .map(|s| (s.table, s.composite, s.reassignment_signal))
                    .collect::<Vec<_>>()
            );
            let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
            assert!(
                t4_score.reassignment_signal > 0.8,
                "SpV4 table-4 signal should be > 0.8, got {}",
                t4_score.reassignment_signal
            );
        }
    }

    #[test]
    fn regression_lambda_table11_wins() {
        use crate::genome::read_fasta;
        let genomes = read_fasta("../PHANOTATE/tests/NC_001416.1.fasta").unwrap();
        for genome in &genomes {
            let scores = score_tables(&genome.seq, 90);
            assert!(!scores.is_empty());
            let top = scores[0].table;
            assert!(
                top == 11 || top == 1,
                "Lambda should recommend table 11 or 1, got {}: {:?}",
                top,
                scores
                    .iter()
                    .map(|s| (s.table, s.composite, s.reassignment_signal))
                    .collect::<Vec<_>>()
            );
            let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
            assert!(
                t4_score.reassignment_signal < 1.0,
                "Lambda table-4 signal should be < 1.0 (TGA is a real stop), got {}",
                t4_score.reassignment_signal
            );
        }
    }

    #[test]
    fn regression_crass_table15_wins() {
        use crate::genome::read_fasta;
        let genomes = read_fasta("../test_genomes/BK025033.fna").unwrap();
        for genome in &genomes {
            let scores = score_tables(&genome.seq, 90);
            assert!(!scores.is_empty());
            assert_eq!(
                scores[0].table,
                15,
                "CrAss (BK025033) should recommend table 15, got {:?}",
                scores
                    .iter()
                    .map(|s| (s.table, s.composite, s.reassignment_signal))
                    .collect::<Vec<_>>()
            );
            let t15_score = scores.iter().find(|s| s.table == 15).unwrap();
            assert!(
                t15_score.reassignment_signal > 0.002,
                "CrAss table-15 signal should be > 0.002, got {}",
                t15_score.reassignment_signal
            );
        }
    }

    #[test]
    fn table11_genome_table4_signal_low() {
        let seq = random_seq(3000, 42);
        let scores = score_tables(&seq, 90);
        let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
        assert!(
            t4_score.reassignment_signal < 1.5,
            "Random seq table-4 signal should be < 1.5, got {}",
            t4_score.reassignment_signal
        );
    }

    #[test]
    fn table4_genome_table4_signal_high() {
        use crate::genome::read_fasta;
        let genomes = read_fasta("../test_genomes/NC_003438.1.fna").unwrap();
        for genome in &genomes {
            let scores = score_tables(&genome.seq, 90);
            let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
            assert!(
                t4_score.reassignment_signal > 0.8,
                "SpV4 table-4 signal should be > 0.8, got {}",
                t4_score.reassignment_signal
            );
        }
    }

    fn synthetic_seq_with_tga_rate(len: usize, tga_rate: f64) -> Vec<u8> {
        let non_stops: &[[u8; 3]] = &[
            *b"atg", *b"aaa", *b"gct", *b"ggc", *b"cgt", *b"tta", *b"cca", *b"gac",
        ];
        let mut seq = Vec::with_capacity(len);
        let mut idx = 0usize;
        while seq.len() < len {
            let codon = non_stops[idx % non_stops.len()];
            seq.extend_from_slice(&codon);
            idx += 1;
        }
        let frame_len = (len / 3) * 3;
        seq.truncate(frame_len);

        let mut state: u64 = 42;
        let n_positions = frame_len / 3;
        let n_tga = (n_positions as f64 * tga_rate).round() as usize;
        let mut positions: Vec<usize> = (0..n_positions).collect();
        for i in (1..positions.len()).rev() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (state as usize) % (i + 1);
            positions.swap(i, j);
        }
        for &pos in &positions[..n_tga.min(positions.len())] {
            let byte_pos = pos * 3;
            seq[byte_pos] = b't';
            seq[byte_pos + 1] = b'g';
            seq[byte_pos + 2] = b'a';
        }
        seq
    }

    #[test]
    fn synthetic_tga_readthrough() {
        let seq = synthetic_seq_with_tga_rate(3000, 0.04);
        let scores = score_tables(&seq, 90);
        let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
        assert!(
            t4_score.reassignment_signal > 0.5,
            "Synthetic TGA-rich seq: table-4 signal should be > 0.5, got {}",
            t4_score.reassignment_signal
        );
    }

    #[test]
    fn synthetic_tga_stop() {
        let seq = synthetic_seq_with_tga_rate(3000, 0.0005);
        let scores = score_tables(&seq, 90);
        let t4_score = scores.iter().find(|s| s.table == 4).unwrap();
        assert!(
            t4_score.reassignment_signal < 2.0,
            "Synthetic TGA-scarce seq: table-4 signal should be < 2.0, got {}",
            t4_score.reassignment_signal
        );
    }
}
