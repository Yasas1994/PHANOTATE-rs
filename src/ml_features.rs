//! Feature extraction for ML-based ORF scoring.
//!
//! Extracts a fixed-length feature vector from each `Orf` suitable for
//! feeding into a lightweight ONNX regression model.

use crate::orf::Orf;

/// Number of features extracted per ORF.
pub const NUM_FEATURES: usize = 34;

/// Fixed-length feature vector for ML inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrfFeatures(pub [f32; NUM_FEATURES]);

/// Column names for the coordinate columns prepended to TSV export rows.
pub const COORD_NAMES: [&str; 2] = ["start", "stop"];

/// Column names for TSV export, in order (after the coordinate columns).
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "log_length",
    "rbs_bin",
    "log_hold",
    "pstop",
    "sd_rbs_score",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
    "non_sd_rbs_score",
    "cscore",
    "cai",
    "gc1",
    "gc2",
    "gc3",
    "overlap_upstream_length",
    "overlap_upstream_same_strand",
    "overlap_downstream_length",
    "overlap_downstream_same_strand",
    "stop_sharing_count",
    "gc_skew",
    "truncation_penalty",
    "upstream_pwm_score",
    "rbs_spacer",
    "heuristic_score",
    "best_alt_pwm_score",
    "pwm_ratio",
    "start_rank",
    "num_alt_starts",
    "start_codon_log_freq",
];

/// Map a 3-base codon to an integer 0..63 (a=0, c=1, g=2, t=3).
#[inline]
fn codon_index(codon: &[u8]) -> usize {
    fn base(b: u8) -> usize {
        match b {
            b'a' | b'A' => 0,
            b'c' | b'C' => 1,
            b'g' | b'G' => 2,
            b't' | b'T' | b'u' | b'U' => 3,
            _ => 0,
        }
    }
    (base(codon[0]) << 4) | (base(codon[1]) << 2) | base(codon[2])
}

/// Amino-acid assignment for translation table 11 (Bacterial/Archaeal).
/// Used only for grouping synonymous codons when computing CAI.
fn aa_table11(idx: usize) -> char {
    const AA: [char; 64] = [
        'F', 'F', 'L', 'L', 'S', 'S', 'S', 'S', 'Y', 'Y', '*', '*', 'C', 'C', '*', 'W', 'L', 'L',
        'L', 'L', 'P', 'P', 'P', 'P', 'H', 'H', 'Q', 'Q', 'R', 'R', 'R', 'R', 'I', 'I', 'I', 'M',
        'T', 'T', 'T', 'T', 'N', 'N', 'K', 'K', 'S', 'S', 'R', 'R', 'V', 'V', 'V', 'V', 'A', 'A',
        'A', 'A', 'D', 'D', 'E', 'E', 'G', 'G', 'G', 'G',
    ];
    AA[idx]
}

// -----------------------------------------------------------------
// Helpers for the Prodigal-inspired features
// -----------------------------------------------------------------

/// Map a base to a 0..4 index for PWM / stop-codon checks.
#[inline]
fn base_idx(b: u8) -> Option<usize> {
    match b {
        b'a' | b'A' => Some(0),
        b'c' | b'C' => Some(1),
        b'g' | b'G' => Some(2),
        b't' | b'T' | b'u' | b'U' => Some(3),
        _ => None,
    }
}

/// Return true if `codon` matches any of the supplied stop codons.
fn is_stop(codon: &[u8], stops: &[Vec<u8>]) -> bool {
    if codon.len() < 3 {
        return false;
    }
    stops
        .iter()
        .any(|s| s.len() == 3 && s[0] == codon[0] && s[1] == codon[1] && s[2] == codon[2])
}

/// Number of codons from the previous in-frame stop to the start codon.
/// `start_idx` is the 0-based index of the first base of the start codon.
fn codons_before_stop(start_idx: usize, seq: &[u8], stops: &[Vec<u8>]) -> f64 {
    if start_idx < 3 {
        return (start_idx as f64) / 3.0;
    }
    let mut pos = start_idx.saturating_sub(3);
    loop {
        if is_stop(&seq[pos..pos + 3], stops) {
            return ((start_idx - pos) as f64) / 3.0;
        }
        if pos < 3 {
            break;
        }
        pos -= 3;
    }
    (start_idx as f64) / 3.0
}

/// Distance (in nt) from the end of the detected RBS motif to the start codon.
fn rbs_spacer(window: &[u8], motif: Option<&str>) -> f64 {
    let motif = match motif {
        None => return 20.0,
        Some(m) => m.as_bytes(),
    };
    if motif.is_empty() || motif.len() > window.len() {
        return 20.0;
    }
    for i in (0..=window.len() - motif.len()).rev() {
        if &window[i..i + motif.len()] == motif {
            return (window.len() - (i + motif.len())) as f64;
        }
    }
    20.0
}

const PWM_LEN: usize = 45;
const PWM_PSEUDO: f64 = 1.0;

/// 45-nt window immediately upstream of a forward start codon.
fn upstream_window_forward(start: usize, dna: &[u8]) -> Option<&[u8]> {
    if start > PWM_LEN {
        Some(&dna[start - PWM_LEN - 1..start - 1])
    } else {
        None
    }
}

/// 45-nt window immediately upstream of a reverse start codon (in RC
/// orientation, so it can be read 5'->3').
fn upstream_window_reverse(start: usize, dna_len: usize, rc_dna: &[u8]) -> Option<&[u8]> {
    let end = dna_len.saturating_sub(start + 2);
    if end >= PWM_LEN {
        Some(&rc_dna[end - PWM_LEN..end])
    } else {
        None
    }
}

/// Compute log-likelihood of an upstream window under a gene PWM vs. a
/// background PWM.
fn score_upstream_window(
    window: &[u8],
    gene_pwm: &[[f64; PWM_LEN]; 4],
    bg_pwm: &[[f64; PWM_LEN]; 4],
    gene_total: f64,
    bg_total: f64,
) -> f64 {
    let mut score = 0.0;
    for (i, &b) in window.iter().enumerate() {
        if let Some(idx) = base_idx(b) {
            let g = gene_pwm[idx][i] / gene_total;
            let bg = bg_pwm[idx][i] / bg_total;
            if g > 0.0 && bg > 0.0 {
                score += (g / bg).ln();
            }
        }
    }
    score
}

/// Compute relative start-site features for ORFs that share the same stop
/// codon and reading frame.
///
/// `pwm_scores` is the upstream PWM score computed for each ORF.  ORFs with
/// the same `(stop, frame)` key are ranked by PWM score (descending); ties
/// are broken by ORF index for deterministic ordering.
fn compute_relative_start_features(orfs: &mut [Orf], pwm_scores: &[f64]) {
    let mut groups: std::collections::HashMap<(usize, i8), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, orf) in orfs.iter().enumerate() {
        groups.entry((orf.stop, orf.frame)).or_default().push(i);
    }

    for (_, mut group) in groups {
        group.sort_by(|&a, &b| {
            pwm_scores[b]
                .total_cmp(&pwm_scores[a])
                .then_with(|| a.cmp(&b))
        });

        let best_score = pwm_scores[group[0]];
        let second_best = if group.len() > 1 {
            pwm_scores[group[1]]
        } else {
            0.0
        };

        for (rank, &idx) in group.iter().enumerate() {
            let orf = &mut orfs[idx];
            let own_score = pwm_scores[idx];

            orf.start_rank = (rank + 1) as f64;
            orf.num_alt_starts = group.len() as f64;

            let best_alt = if group.len() == 1 {
                0.0
            } else if rank == 0 {
                second_best
            } else {
                best_score
            };
            orf.best_alt_pwm_score = best_alt;
            orf.pwm_ratio = if best_alt > 0.0 {
                own_score / best_alt
            } else {
                1.0
            };
        }
    }
}

/// Compute extra ML features that need a full-genome view:
///   - codon adaptation index (CAI)
///   - per-codon-position GC content
///   - overlap length/type with nearest upstream/downstream ORF
///   - number of ORFs sharing the same stop codon
///   - GC skew of the ORF sequence
///   - truncation/sharpening penalty
///   - upstream start-context PWM log-likelihood
///   - distance from detected RBS motif to start codon
///
/// `dna` and `rc_dna` are the lower-case forward and reverse-complement
/// sequences. `stop_codons` is the set of stop codons for the active
/// translation table and is used to compute the truncation penalty.
pub fn compute_extra_ml_features(
    orfs: &mut [Orf],
    dna: &[u8],
    rc_dna: &[u8],
    stop_codons: &[Vec<u8>],
) {
    if orfs.is_empty() {
        return;
    }

    // -----------------------------------------------------------------
    // GC positions and CAI reference counts
    // -----------------------------------------------------------------
    let mut codon_counts = [0.0f64; 64];
    let mut max_per_aa: std::collections::HashMap<char, f64> = std::collections::HashMap::new();

    for orf in orfs.iter_mut() {
        let seq = &orf.seq;
        let mut gc = [0usize; 3];
        let mut n_codons = 0usize;
        for chunk in seq.chunks(3) {
            if chunk.len() < 3 {
                break;
            }
            n_codons += 1;
            for (i, &b) in chunk.iter().enumerate() {
                if b == b'g' || b == b'c' || b == b'G' || b == b'C' {
                    gc[i] += 1;
                }
            }
            if chunk.iter().all(|&b| b != b'n' && b != b'N') {
                let idx = codon_index(chunk);
                codon_counts[idx] += 1.0;
            }
        }
        orf.gc1 = if n_codons > 0 {
            gc[0] as f64 / n_codons as f64
        } else {
            0.0
        };
        orf.gc2 = if n_codons > 0 {
            gc[1] as f64 / n_codons as f64
        } else {
            0.0
        };
        orf.gc3 = if n_codons > 0 {
            gc[2] as f64 / n_codons as f64
        } else {
            0.0
        };
    }

    // Max count per amino acid for CAI weights
    for (idx, &count) in codon_counts.iter().enumerate() {
        let aa = aa_table11(idx);
        if aa == '*' {
            continue;
        }
        let entry = max_per_aa.entry(aa).or_insert(0.0);
        if count > *entry {
            *entry = count;
        }
    }

    // Precompute CAI weights (relative adaptiveness)
    let mut weights = [1.0f64; 64];
    for (idx, &count) in codon_counts.iter().enumerate() {
        let aa = aa_table11(idx);
        if aa == '*' {
            continue;
        }
        let max = max_per_aa.get(&aa).copied().unwrap_or(1.0);
        if max > 0.0 {
            weights[idx] = (count / max).max(1e-6);
        }
    }

    // CAI per ORF
    for orf in orfs.iter_mut() {
        let mut log_sum = 0.0f64;
        let mut n = 0usize;
        for chunk in orf.seq.chunks(3) {
            if chunk.len() < 3 {
                break;
            }
            if chunk.iter().any(|&b| b == b'n' || b == b'N') {
                continue;
            }
            let idx = codon_index(chunk);
            if aa_table11(idx) == '*' {
                continue;
            }
            log_sum += weights[idx].ln();
            n += 1;
        }
        orf.cai = if n > 0 {
            (log_sum / n as f64).exp()
        } else {
            1.0
        };
    }

    // -----------------------------------------------------------------
    // GC skew
    // -----------------------------------------------------------------
    for orf in orfs.iter_mut() {
        let g = orf.seq.iter().filter(|&&b| b == b'g' || b == b'G').count() as f64;
        let c = orf.seq.iter().filter(|&&b| b == b'c' || b == b'C').count() as f64;
        orf.gc_skew = if g + c > 0.0 { (g - c) / (g + c) } else { 0.0 };
    }

    // -----------------------------------------------------------------
    // Truncation / sharpening penalty
    // -----------------------------------------------------------------
    for orf in orfs.iter_mut() {
        if orf.frame > 0 {
            let start_idx = orf.start.saturating_sub(1);
            orf.truncation_penalty = codons_before_stop(start_idx, dna, stop_codons);
        } else {
            let start_idx = dna.len().saturating_sub(orf.start.saturating_add(2));
            let start_idx = start_idx.min(rc_dna.len().saturating_sub(3));
            orf.truncation_penalty = codons_before_stop(start_idx, rc_dna, stop_codons);
        }
    }

    // -----------------------------------------------------------------
    // Upstream start-context PWM (gene vs. background)
    // -----------------------------------------------------------------
    let mean_len =
        orfs.iter().map(|o| o.seq.len()).sum::<usize>() as f64 / orfs.len().max(1) as f64;

    // Precompute start-codon log frequencies using the start codon directly
    // as the hash key (no Vec allocation).  The borrow on `orfs` ends when
    // the scope closes, before the later mutable pass.
    let mut start_codon_log_freqs = vec![0.0f64; orfs.len()];
    {
        let mut start_counts: std::collections::HashMap<&[u8], usize> =
            std::collections::HashMap::new();
        let mut total_confident = 0usize;
        for orf in orfs.iter() {
            if orf.seq.len() as f64 >= mean_len * 0.8
                && (orf.sd_rbs_score > 1.0 || orf.non_sd_rbs_score > 1.0)
            {
                *start_counts.entry(orf.start_codon()).or_insert(0) += 1;
                total_confident += 1;
            }
        }
        for (i, orf) in orfs.iter().enumerate() {
            let freq = if total_confident > 0 {
                (*start_counts.get(orf.start_codon()).unwrap_or(&0) as f64 / total_confident as f64)
                    .max(1e-6)
            } else {
                1e-6
            };
            start_codon_log_freqs[i] = freq.ln();
        }
    }

    let mut gene_pwm = [[PWM_PSEUDO; PWM_LEN]; 4];
    let mut bg_pwm = [[PWM_PSEUDO; PWM_LEN]; 4];
    let mut gene_total = PWM_PSEUDO * 4.0 * PWM_LEN as f64;
    let mut bg_total = PWM_PSEUDO * 4.0 * PWM_LEN as f64;

    for orf in orfs.iter() {
        let window = if orf.frame > 0 {
            upstream_window_forward(orf.start, dna)
        } else {
            upstream_window_reverse(orf.start, dna.len(), rc_dna)
        };
        let Some(win) = window else { continue };
        let high_conf = orf.seq.len() as f64 >= mean_len * 0.8
            && (orf.sd_rbs_score > 1.0 || orf.non_sd_rbs_score > 1.0);
        for (i, &b) in win.iter().enumerate() {
            if let Some(idx) = base_idx(b) {
                bg_pwm[idx][i] += 1.0;
                bg_total += 1.0;
                if high_conf {
                    gene_pwm[idx][i] += 1.0;
                    gene_total += 1.0;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Stop-codon sharing
    // -----------------------------------------------------------------
    let mut stop_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for orf in orfs.iter() {
        *stop_counts.entry(orf.stop).or_insert(0) += 1;
    }
    for orf in orfs.iter_mut() {
        orf.stop_sharing_count = *stop_counts.get(&orf.stop).unwrap_or(&1) as f64;
    }

    // -----------------------------------------------------------------
    // Overlap with nearest upstream/downstream ORF
    // -----------------------------------------------------------------
    let mut indexed: Vec<(usize, usize, i8, usize)> = orfs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let (lo, hi) = if o.start <= o.stop {
                (o.start, o.stop)
            } else {
                (o.stop, o.start)
            };
            (lo, hi, o.frame, i)
        })
        .collect();
    indexed.sort_unstable_by_key(|k| (k.0, k.1));

    for sorted_idx in 0..indexed.len() {
        let (lo, hi, frame, orig_idx) = indexed[sorted_idx];

        // upstream neighbor (next lower coordinate)
        if sorted_idx > 0 {
            let (plo, phi, pframe, _) = indexed[sorted_idx - 1];
            let overlap_len = if hi < plo || phi < lo {
                0
            } else {
                hi.min(phi) - lo.max(plo) + 1
            };
            let same = (frame > 0) == (pframe > 0);
            let orf = &mut orfs[orig_idx];
            orf.overlap_upstream_length = overlap_len as f64;
            orf.overlap_upstream_same_strand = if overlap_len > 0 && same { 1.0 } else { 0.0 };
        }

        // downstream neighbor (next higher coordinate)
        if sorted_idx + 1 < indexed.len() {
            let (nlo, nhi, nframe, _) = indexed[sorted_idx + 1];
            let overlap_len = if hi < nlo || nhi < lo {
                0
            } else {
                hi.min(nhi) - lo.max(nlo) + 1
            };
            let same = (frame > 0) == (nframe > 0);
            let orf = &mut orfs[orig_idx];
            orf.overlap_downstream_length = overlap_len as f64;
            orf.overlap_downstream_same_strand = if overlap_len > 0 && same { 1.0 } else { 0.0 };
        }
    }

    // -----------------------------------------------------------------
    // Upstream PWM score, RBS spacer, and relative start-site features
    // -----------------------------------------------------------------
    let n = orfs.len();
    let mut pwm_scores = vec![0.0f64; n];
    for (i, orf) in orfs.iter_mut().enumerate() {
        let pwm_window = if orf.frame > 0 {
            upstream_window_forward(orf.start, dna)
        } else {
            upstream_window_reverse(orf.start, dna.len(), rc_dna)
        };
        let score = pwm_window.map_or(0.0, |w| {
            score_upstream_window(w, &gene_pwm, &bg_pwm, gene_total, bg_total)
        });
        orf.upstream_pwm_score = score;
        pwm_scores[i] = score;

        let rbs_window = if orf.frame > 0 {
            if orf.start > 21 {
                Some(&dna[orf.start - 22..orf.start - 1])
            } else {
                None
            }
        } else {
            let end = dna.len().saturating_sub(orf.start.saturating_add(2));
            if end >= 21 {
                Some(&rc_dna[end - 21..end])
            } else {
                None
            }
        };
        orf.rbs_spacer = rbs_spacer(rbs_window.unwrap_or(&[]), orf.rbs_motif.as_deref());
    }

    // Group ORFs that share the same stop codon and reading frame.  Within
    // each group the best PWM score identifies the most likely start site;
    // the remaining features quantify how the current ORF compares to its
    // alternatives.
    compute_relative_start_features(orfs, &pwm_scores);
    for (i, orf) in orfs.iter_mut().enumerate() {
        orf.start_codon_log_freq = start_codon_log_freqs[i];
    }
}

impl Orf {
    /// Extract a fixed-length feature vector for ML inference.
    ///
    /// Features are normalised to roughly zero-mean, unit-variance ranges
    /// where possible, and all are cast to `f32` for ONNX compatibility.
    pub fn extract_features(&self) -> OrfFeatures {
        let mut features = [0.0f32; NUM_FEATURES];

        // 0. log(ORF length in nucleotides)
        let length = if self.start <= self.stop {
            self.stop - self.start + 1
        } else {
            self.start - self.stop + 1
        };
        features[0] = (length as f32).ln();

        // 1. RBS bin / raw RBS score (0-27)
        features[1] = self.rbs_score as f32;

        // 2. log(hold) — hold is a product, so log-space is more stable
        features[2] = self.hold.ln() as f32;

        // 3. P(stop) for this ORF
        features[3] = self.pstop as f32;

        // 4. SD RBS likelihood ratio
        features[4] = self.sd_rbs_score as f32;

        // 5-7. Start codon one-hot (ATG, GTG, TTG)
        let sc = self.start_codon();
        features[5] = if sc == b"atg" { 1.0 } else { 0.0 };
        features[6] = if sc == b"gtg" { 1.0 } else { 0.0 };
        features[7] = if sc == b"ttg" { 1.0 } else { 0.0 };

        // 8. GC content of the ORF sequence
        let gc_count = self.seq.iter().filter(|&&b| b == b'g' || b == b'c').count();
        features[8] = if !self.seq.is_empty() {
            gc_count as f32 / self.seq.len() as f32
        } else {
            0.0
        };

        // 9. Forward strand indicator
        features[9] = if self.frame > 0 { 1.0 } else { 0.0 };

        // 10-12. Frame one-hot (absolute value: 1, 2, 3)
        let abs_frame = self.frame.abs();
        features[10] = if abs_frame == 1 { 1.0 } else { 0.0 };
        features[11] = if abs_frame == 2 { 1.0 } else { 0.0 };
        features[12] = if abs_frame == 3 { 1.0 } else { 0.0 };

        // 13. Non-SD motif score
        features[13] = self.non_sd_rbs_score as f32;

        // 14. Prodigal-style raw cscore (sum of hexamer log-odds)
        features[14] = self.cscore as f32;

        // 15. codon adaptation index
        features[15] = self.cai as f32;

        // 16-18. GC content at codon positions 1, 2, 3
        features[16] = self.gc1 as f32;
        features[17] = self.gc2 as f32;
        features[18] = self.gc3 as f32;

        // 19-22. overlap with nearest upstream/downstream ORF
        features[19] = self.overlap_upstream_length as f32;
        features[20] = self.overlap_upstream_same_strand as f32;
        features[21] = self.overlap_downstream_length as f32;
        features[22] = self.overlap_downstream_same_strand as f32;

        // 23. number of ORFs sharing this stop codon
        features[23] = self.stop_sharing_count as f32;

        // 24. GC skew = (G - C) / (G + C)
        features[24] = self.gc_skew as f32;

        // 25. Prodigal-style truncation/sharpening penalty
        features[25] = self.truncation_penalty as f32;

        // 26. Upstream start-context PWM log-likelihood
        features[26] = self.upstream_pwm_score as f32;

        // 27. Distance from detected RBS motif to start codon
        features[27] = self.rbs_spacer as f32;

        // 28. PHANOTATE-style heuristic score (hexamer × start × RBS)
        // This lets a learned model fall back to the strong heuristic signal
        // when the cscore and RBS features already explain the label.
        let start_weight = if sc == b"atg" {
            1.0
        } else if sc == b"gtg" || sc == b"ttg" {
            0.5
        } else {
            0.3
        };
        features[28] =
            (self.coding_potential * start_weight * self.sd_rbs_score * self.non_sd_rbs_score)
                as f32;

        // 29-33. Relative start-site features (same-stop-alternative starts)
        features[29] = self.best_alt_pwm_score as f32;
        features[30] = self.pwm_ratio as f32;
        features[31] = self.start_rank as f32;
        features[32] = self.num_alt_starts as f32;
        features[33] = self.start_codon_log_freq as f32;

        OrfFeatures(features)
    }
}

/// Write feature vectors for a batch of ORFs as a TSV.
///
/// Each row corresponds to one ORF.  The first two columns are the ORF
/// coordinates (`start`, `stop`), followed by the feature values in the
/// order defined by [`FEATURE_NAMES`].  If `include_header` is true, a
/// header row is written first.  When `genome_id` is provided it is appended
/// as an extra column so downstream training scripts can associate rows with
/// their source genome.
pub fn write_features_tsv<W: std::io::Write>(
    writer: &mut W,
    orfs: &[Orf],
    include_header: bool,
    genome_id: Option<&str>,
) -> std::io::Result<()> {
    if include_header {
        let mut headers: Vec<&str> = COORD_NAMES.to_vec();
        headers.extend(FEATURE_NAMES.iter().copied());
        if genome_id.is_some() {
            headers.push("genome_id");
        }
        writeln!(writer, "{}", headers.join("\t"))?;
    }
    for orf in orfs {
        let f = orf.extract_features();
        // Export display coordinates that match PHANOTATE's SCO output:
        // forward genes include the full stop codon; reverse genes show the
        // last base of the start codon as the higher coordinate.
        let (display_start, display_stop) = if orf.frame > 0 {
            (orf.start, orf.stop + 2)
        } else {
            (orf.start + 2, orf.stop)
        };
        let mut vals: Vec<String> = vec![display_start.to_string(), display_stop.to_string()];
        vals.extend(f.0.iter().map(|v| format!("{:.6}", v)));
        if let Some(id) = genome_id {
            vals.push(id.to_string());
        }
        writeln!(writer, "{}", vals.join("\t"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_orf() -> Orf {
        let hold = 0.8;
        Orf {
            start: 100,
            stop: 300,
            frame: 1,
            seq: b"atggctagctagctagc".to_vec(),
            rbs_score: 15,
            pstop: 0.05,
            sd_rbs_score: 2.5,
            non_sd_rbs_score: 1.0,
            rbs_motif: None,
            hold,
            coding_potential: 1.0 / hold,
            weight: -1.0,
            cscore: 0.0,
            cai: 0.0,
            gc1: 0.0,
            gc2: 0.0,
            gc3: 0.0,
            overlap_upstream_length: 0.0,
            overlap_upstream_same_strand: 0.0,
            overlap_downstream_length: 0.0,
            overlap_downstream_same_strand: 0.0,
            stop_sharing_count: 0.0,
            gc_skew: 0.0,
            truncation_penalty: 0.0,
            upstream_pwm_score: 0.0,
            rbs_spacer: 0.0,
            best_alt_pwm_score: 0.0,
            pwm_ratio: 1.0,
            start_rank: 1.0,
            num_alt_starts: 1.0,
            start_codon_log_freq: 0.0,
            wraps_origin: false,
        }
    }

    #[test]
    fn test_feature_length() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0.len(), NUM_FEATURES);
    }

    #[test]
    fn test_log_length() {
        let orf = test_orf();
        let f = orf.extract_features();
        // length = 201, ln(201) ≈ 5.303
        assert!((f.0[0] - 201.0f32.ln()).abs() < 0.01);
    }

    #[test]
    fn test_rbs_bin() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert!((f.0[1] - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_start_codon_onehot() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0[5], 1.0); // ATG
        assert_eq!(f.0[6], 0.0); // GTG
        assert_eq!(f.0[7], 0.0); // TTG
    }

    #[test]
    fn test_start_codon_gtg() {
        let mut orf = test_orf();
        orf.seq = b"gtggctagctagctagc".to_vec();
        let f = orf.extract_features();
        assert_eq!(f.0[5], 0.0); // ATG
        assert_eq!(f.0[6], 1.0); // GTG
        assert_eq!(f.0[7], 0.0); // TTG
    }

    #[test]
    fn test_gc_content() {
        let orf = test_orf();
        let f = orf.extract_features();
        // seq = "atggctagctagctagc" -> g/c count = 9, len = 17
        let expected = 9.0 / 17.0;
        assert!((f.0[8] - expected).abs() < 0.001);
    }

    #[test]
    fn test_frame_onehot() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0[9], 1.0); // fwd
        assert_eq!(f.0[10], 1.0); // frame 1
        assert_eq!(f.0[11], 0.0); // frame 2
        assert_eq!(f.0[12], 0.0); // frame 3
    }

    #[test]
    fn test_reverse_frame() {
        let mut orf = test_orf();
        orf.frame = -2;
        let f = orf.extract_features();
        assert_eq!(f.0[9], 0.0); // not fwd
        assert_eq!(f.0[10], 0.0); // not frame 1
        assert_eq!(f.0[11], 1.0); // frame 2
        assert_eq!(f.0[12], 0.0); // not frame 3
    }

    #[test]
    fn test_sd_rbs_score_feature() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert!((f.0[4] - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_non_sd_rbs_score_feature() {
        let mut orf = test_orf();
        orf.non_sd_rbs_score = 2.0;
        let f = orf.extract_features();
        assert_eq!(f.0.len(), NUM_FEATURES);
        assert!((f.0[13] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tsv_header_includes_coordinates_and_features() {
        let orfs = vec![test_orf()];
        let mut buf = Vec::new();
        write_features_tsv(&mut buf, &orfs, true, None).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let expected_header = "start\tstop\tlog_length\trbs_bin\tlog_hold\tpstop\tsd_rbs_score\tstart_codon_atg\tstart_codon_gtg\tstart_codon_ttg\tgc_content\tframe_fwd\tframe_1\tframe_2\tframe_3\tnon_sd_rbs_score\tcscore";
        assert!(s.starts_with(expected_header));
    }

    #[test]
    fn test_tsv_export() {
        let orfs = vec![test_orf()];
        let mut buf = Vec::new();
        write_features_tsv(&mut buf, &orfs, true, None).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("start\tstop"));
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
        let cols: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(cols.len(), 2 + NUM_FEATURES); // start, stop + features
        assert_eq!(cols[0], "100"); // start
        assert_eq!(cols[1], "302"); // stop (display coord includes stop codon)
    }

    #[test]
    fn test_cscore_feature() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0[14], orf.cscore as f32);
    }

    #[test]
    fn test_tsv_no_header() {
        let orfs = vec![test_orf()];
        let mut buf = Vec::new();
        write_features_tsv(&mut buf, &orfs, false, None).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 1); // just data row
    }

    #[test]
    fn test_relative_start_single_orf() {
        let mut orf = test_orf();
        orf.stop = 300;
        orf.frame = 1;
        let mut orfs = vec![orf];
        let pwm_scores = vec![2.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        assert_eq!(orfs[0].num_alt_starts, 1.0);
        assert_eq!(orfs[0].start_rank, 1.0);
        assert_eq!(orfs[0].pwm_ratio, 1.0);
    }

    #[test]
    fn test_relative_start_higher_pwm() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 150;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();
        let mut orfs = vec![orf_a, orf_b];
        let pwm_scores = vec![5.0, 3.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        assert_eq!(orfs[0].num_alt_starts, 2.0);
        assert_eq!(orfs[0].start_rank, 1.0);
        assert!(orfs[0].pwm_ratio > 1.0);
    }

    #[test]
    fn test_relative_start_lower_pwm() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 150;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();
        let mut orfs = vec![orf_a, orf_b];
        let pwm_scores = vec![3.0, 5.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        assert_eq!(orfs[0].num_alt_starts, 2.0);
        assert_eq!(orfs[0].start_rank, 2.0);
        assert!(orfs[0].pwm_ratio < 1.0);
    }

    #[test]
    fn test_relative_start_tied_pwm() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 150;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();
        let mut orfs = vec![orf_a, orf_b];
        let pwm_scores = vec![5.0, 5.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        // Tie-break by ORF index: index 0 wins rank 1.
        assert_eq!(orfs[0].start_rank, 1.0);
        assert_eq!(orfs[1].start_rank, 2.0);
        assert_eq!(orfs[0].num_alt_starts, 2.0);
        assert_eq!(orfs[1].num_alt_starts, 2.0);
        assert_eq!(orfs[0].best_alt_pwm_score, 5.0);
        assert_eq!(orfs[1].best_alt_pwm_score, 5.0);
        assert_eq!(orfs[0].pwm_ratio, 1.0);
        assert_eq!(orfs[1].pwm_ratio, 1.0);
    }

    #[test]
    fn test_relative_start_three_orfs() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 130;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();
        let mut orf_c = test_orf();
        orf_c.start = 160;
        orf_c.stop = 300;
        orf_c.frame = 1;
        orf_c.seq = b"atggctagctagctagc".to_vec();
        let mut orfs = vec![orf_a, orf_b, orf_c];
        // Scores deliberately out of input order: index 1 is best, then 2, then 0.
        let pwm_scores = vec![3.0, 5.0, 4.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);

        assert_eq!(orfs[0].num_alt_starts, 3.0);
        assert_eq!(orfs[1].num_alt_starts, 3.0);
        assert_eq!(orfs[2].num_alt_starts, 3.0);

        assert_eq!(orfs[0].start_rank, 3.0);
        assert_eq!(orfs[1].start_rank, 1.0);
        assert_eq!(orfs[2].start_rank, 2.0);

        // Each ORF uses the best non-self PWM score as its alternative.
        assert_eq!(orfs[0].best_alt_pwm_score, 5.0);
        assert_eq!(orfs[1].best_alt_pwm_score, 4.0); // second best
        assert_eq!(orfs[2].best_alt_pwm_score, 5.0); // best

        assert!((orfs[0].pwm_ratio - 0.6).abs() < 1e-9);
        assert!((orfs[1].pwm_ratio - 1.25).abs() < 1e-9);
        assert!((orfs[2].pwm_ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_relative_start_zero_or_negative_alt_pwm() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 150;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();

        // Best alternative is zero, so the top-ranked ORF falls back to
        // pwm_ratio = 1.0 (best_alt > 0 guard).
        let mut orfs = vec![orf_a.clone(), orf_b.clone()];
        let pwm_scores = vec![2.0, 0.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        assert_eq!(orfs[0].best_alt_pwm_score, 0.0);
        assert_eq!(orfs[0].pwm_ratio, 1.0);
        assert_eq!(orfs[1].best_alt_pwm_score, 2.0);
        assert_eq!(orfs[1].pwm_ratio, 0.0);

        // Negative alternative: same fallback for the top-ranked ORF.
        let mut orfs = vec![orf_a, orf_b];
        let pwm_scores = vec![2.0, -1.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);
        assert_eq!(orfs[0].best_alt_pwm_score, -1.0);
        assert_eq!(orfs[0].pwm_ratio, 1.0);
        assert_eq!(orfs[1].best_alt_pwm_score, 2.0);
        assert!((orfs[1].pwm_ratio - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_relative_start_independent_groups() {
        let mut orf_a = test_orf();
        orf_a.start = 100;
        orf_a.stop = 300;
        orf_a.frame = 1;
        let mut orf_b = test_orf();
        orf_b.start = 150;
        orf_b.stop = 300;
        orf_b.frame = 1;
        orf_b.seq = b"atggctagctagctagc".to_vec();

        let mut orf_c = test_orf();
        orf_c.start = 400;
        orf_c.stop = 700;
        orf_c.frame = 2;
        let mut orf_d = test_orf();
        orf_d.start = 450;
        orf_d.stop = 700;
        orf_d.frame = 2;
        orf_d.seq = b"atggctagctagctagc".to_vec();

        let mut orfs = vec![orf_a, orf_b, orf_c, orf_d];
        let pwm_scores = vec![5.0, 3.0, 4.0, 6.0];
        compute_relative_start_features(&mut orfs, &pwm_scores);

        // Group 1 (stop 300, frame 1)
        assert_eq!(orfs[0].start_rank, 1.0);
        assert_eq!(orfs[1].start_rank, 2.0);
        assert_eq!(orfs[0].best_alt_pwm_score, 3.0);
        assert_eq!(orfs[1].best_alt_pwm_score, 5.0);

        // Group 2 (stop 700, frame 2)
        assert_eq!(orfs[2].start_rank, 2.0);
        assert_eq!(orfs[3].start_rank, 1.0);
        assert_eq!(orfs[2].best_alt_pwm_score, 6.0);
        assert_eq!(orfs[3].best_alt_pwm_score, 4.0);
    }
}
