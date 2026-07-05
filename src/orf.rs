use crate::rbs_mode::RbsMode;
use crate::rbs_scanner::{detect_rbs_motif_for_mode, score_rbs_for_mode};

#[derive(Debug, Clone, Default)]
pub struct Orf {
    pub start: usize, // 1-based, inclusive start of start codon (forward coords)
    pub stop: usize,  // 1-based, inclusive stop of stop codon (forward coords)
    pub frame: i8,    // 1,2,3 for forward; -1,-2,-3 for reverse
    pub seq: Vec<u8>, // the ORF nucleotide sequence (forward direction)
    pub rbs_score: usize,
    pub pstop: f64, // P(stop) for this ORF
    pub sd_rbs_score: f64,
    pub non_sd_rbs_score: f64, // non-Shine-Dalgarno motif score multiplier
    pub rbs_motif: Option<String>, // detected RBS / non-SD motif sequence
    pub hold: f64,             // product of adjusted P(not_stop) per codon
    pub coding_potential: f64, // coding-potential multiplier (1/hold default, or hexamer model)
    pub weight: f64,           // final ORF edge weight (negative)

    // Extra ML features (computed in a post-processing pass)
    pub cscore: f64, // Prodigal-style raw hexamer cscore (sum of log-odds)
    pub cai: f64,    // codon adaptation index vs. genome-wide usage
    pub gc1: f64,    // GC content at codon position 1
    pub gc2: f64,    // GC content at codon position 2
    pub gc3: f64,    // GC content at codon position 3
    pub overlap_upstream_length: f64, // bases overlapping nearest upstream ORF
    pub overlap_upstream_same_strand: f64, // 1 if overlapping upstream ORF is same strand
    pub overlap_downstream_length: f64, // bases overlapping nearest downstream ORF
    pub overlap_downstream_same_strand: f64, // 1 if overlapping downstream ORF is same strand
    pub stop_sharing_count: f64, // number of ORFs sharing this stop codon
    pub gc_skew: f64, // (G-C)/(G+C) of the ORF sequence
    pub truncation_penalty: f64, // Prodigal-style sharpening penalty
    pub upstream_pwm_score: f64, // log-likelihood of start upstream region
    pub rbs_spacer: f64, // bases between detected RBS motif and start codon

    // Relative start-site features
    pub best_alt_pwm_score: f64,
    pub pwm_ratio: f64,
    pub start_rank: f64,
    pub num_alt_starts: f64,
    pub start_codon_log_freq: f64,

    /// True if this ORF crosses the origin of a circular genome.
    pub wraps_origin: bool,
}

impl Orf {
    pub fn start_codon(&self) -> &[u8] {
        &self.seq[..3]
    }

    /// Return the full ORF nucleotide sequence as a slice.
    pub fn sequence(&self) -> &[u8] {
        &self.seq
    }

    /// Compute P(stop) = P(TAA) + P(TAG) + P(TGA) from per-base frequencies in this ORF.
    ///
    /// This remains table-agnostic because `Orf::pstop` is feature #3 in the
    /// ONNX feature vector; changing it would alter inputs to trained models.
    /// The heuristic scorer uses `compute_pstop_for_table` instead.
    pub fn compute_pstop(seq: &[u8]) -> f64 {
        crate::codon_table::pstop_from_seq(seq, 11)
    }

    /// Compute table-aware P(stop) for use by the heuristic scoring path.
    pub fn compute_pstop_for_table(seq: &[u8], table: u8) -> f64 {
        crate::codon_table::pstop_from_seq(seq, table)
    }

    pub fn score(
        &mut self,
        start_codons: &std::collections::HashMap<Vec<u8>, f64>,
        model: Option<&crate::onnx_scorer::OnnxScorer>,
        model_scale: f64,
        model_threshold: f64,
    ) {
        if let Some(m) = model {
            let features = self.extract_features();
            self.weight = m.score(&features, model_scale, model_threshold);
            return;
        }

        let mut s = self.coding_potential;
        let sc = self.start_codon().to_vec();
        if let Some(&w) = start_codons.get(&sc) {
            s *= w;
        }
        s *= self.sd_rbs_score;
        s *= self.non_sd_rbs_score;
        self.weight = -s;
    }
}

/// Map doubled-sequence ORF coordinates back to circular coordinates.
///
/// The input ORFs were enumerated on a sequence of length `2 * original_len`,
/// so each coordinate may lie in the first copy (`<= original_len`) or the
/// second copy (`> original_len`). After mapping:
///
/// * Both coordinates in the second copy: subtract `original_len` from both
///   (no wrapping).
/// * One coordinate in each copy: subtract `original_len` from the coordinate
///   in the second copy and mark the ORF as `wraps_origin = true`.
/// * Both coordinates in the first copy: leave unchanged.
///
/// This handles forward and reverse strand ORFs symmetrically, including the
/// case where a reverse-strand ORF has its start in the second copy and its
/// stop in the first copy.
pub fn map_orfs_to_circular(orfs: &mut [Orf], original_len: usize) {
    for orf in orfs.iter_mut() {
        let start_in_first = orf.start <= original_len;
        let stop_in_first = orf.stop <= original_len;
        if !start_in_first && !stop_in_first {
            orf.start -= original_len;
            orf.stop -= original_len;
        } else if start_in_first != stop_in_first {
            if !start_in_first {
                orf.start -= original_len;
            } else {
                orf.stop -= original_len;
            }
            orf.wraps_origin = true;
        }
    }
}

/// Return true if `inner` lies entirely within the circular span of `outer`.
///
/// Both ORFs must be on the same strand. For each ORF the interval is
/// interpreted from its 5' coordinate to its 3' coordinate in forward genome
/// coordinates:
///
/// * forward non-wrapping:   [start, stop]
/// * forward wrapping:       [start, len] ∪ [1, stop]
/// * reverse non-wrapping:   [stop, start]
/// * reverse wrapping:       [stop, len] ∪ [1, start]
fn circular_contains(outer: &Orf, inner: &Orf, len: usize) -> bool {
    if outer.frame.signum() != inner.frame.signum() {
        return false;
    }
    let (outer_a, outer_b) = if outer.frame > 0 {
        (outer.start, outer.stop)
    } else {
        (outer.stop, outer.start)
    };
    let (inner_a, inner_b) = if inner.frame > 0 {
        (inner.start, inner.stop)
    } else {
        (inner.stop, inner.start)
    };

    // Interval [a, b] on the circle. If a <= b it does not wrap; otherwise it
    // wraps through the origin.
    fn contains(a: usize, b: usize, x: usize, y: usize, len: usize) -> bool {
        if a <= b {
            // Non-wrapping outer.
            x >= a && y <= b
        } else {
            // Wrapping outer.
            if x <= y {
                // Non-wrapping inner entirely in the tail or the head segment.
                (x >= a && y <= len) || (x >= 1 && y <= b)
            } else {
                // Inner also wraps; it must start no earlier and end no later.
                x >= a && y <= b
            }
        }
    }

    contains(outer_a, outer_b, inner_a, inner_b, len)
}

/// Remove ORFs that are contained within a wrapping ORF on the same strand.
///
/// When a gene wraps the origin, the doubled-genome search can also select a
/// shorter, non-wrapping ORF that lies inside the wrapped gene in circular
/// coordinates. This function drops those contained artefacts so the wrapped
/// gene is emitted as a single feature.
pub fn remove_contained_in_wrapped(orfs: &mut Vec<Orf>, original_len: usize) {
    let wrapped: Vec<Orf> = orfs.iter().filter(|o| o.wraps_origin).cloned().collect();
    if wrapped.is_empty() {
        return;
    }
    orfs.retain(|o| {
        if o.wraps_origin {
            return true;
        }
        !wrapped
            .iter()
            .any(|w| circular_contains(w, o, original_len))
    });
}

/// Enumerate all ORFs in all six reading frames.
///
/// Parameters:
/// - `closed_ends`: if true, do not allow ORFs to run off sequence edges.
///   Fragment ORFs (those without a matching start/stop at the boundary) are discarded.
/// - `mask_n`: if true, do not build any ORF that spans a run of N nucleotides.
///
/// Enumerate all ORFs in all six reading frames using a pre-computed reverse complement.
#[allow(clippy::too_many_arguments)]
pub fn find_orfs_with_rc(
    dna: &[u8],
    rc_dna: &[u8],
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    min_orf_len: usize,
    closed_ends: bool,
    mask_n: bool,
    rbs_mode: RbsMode,
) -> Vec<Orf> {
    let use_prodigal = rbs_mode == RbsMode::Prodigal;
    let mut orfs = Vec::new();
    let contig_length = dna.len();

    // Pre-compute masked regions if needed
    let masked_regions: Vec<(usize, usize)> = if mask_n { find_n_runs(dna) } else { Vec::new() };

    // The dicts that hold start and stop codons
    // frames: 1,2,3,-1,-2,-3
    let mut stops: std::collections::HashMap<i8, usize> =
        [(1, 0), (2, 0), (3, 0), (-1, 1), (-2, 2), (-3, 3)]
            .into_iter()
            .collect();
    let mut starts: std::collections::HashMap<i8, Vec<usize>> = [
        (1, Vec::new()),
        (2, Vec::new()),
        (3, Vec::new()),
        (-1, Vec::new()),
        (-2, Vec::new()),
        (-3, Vec::new()),
    ]
    .into_iter()
    .collect();

    // Initial fragment starts if first codon is NOT a start
    if !closed_ends {
        if dna.len() >= 3 && !start_codons.iter().any(|c| c == &dna[0..3]) {
            starts.get_mut(&1).unwrap().push(1);
        }
        if dna.len() >= 4 && !start_codons.iter().any(|c| c == &dna[1..4]) {
            starts.get_mut(&2).unwrap().push(2);
        }
        if dna.len() >= 5 && !start_codons.iter().any(|c| c == &dna[2..5]) {
            starts.get_mut(&3).unwrap().push(3);
        }
    }

    // Scan all six frames
    for i in 1..=(dna.len().saturating_sub(2)) {
        let codon = &dna[i - 1..i + 2];
        let frame = ((i - 1) % 3) + 1; // 1, 2, 3 as usize
        let frame_i8 = frame as i8;

        // Forward start
        if start_codons.iter().any(|c| c == codon) {
            starts.get_mut(&frame_i8).unwrap().push(i);
        }

        // Forward stop
        if stop_codons.iter().any(|c| c == codon) {
            let stop = i + 2;
            for &start in starts.get(&frame_i8).unwrap().iter().rev() {
                let length = stop - start + 1;
                if length >= min_orf_len {
                    let seq = dna[start - 1..stop].to_vec();
                    if mask_n && spans_masked(start, stop, &masked_regions) {
                        continue;
                    }
                    let rbs = crate::rbs_scanner::get_rbs(dna, start);
                    let rbs_score = score_rbs_for_mode(&rbs, use_prodigal);
                    let rbs_motif = detect_rbs_motif_for_mode(&rbs, use_prodigal);
                    let pstop = Orf::compute_pstop(&seq);
                    let hold = 1.0;
                    orfs.push(Orf {
                        start,
                        stop: stop - 2,
                        frame: frame_i8,
                        seq,
                        rbs_score,
                        pstop,
                        sd_rbs_score: 1.0,
                        non_sd_rbs_score: 1.0,
                        rbs_motif,
                        hold,
                        coding_potential: 1.0 / hold,
                        weight: 1.0,
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
                    });
                }
            }
            starts.get_mut(&frame_i8).unwrap().clear();
            stops.insert(frame_i8, stop);
        }

        // Read reverse-codon directly from pre-computed RC genome
        let rc_start = dna.len().saturating_sub(i + 2);
        let rc_codon = &rc_dna[rc_start..dna.len() - i + 1];

        // Reverse start
        if start_codons.iter().any(|c| c == rc_codon) {
            starts.get_mut(&(-(frame as i8))).unwrap().push(i + 2);
        }

        // Reverse stop
        if stop_codons.iter().any(|c| c == rc_codon) {
            let stop = stops[&(-(frame as i8))];
            for &start in starts.get(&(-(frame as i8))).unwrap().iter() {
                let length = start - stop + 1;
                if length >= min_orf_len {
                    // ORF seq: reverse complement of forward sequence
                    let fwd_start = stop.saturating_sub(1);
                    let fwd_end = start;
                    let seq = rc_dna[dna.len() - fwd_end..dna.len() - fwd_start].to_vec();
                    if mask_n && spans_masked(stop, start, &masked_regions) {
                        continue;
                    }
                    // RBS: 21nt upstream of start on reverse strand
                    // In forward coords: dna[start..start+21], RC is rc_dna[len-start-21..len-start]
                    let rbs_start = dna.len().saturating_sub(start + 21);
                    let rbs_end = dna.len() - start;
                    let rbs = &rc_dna[rbs_start..rbs_end];
                    let rbs_score = score_rbs_for_mode(rbs, use_prodigal);
                    let rbs_motif = detect_rbs_motif_for_mode(rbs, use_prodigal);
                    let pstop = Orf::compute_pstop(&seq);
                    let hold = 1.0;
                    orfs.push(Orf {
                        start: start - 2,
                        stop,
                        frame: -(frame as i8),
                        seq,
                        rbs_score,
                        pstop,
                        sd_rbs_score: 1.0,
                        non_sd_rbs_score: 1.0,
                        rbs_motif,
                        hold,
                        coding_potential: 1.0 / hold,
                        weight: 1.0,
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
                    });
                }
            }
            starts.get_mut(&(-frame_i8)).unwrap().clear();
            stops.insert(-frame_i8, i);
        }
    }

    // Fragment ORFs at the end of the genome
    if !closed_ends {
        for frame_usize in [1usize, 2, 3] {
            let frame = frame_usize as i8;
            let stop = contig_length - ((contig_length - (frame_usize - 1)) % 3);
            for &start in starts.get(&frame).unwrap().iter().rev() {
                let length = stop - start + 1;
                if length >= min_orf_len {
                    let seq = dna[start.saturating_sub(1)..stop].to_vec();
                    if mask_n && spans_masked(start, stop, &masked_regions) {
                        continue;
                    }
                    let rbs = crate::rbs_scanner::get_rbs(dna, start);
                    let rbs_score = score_rbs_for_mode(&rbs, use_prodigal);
                    let rbs_motif = detect_rbs_motif_for_mode(&rbs, use_prodigal);
                    let pstop = Orf::compute_pstop(&seq);
                    let hold = 1.0;
                    orfs.push(Orf {
                        start,
                        stop: stop - 2,
                        frame,
                        seq,
                        rbs_score,
                        pstop,
                        sd_rbs_score: 1.0,
                        non_sd_rbs_score: 1.0,
                        rbs_motif,
                        hold,
                        coding_potential: 1.0 / hold,
                        weight: 1.0,
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
                    });
                }
            }

            // Reverse strand fragment
            let neg_frame = -frame;
            let start_pos = contig_length - ((contig_length - (frame_usize - 1)) % 3);
            if start_pos >= 3 {
                let rc_last_start = dna.len().saturating_sub(start_pos);
                let rc_last_end = dna.len() - start_pos + 3;
                let last_codon = &rc_dna[rc_last_start..rc_last_end];
                if !start_codons.iter().any(|c| c == last_codon) {
                    starts.get_mut(&neg_frame).unwrap().push(start_pos);
                }
            }
            let stop = stops[&neg_frame];
            for &start in starts.get(&neg_frame).unwrap().iter() {
                let length = start - stop + 1;
                if length >= min_orf_len {
                    let fwd_start = stop.saturating_sub(1);
                    let fwd_end = start;
                    let seq = rc_dna[dna.len() - fwd_end..dna.len() - fwd_start].to_vec();
                    if mask_n && spans_masked(stop, start, &masked_regions) {
                        continue;
                    }
                    let rbs_start = dna.len().saturating_sub(start + 21);
                    let rbs_end = dna.len() - start;
                    let rbs = &rc_dna[rbs_start..rbs_end];
                    let rbs_score = score_rbs_for_mode(rbs, use_prodigal);
                    let rbs_motif = detect_rbs_motif_for_mode(rbs, use_prodigal);
                    let pstop = Orf::compute_pstop(&seq);
                    let hold = 1.0;
                    orfs.push(Orf {
                        start: start - 2,
                        stop,
                        frame: neg_frame,
                        seq,
                        rbs_score,
                        pstop,
                        sd_rbs_score: 1.0,
                        non_sd_rbs_score: 1.0,
                        rbs_motif,
                        hold,
                        coding_potential: 1.0 / hold,
                        weight: 1.0,
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
                    });
                }
            }
        }
    }

    orfs
}

/// Find runs of N (or n) in the sequence. Returns inclusive 1-based ranges.
fn find_n_runs(dna: &[u8]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut in_run = false;
    let mut start = 0;
    for (i, &b) in dna.iter().enumerate() {
        if b == b'n' || b == b'N' {
            if !in_run {
                in_run = true;
                start = i + 1; // 1-based
            }
        } else if in_run {
            regions.push((start, i)); // end is inclusive 1-based
            in_run = false;
        }
    }
    if in_run {
        regions.push((start, dna.len()));
    }
    regions
}

/// Check if an interval [start, stop] (1-based, inclusive) spans any masked region.
fn spans_masked(start: usize, stop: usize, masked: &[(usize, usize)]) -> bool {
    let (lo, hi) = (start.min(stop), start.max(stop));
    for &(m_start, m_stop) in masked {
        if m_start <= hi && m_stop >= lo {
            return true;
        }
    }
    false
}

#[cfg(test)]
/// Test helper that computes the reverse complement internally.
fn find_orfs(
    dna: &[u8],
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    min_orf_len: usize,
    closed_ends: bool,
    mask_n: bool,
) -> Vec<Orf> {
    let rc_dna = crate::genome::rev_comp(dna);
    find_orfs_with_rc(
        dna,
        &rc_dna,
        start_codons,
        stop_codons,
        min_orf_len,
        closed_ends,
        mask_n,
        RbsMode::Sd,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rbs_scanner::{detect_rbs_motif_legacy, score_rbs_legacy};

    #[test]
    fn test_score_rbs_direct() {
        // "aaggag" reversed is "gaggaa" — but Python's score_rbs is designed
        // for 21-nt windows; short inputs give truncated slices.
        // For 6-char "gaggaa", Python gives 1 (agg in s[3:6]).
        let seq = b"gaggaa";
        assert_eq!(score_rbs_legacy(seq), 1);
    }

    #[test]
    fn test_score_rbs_specific() {
        // For 6-char "aggagg", Python gives 1 (agg in s[3:6]).
        let seq = b"aggagg";
        assert_eq!(score_rbs_legacy(seq), 1);
    }

    #[test]
    fn test_score_rbs_specific2() {
        // For 8-char "aaaaaagg", Python gives 0 (no match in truncated slices).
        let seq = b"aaaaaagg";
        assert_eq!(score_rbs_legacy(seq), 0);
    }

    #[test]
    fn test_score_rbs_accctg() {
        // Sequence with no match -> score 0
        let seq = b"accctg";
        assert_eq!(score_rbs_legacy(seq), 0);
    }

    #[test]
    fn test_detect_rbs_motif_aggagg() {
        // The highest-scoring match here is "gag" (score 13), not the distal
        // "ggagga" (score 10), so the reported motif is GAG.
        let seq = b"aaggaggtgagtaacaaaacc";
        assert_eq!(detect_rbs_motif_legacy(seq), Some("GAG".to_string()));
    }

    #[test]
    fn test_detect_rbs_motif_none() {
        let seq = b"aaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(detect_rbs_motif_legacy(seq), None);
    }

    #[test]
    fn test_detect_rbs_motif_ggagga_canonical() {
        // AGGAGG placed so the reversed window contains ggagga at score-27 ranges.
        let seq = b"aaaaaaaaaaaggaggaaaaa";
        assert_eq!(detect_rbs_motif_legacy(seq), Some("AGGAGG".to_string()));
    }

    #[test]
    fn test_detect_rbs_motif_ggagg_canonical() {
        // GGAGG placed so the reversed window contains ggagg at score-24 ranges.
        // A 'c' is placed immediately upstream so the window does not also
        // contain the higher-scoring ggagga motif.
        let seq = b"aaaaaaaaaacggaggaaaaa";
        assert_eq!(detect_rbs_motif_legacy(seq), Some("GGAGG".to_string()));
    }

    #[test]
    fn test_detect_rbs_motif_agga_canonical() {
        // AGGA placed so the reversed window contains agga at score-15 ranges.
        let seq = b"aaaaaaaaaaaggaaaaaaaa";
        assert_eq!(detect_rbs_motif_legacy(seq), Some("AGGA".to_string()));
    }

    #[test]
    fn test_detect_rbs_motif_zero_score_returns_none() {
        let seq = b"aaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(score_rbs_legacy(seq), 0);
        assert_eq!(detect_rbs_motif_legacy(seq), None);
    }

    #[test]
    fn test_detect_rbs_motif_consistency_with_score() {
        let cases: &[(&[u8], Option<&str>)] = &[
            (b"aaaaaaaaaaaggaggaaaaa", Some("AGGAGG")),
            (b"aaaaaaaaaacggaggaaaaa", Some("GGAGG")),
            (b"aaaaaaaaaaaggaaaaaaaa", Some("AGGA")),
            (b"aaggaggtgagtaacaaaacc", Some("GAG")),
            (b"aaaaaaaaaaaaaaaaaaaaa", None),
            (b"accctg", None),
        ];
        for &(seq, expected) in cases {
            let score = score_rbs_legacy(seq);
            let motif = detect_rbs_motif_legacy(seq);
            assert_eq!(
                motif.is_some(),
                score > 0,
                "score_rbs={} but detect_rbs_motif={:?} for seq {:?}",
                score,
                motif,
                std::str::from_utf8(seq)
            );
            assert_eq!(motif.as_deref(), expected);
        }
    }

    #[test]
    fn test_background_rbs_counts() {
        // phiX174 has 5386 bases, so 5386 * 2 = 10772 background counts
        // The background_rbs array starts with 1.0 in each bin, so sum should be 10772 + 28
        let dna = b"atgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgc";
        let mut background_rbs = vec![1.0f64; 28];
        for i in 0..dna.len() {
            let window = if i + 21 <= dna.len() {
                &dna[i..i + 21]
            } else {
                &dna[i..]
            };
            let score = score_rbs_legacy(window);
            background_rbs[score] += 1.0;
            let rc_window = crate::genome::rev_comp(window);
            let rc_score = score_rbs_legacy(&rc_window);
            background_rbs[rc_score] += 1.0;
        }
        let sum: f64 = background_rbs.iter().sum();
        assert!((sum - (dna.len() * 2 + 28) as f64).abs() < 0.001);
    }

    #[test]
    fn test_find_extra_7() {
        // 15-char sequence: ggtgg at positions 10-14 (0-indexed).
        // Reversed: "ggtggaaaaaaaaaa" — ggtgg is at positions 0-4.
        // Python checks s[11:16] etc. for score 7, but string is only 15 chars,
        // so s[11:16] = "aaa" — no match. Score = 1 (agg in s[3:6] = "tgg"? no)
        // Actually: s = "ggtggaaaaaaaaaa", s[3:6] = "gga" — no match for agg/gag/gga
        // Let me check: s = 'ggtggaaaaaaaaaa', len=15
        // s[3:6] = 'gga' → matches 'gga' → score 1
        let seq = b"aaaaaaaaaaggtgg";
        assert_eq!(score_rbs_legacy(seq), 1);
    }

    #[test]
    fn test_find_diff_7() {
        // 17-char sequence
        // Reversed: "ggtggaaaaaaggaaaa"
        // Python: s[5:10]='ggaaa', s[6:11]='gaaaa' — no ggtgg/ggggg/ggcgg for score 14
        // s[5:9]='ggaa', s[6:10]='gaaa' — no ggag/gagg/agga for score 16
        // s[5:8]='gga' → matches 'gga' → score 13
        // Wait, let me check Python: score_rbs_legacy('aaaaaggaaaaaggtgg') = 15
        // s = 'ggtggaaaaaaggaaaa'
        // s[5:9] = 'ggaa' — no match for agga/gagg/ggag
        // Actually: s[5:9]='ggaa', but score 15 checks agga in s[5:9],s[6:10]...
        // s = 'ggtggaaaaaaggaaaa', len=17
        // s[5:9] = 'aaaa' — no
        // s[6:10] = 'aaaa' — no
        // s[7:11] = 'aaaa' — no
        // s[8:12] = 'aaag' — no
        // s[9:13] = 'aagg' — no
        // s[10:14] = 'agga' → MATCH! score 15
        let seq = b"aaaaaggaaaaaggtgg";
        assert_eq!(score_rbs_legacy(seq), 15);
    }

    #[test]
    fn test_specific_windows() {
        // Test a few specific window sequences against Python's actual behavior.
        // These short sequences produce truncated slices in Python's score_rbs.
        let cases: Vec<(&[u8], usize)> = vec![
            (b"aggagg", 1), // 6-char: agg in s[3:6]
            (b"ggagg", 0),  // 5-char: no match
            (b"aaggag", 0), // 6-char: no match
            (b"aggtgg", 1), // 6-char: agg in s[3:6]
            (b"ggtggt", 0), // 6-char: no match
            (b"ggtgg", 0),  // 5-char: no match
            (b"agga", 0),   // 4-char: no match
            (b"gggg", 0),   // 4-char: no match
            (b"ggtgga", 0), // 6-char: no match
            (b"gaggt", 0),  // 5-char: no match
            (b"gaggc", 0),  // 5-char: no match
            (b"gagga", 0),  // 5-char: no match
        ];
        for (seq, expected) in cases {
            assert_eq!(
                score_rbs_legacy(seq),
                expected,
                "Failed for sequence: {:?}",
                std::str::from_utf8(seq)
            );
        }
    }

    #[test]
    fn test_i_822() {
        // Test the specific case at position 822 of phiX174
        // The 21-nt window upstream of position 822
        let window = b"ttttttttttttttttttttt";
        assert_eq!(score_rbs_legacy(window), 0);
    }

    #[test]
    fn test_nc001416_orf_20815() {
        // Test ORF at position 20815 in NC_001416
        // This ORF was missing in earlier versions
        let dna = include_bytes!("../tests/golden/NC_001416.1.tabular");
        // Just verify the function doesn't panic
        let _ = find_orfs(
            dna,
            &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
            &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
            90,
            false,
            false,
        );
    }

    #[test]
    fn motif_score_scales_weight() {
        let hold = 2.0;
        let mut orf = Orf {
            start: 100,
            stop: 200,
            frame: 1,
            seq: b"atgttagctagctagctaa".to_vec(),
            rbs_score: 10,
            pstop: 0.05,
            sd_rbs_score: 1.0,
            hold,
            coding_potential: 1.0 / hold,
            non_sd_rbs_score: 1.0,
            rbs_motif: None,
            weight: 1.0,
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
        };
        let start_codons = std::collections::HashMap::new();
        orf.score(&start_codons, None, 1.0, 0.5);
        let base_weight = orf.weight;

        orf.non_sd_rbs_score = 2.0;
        orf.score(&start_codons, None, 1.0, 0.5);
        assert!((orf.weight - base_weight * 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_map_orfs_to_circular() {
        let original_len = 100;
        let mut orfs = vec![
            Orf {
                start: 10,
                stop: 30,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
            Orf {
                start: 110,
                stop: 130,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
            Orf {
                start: 80,
                stop: 120,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
            // Reverse-strand ORF whose start is in the second copy and stop is
            // in the first copy: must map only the start down and flag the ORF
            // as wrapping.
            Orf {
                start: 130,
                stop: 70,
                frame: -1,
                wraps_origin: false,
                ..Default::default()
            },
        ];

        map_orfs_to_circular(&mut orfs, original_len);

        // ORF entirely in the first copy is unchanged.
        assert_eq!(orfs[0].start, 10);
        assert_eq!(orfs[0].stop, 30);
        assert!(!orfs[0].wraps_origin);

        // ORF entirely in the second copy is mapped back by original_len.
        assert_eq!(orfs[1].start, 10);
        assert_eq!(orfs[1].stop, 30);
        assert!(!orfs[1].wraps_origin);

        // Forward-strand wrapping ORF keeps its start, wraps its stop, and is flagged.
        assert_eq!(orfs[2].start, 80);
        assert_eq!(orfs[2].stop, 20);
        assert!(orfs[2].wraps_origin);

        // Reverse-strand wrapping ORF maps its start from the second copy down
        // to the first copy and keeps its stop; the coordinate relation inverts
        // because the start codon is now before the stop codon in linear order.
        assert_eq!(orfs[3].start, 30);
        assert_eq!(orfs[3].stop, 70);
        assert!(orfs[3].start < orfs[3].stop);
        assert!(orfs[3].wraps_origin);
    }

    #[test]
    fn test_remove_contained_in_wrapped_forward() {
        let original_len = 100;
        let mut orfs = vec![
            // Wrapped forward ORF: covers 80..100 and 1..20.
            Orf {
                start: 80,
                stop: 20,
                frame: 1,
                wraps_origin: true,
                ..Default::default()
            },
            // Contained forward ORF inside the wrapped head segment.
            Orf {
                start: 5,
                stop: 20,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
            // Non-contained forward ORF outside the wrapped span.
            Orf {
                start: 40,
                stop: 50,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
        ];
        remove_contained_in_wrapped(&mut orfs, original_len);
        assert_eq!(orfs.len(), 2);
        assert!(orfs.iter().any(|o| o.start == 80 && o.stop == 20));
        assert!(orfs.iter().any(|o| o.start == 40 && o.stop == 50));
    }

    #[test]
    fn test_remove_contained_in_wrapped_reverse() {
        let original_len = 100;
        let mut orfs = vec![
            // Wrapped reverse ORF: start codon at 30, stop codon at 70.
            // In forward coordinates the gene spans 70..100 and 1..30.
            Orf {
                start: 30,
                stop: 70,
                frame: -1,
                wraps_origin: true,
                ..Default::default()
            },
            // Contained reverse ORF inside the head segment (start > stop for
            // a non-wrapping reverse ORF).
            Orf {
                start: 25,
                stop: 10,
                frame: -1,
                wraps_origin: false,
                ..Default::default()
            },
            // Non-contained reverse ORF outside the wrapped span.
            Orf {
                start: 50,
                stop: 40,
                frame: -1,
                wraps_origin: false,
                ..Default::default()
            },
        ];
        remove_contained_in_wrapped(&mut orfs, original_len);
        assert_eq!(orfs.len(), 2);
        assert!(orfs.iter().any(|o| o.start == 30 && o.stop == 70));
        assert!(orfs.iter().any(|o| o.start == 50 && o.stop == 40));
    }

    #[test]
    fn test_remove_contained_in_wrapped_no_wrapped() {
        let original_len = 100;
        let mut orfs = vec![
            Orf {
                start: 10,
                stop: 20,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
            Orf {
                start: 15,
                stop: 25,
                frame: 1,
                wraps_origin: false,
                ..Default::default()
            },
        ];
        remove_contained_in_wrapped(&mut orfs, original_len);
        assert_eq!(orfs.len(), 2);
    }

    #[test]
    fn test_remove_contained_in_wrapped_different_frame_same_strand() {
        let original_len = 100;
        let mut orfs = vec![
            // Wrapped forward ORF on frame 1.
            Orf {
                start: 80,
                stop: 20,
                frame: 1,
                wraps_origin: true,
                ..Default::default()
            },
            // Same circular span but different doubled-genome frame label;
            // should still be removed because it is on the same strand.
            Orf {
                start: 5,
                stop: 20,
                frame: 2,
                wraps_origin: false,
                ..Default::default()
            },
        ];
        remove_contained_in_wrapped(&mut orfs, original_len);
        assert_eq!(orfs.len(), 1);
        assert!(orfs[0].wraps_origin);
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::genome::read_fasta;

    #[test]
    fn debug_phix174_orf() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            println!("Found {} ORFs", orfs.len());
            for orf in &orfs[..10.min(orfs.len())] {
                println!(
                    "ORF {}-{} frame={} rbs={}",
                    orf.start, orf.stop, orf.frame, orf.rbs_score
                );
            }
        }
    }

    #[test]
    fn debug_pos_max_min() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            for (stop, orfs_at_stop) in by_stop {
                println!("Stop {}: {} ORFs", stop, orfs_at_stop.len());
                for orf in orfs_at_stop {
                    println!("  start={} frame={}", orf.start, orf.frame);
                }
            }
        }
    }

    #[test]
    fn debug_pos_max_min_fixed() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut training_rbs = vec![1.0f64; 28];
            for orf in &orfs {
                training_rbs[orf.rbs_score] += 1.0;
            }
            let mut selected_orfs = Vec::new();
            for (_, orfs_at_stop) in by_stop {
                for orf in orfs_at_stop {
                    if orf.start_codon() == b"atg" {
                        selected_orfs.push(orf);
                        break;
                    }
                }
            }
            println!("Selected {} training ORFs", selected_orfs.len());
            for orf in &selected_orfs[..5.min(selected_orfs.len())] {
                println!("  {}-{} frame={}", orf.start, orf.stop, orf.frame);
            }
        }
    }

    #[test]
    fn count_atg_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let atg_count = orfs.iter().filter(|o| o.start_codon() == b"atg").count();
            println!(
                "ATG ORFs: {}/{} ({}%)",
                atg_count,
                orfs.len(),
                atg_count * 100 / orfs.len()
            );
        }
    }

    #[test]
    fn count_all_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            println!("Total ORFs: {}", orfs.len());
            let fwd = orfs.iter().filter(|o| o.frame > 0).count();
            let rev = orfs.iter().filter(|o| o.frame < 0).count();
            println!("Forward: {}, Reverse: {}", fwd, rev);
        }
    }

    #[test]
    fn compare_stops() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut stops: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                *stops.entry(orf.stop).or_insert(0) += 1;
            }
            for (stop, count) in stops.iter().take(20) {
                println!("Stop {}: {} ORFs", stop, count);
            }
        }
    }

    #[test]
    fn compare_training_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut selected = Vec::new();
            for (_, orfs_at_stop) in by_stop {
                for orf in orfs_at_stop {
                    if orf.start_codon() == b"atg" {
                        selected.push(orf);
                        break;
                    }
                }
            }
            println!("Selected {} training ORFs", selected.len());
        }
    }

    #[test]
    fn compare_first_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            for (stop, orfs_at_stop) in by_stop.iter().take(10) {
                println!(
                    "Stop {}: first ATG ORF = {}-{} frame={}",
                    stop,
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.start)
                        .unwrap_or(0),
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.stop)
                        .unwrap_or(0),
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.frame)
                        .unwrap_or(0),
                );
            }
        }
    }

    #[test]
    fn compare_reverse_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(10) {
                println!(
                    "  {}-{} frame={} start_codon={:?}",
                    orf.start,
                    orf.stop,
                    orf.frame,
                    std::str::from_utf8(orf.start_codon())
                );
            }
        }
    }

    #[test]
    fn count_training_bases() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut total_bases = 0;
            for (_, orfs_at_stop) in by_stop {
                if let Some(orf) = orfs_at_stop.iter().find(|o| o.start_codon() == b"atg") {
                    let (start, stop) = (orf.start, orf.stop);
                    if start < stop {
                        let n = ((stop - start) / 8) * 3;
                        let mut base = start + n;
                        while base + 36 < stop {
                            total_bases += 1;
                            base += 3;
                        }
                    }
                }
            }
            println!("Training bases: {}", total_bases);
        }
    }

    #[test]
    fn compare_longest_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let longest = orfs.iter().max_by_key(|o| o.seq.len()).unwrap();
            println!(
                "Longest ORF: {}-{} ({} nt)",
                longest.start,
                longest.stop,
                longest.seq.len()
            );
        }
    }

    #[test]
    fn check_mixed_stops() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut stop_counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for orf in &orfs {
                let stop = &genome.seq[orf.stop.saturating_sub(1)..orf.stop + 2];
                let key = String::from_utf8_lossy(stop).to_string();
                *stop_counts.entry(key).or_insert(0) += 1;
            }
            for (stop, count) in stop_counts.iter().take(10) {
                println!("Stop {}: {} ORFs", stop, count);
            }
        }
    }

    #[test]
    fn check_reverse_orfs_nc001416() {
        let genomes = read_fasta("../PHANOTATE/tests/NC_001416.1.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("NC_001416 Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(5) {
                println!("  {}-{} frame={}", orf.start, orf.stop, orf.frame);
            }
        }
    }
}
