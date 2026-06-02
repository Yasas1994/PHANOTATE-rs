//! Overlapping gene detection for PHANOTATE-rs.
//!
//! After the primary shortest path is computed, this module scans all ORFs
//! that were excluded from the path due to overlap. Each excluded ORF is
//! scored for "overlap plausibility" using multiple signals. High-confidence
//! overlaps are returned as secondary annotations.

use crate::graph::Node;
use crate::orf::Orf;

/// Information about an overlap between two ORFs.
#[derive(Debug, Clone)]
pub struct OverlapInfo {
    /// The overlapping ORF (the one not in the primary path).
    pub orf: Orf,
    /// The primary-path ORF that it overlaps.
    pub primary_orf: Orf,
    /// Length of the overlapping region in nucleotides.
    pub overlap_len: usize,
    /// Frame relationship: +1, +2, -1, -2, or 0 for same-frame.
    /// +1 means the overlapping ORF is 1 nt shifted (e.g., frame 1 vs frame 2).
    /// +2 means 2 nt shifted (e.g., frame 1 vs frame 3).
    pub frame_shift: i8,
    /// Same strand (true) or opposite strand (false).
    pub same_strand: bool,
    /// Normalized overlap score [0, 1]. Higher = more likely real.
    pub score: f64,
}

/// Find overlapping genes that were excluded from the primary path.
///
/// For each ORF not in the primary path, check if it overlaps any ORF
/// in the path. If so, compute an overlap plausibility score.
///
/// Returns ORFs that pass the threshold, sorted by score (highest first).
pub fn find_overlapping_genes(
    path_edges: &[(Node, Node, f64)],
    orfs: &[Orf],
    threshold: f64,
) -> Vec<OverlapInfo> {
    // Build a set of primary-path ORFs for fast lookup.
    let primary_orfs = collect_primary_orfs(path_edges, orfs);

    // Build a set of ORFs that are in the primary path (by identity).
    let in_path: std::collections::HashSet<(usize, usize, i8)> = primary_orfs
        .iter()
        .map(|o| (o.start, o.stop, o.frame))
        .collect();

    let mut overlaps = Vec::new();

    for orf in orfs {
        // Skip ORFs already in the primary path.
        if in_path.contains(&(orf.start, orf.stop, orf.frame)) {
            continue;
        }

        // Find the best overlapping primary ORF.
        if let Some((primary, overlap_len)) = find_best_overlap(orf, &primary_orfs) {
            let score = compute_overlap_score(orf, &primary, overlap_len);
            if score >= threshold {
                let frame_shift = compute_frame_shift(orf.frame, primary.frame);
                let same_strand = orf.frame.signum() == primary.frame.signum();

                overlaps.push(OverlapInfo {
                    orf: orf.clone(),
                    primary_orf: primary.clone(),
                    overlap_len,
                    frame_shift,
                    same_strand,
                    score,
                });
            }
        }
    }

    // Sort by score descending.
    overlaps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Remove duplicates: if the same overlapping ORF appears multiple times
    // (overlapping multiple primary ORFs), keep only the highest-scoring one.
    let mut seen = std::collections::HashSet::new();
    overlaps
        .into_iter()
        .filter(|o| seen.insert((o.orf.start, o.orf.stop, o.orf.frame)))
        .collect()
}

/// Collect all ORFs that are in the primary path.
fn collect_primary_orfs(path_edges: &[(Node, Node, f64)], orfs: &[Orf]) -> Vec<Orf> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (left, right, _weight) in path_edges {
        // Forward strand: start -> stop
        if left.node_type == "start" && right.node_type == "stop" && left.frame > 0 {
            if let Some(orf) = orfs
                .iter()
                .find(|o| o.start == left.position && o.stop == right.position && o.frame == left.frame)
            {
                if seen.insert((orf.start, orf.stop, orf.frame)) {
                    result.push(orf.clone());
                }
            }
        }
        // Reverse strand: stop -> start
        else if left.node_type == "stop" && right.node_type == "start" && left.frame < 0 {
            if let Some(orf) = orfs
                .iter()
                .find(|o| o.stop == left.position && o.start == right.position && o.frame == left.frame)
            {
                if seen.insert((orf.start, orf.stop, orf.frame)) {
                    result.push(orf.clone());
                }
            }
        }
    }

    result
}

/// Find the primary ORF that has the longest overlap with the given ORF.
fn find_best_overlap(orf: &Orf, primary_orfs: &[Orf]) -> Option<(Orf, usize)> {
    let mut best: Option<(Orf, usize)> = None;

    for primary in primary_orfs {
        let overlap = compute_overlap_length(orf, primary);
        if overlap > 0 && best.as_ref().map_or(true, |(_, best_len)| overlap > *best_len) {
            best = Some((primary.clone(), overlap));
        }
    }

    best
}

/// Compute the overlap length between two ORFs in nucleotides.
fn compute_overlap_length(orf1: &Orf, orf2: &Orf) -> usize {
    // Convert to genomic coordinates.
    let (s1, e1) = orf_coords(orf1);
    let (s2, e2) = orf_coords(orf2);

    // Check if intervals overlap.
    let overlap_start = s1.max(s2);
    let overlap_end = e1.min(e2);

    if overlap_start <= overlap_end {
        overlap_end - overlap_start + 1
    } else {
        0
    }
}

/// Get genomic start/end coordinates for an ORF.
/// For forward: start = start, end = stop + 2 (last base of stop codon).
/// For reverse: start = stop, end = start + 2 (last base of start codon).
fn orf_coords(orf: &Orf) -> (usize, usize) {
    if orf.frame > 0 {
        (orf.start, orf.stop + 2)
    } else {
        (orf.stop, orf.start + 2)
    }
}

/// Compute the frame shift between two ORFs.
/// Returns 0 for same frame, 1 or 2 for shifted frames.
fn compute_frame_shift(frame1: i8, frame2: i8) -> i8 {
    let f1 = frame1.abs();
    let f2 = frame2.abs();
    let diff = (f1 - f2).abs();
    if diff == 0 {
        0
    } else {
        diff
    }
}

/// Compute an overlap plausibility score [0, 1].
///
/// Signals used:
/// - RBS strength of the overlapping ORF (stronger = better)
/// - Frame relationship (+2 overlaps are most common in phages)
/// - Length of the overlapping ORF relative to chance
/// - Overlap length ratio
fn compute_overlap_score(orf: &Orf, primary: &Orf, overlap_len: usize) -> f64 {
    let (_, primary_end) = orf_coords(primary);
    let primary_len = primary_end - primary.start + 1;
    let overlap_ratio = overlap_len as f64 / primary_len as f64;

    // Signal 1: RBS strength (normalized to [0, 1]).
    // RBS scores range from 0 to 27 (Prodigal-like).
    let rbs_signal = (orf.rbs_score as f64 / 27.0).min(1.0);

    // Signal 2: Frame relationship bonus.
    // +2 overlaps (1-nt shift) are most common in dsDNA phages.
    // +1 overlaps (2-nt shift) are less common.
    // Same-frame overlaps are usually annotation artifacts.
    // Antisense overlaps are rare but real.
    let frame_shift = compute_frame_shift(orf.frame, primary.frame);
    let same_strand = orf.frame.signum() == primary.frame.signum();
    let frame_bonus = if same_strand {
        match frame_shift {
            2 => 1.0,   // +2 frameshift — most common
            1 => 0.7,   // +1 frameshift — less common
            0 => 0.1,   // same frame — likely artifact
            _ => 0.5,
        }
    } else {
        0.4 // antisense — rare but real
    };

    // Signal 3: ORF length vs. chance.
    // Longer ORFs are less likely to occur by chance.
    // Use a simple sigmoid: score approaches 1.0 as length increases.
    let len_signal = 1.0 - (-(orf.seq.len() as f64) / 300.0).exp();

    // Signal 4: Overlap ratio penalty.
    // Very long overlaps (covering most of the primary gene) are suspicious.
    let overlap_penalty = if overlap_ratio > 0.8 {
        0.3
    } else if overlap_ratio > 0.5 {
        0.6
    } else {
        1.0
    };

    // Combine signals with equal weighting.
    let combined = (rbs_signal + frame_bonus + len_signal) / 3.0 * overlap_penalty;

    combined.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_orf(start: usize, stop: usize, frame: i8, seq: Vec<u8>) -> Orf {
        Orf {
            start,
            stop,
            frame,
            seq,
            rbs_score: 10,
            pstop: 0.05,
            weight_rbs: 1.0,
            hold: 100.0,
            weight: -1.0,
        }
    }

    #[test]
    fn test_orf_coords_forward() {
        let orf = make_orf(10, 30, 1, b"atg".to_vec());
        assert_eq!(orf_coords(&orf), (10, 32));
    }

    #[test]
    fn test_orf_coords_reverse() {
        let orf = make_orf(30, 10, -1, b"atg".to_vec());
        assert_eq!(orf_coords(&orf), (10, 32));
    }

    #[test]
    fn test_compute_overlap_length() {
        let orf1 = make_orf(10, 30, 1, b"atg".to_vec()); // coords: 10-32
        let orf2 = make_orf(25, 50, 2, b"atg".to_vec()); // coords: 25-52
        assert_eq!(compute_overlap_length(&orf1, &orf2), 8); // 25-32 = 8

        let orf3 = make_orf(40, 60, 1, b"atg".to_vec()); // coords: 40-62
        assert_eq!(compute_overlap_length(&orf1, &orf3), 0); // no overlap
    }

    #[test]
    fn test_compute_frame_shift() {
        assert_eq!(compute_frame_shift(1, 2), 1);
        assert_eq!(compute_frame_shift(1, 3), 2);
        assert_eq!(compute_frame_shift(2, 3), 1);
        assert_eq!(compute_frame_shift(1, 1), 0);
        assert_eq!(compute_frame_shift(-1, -2), 1);
        assert_eq!(compute_frame_shift(-1, 2), 1); // different strand
    }

    #[test]
    fn test_overlap_score_range() {
        let orf = make_orf(10, 100, 2, vec![b'a'; 300]);
        let primary = make_orf(10, 100, 1, vec![b'a'; 300]);
        let score = compute_overlap_score(&orf, &primary, 50);
        assert!(score >= 0.0 && score <= 1.0, "score {} out of range", score);
    }

    #[test]
    fn test_find_overlapping_genes_empty() {
        let path_edges: Vec<(Node, Node, f64)> = vec![];
        let orfs: Vec<Orf> = vec![];
        let overlaps = find_overlapping_genes(&path_edges, &orfs, 0.5);
        assert!(overlaps.is_empty());
    }
}
