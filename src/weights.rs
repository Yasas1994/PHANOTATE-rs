//! Weight calculations for ORF edges, gap edges, and overlap edges.

/// Score an overlap edge.
/// `length` can be negative (the formula handles it as in the reference).
/// `direction` is "same" or "diff" (strand switch).
/// `pstop` is the average P(stop) of the two flanking ORFs.
/// `overlap_scale` is a multiplicative calibration factor used in the learned
/// `--model` path to align overlap penalties with the ORF score magnitude.
pub fn score_overlap(length: isize, direction: &str, pstop: f64, overlap_scale: f64) -> f64 {
    let o = 1.0 - pstop;
    let s = 0.05;
    let mut score = o.powi(length as i32);
    score = 1.0 / score;
    if direction == "diff" {
        score += 1.0 / s;
    }
    score * overlap_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_overlap_basic() {
        let score = score_overlap(10, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_overlap_same_vs_diff() {
        let same = score_overlap(10, "same", 0.05, 1.0);
        let diff = score_overlap(10, "diff", 0.05, 1.0);
        // diff direction adds 1/s penalty
        assert!(
            diff > same,
            "diff direction should add penalty: diff={} same={}",
            diff,
            same
        );
    }

    #[test]
    fn test_score_overlap_negative_length() {
        // Negative length (overlap) should still compute
        let score = score_overlap(-5, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_overlap_zero_length() {
        let score = score_overlap(0, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_overlap_high_pstop() {
        // High pstop means low o = 1 - pstop, so score should be higher
        let low_pstop = score_overlap(10, "same", 0.01, 1.0);
        let high_pstop = score_overlap(10, "same", 0.5, 1.0);
        assert!(
            high_pstop > low_pstop,
            "higher pstop should give higher score"
        );
    }

    #[test]
    fn test_score_gap_basic() {
        let score = score_gap(30, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_gap_same_vs_diff() {
        let same = score_gap(30, "same", 0.05, 1.0);
        let diff = score_gap(30, "diff", 0.05, 1.0);
        assert!(diff > same, "diff direction should add penalty");
    }

    #[test]
    fn test_score_gap_long_gap() {
        // Gap > 300 uses different formula
        let short = score_gap(100, "same", 0.05, 1.0);
        let long = score_gap(400, "same", 0.05, 1.0);
        assert!(long > short, "long gap should have higher score");
    }

    #[test]
    fn test_score_gap_negative_length() {
        let score = score_gap(-10, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_gap_zero_length() {
        let score = score_gap(0, "same", 0.05, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_gap_high_pgap() {
        let low_pgap = score_gap(30, "same", 0.01, 1.0);
        let high_pgap = score_gap(30, "same", 0.5, 1.0);
        assert!(high_pgap > low_pgap, "higher pgap should give higher score");
    }

    #[test]
    fn test_score_gap_vs_overlap_same_params() {
        // With same pstop/pgap and direction, gap and overlap should differ
        let gap = score_gap(30, "same", 0.05, 1.0);
        let overlap = score_overlap(10, "same", 0.05, 1.0);
        // Both should be positive but different formulas
        assert!(gap > 0.0);
        assert!(overlap > 0.0);
    }
}

/// Score a gap edge.
/// `length` is the gap length in nucleotides (can be negative for overlaps).
/// `direction` is "same" or "diff".
/// `pgap` is the genome-wide average P(not_stop).
/// `gap_scale` is a multiplicative calibration factor; see `score_overlap`.
pub fn score_gap(length: isize, direction: &str, pgap: f64, gap_scale: f64) -> f64 {
    let g = 1.0 - pgap;
    let s = 0.05;

    if length > 300 {
        return (g.powi(100) + length as f64) * gap_scale;
    }
    let mut score = g.powf(length as f64 / 3.0);
    score = 1.0 / score;
    if direction == "diff" {
        score += 1.0 / s;
    }
    score * gap_scale
}
