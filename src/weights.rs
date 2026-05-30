/// Weight calculations for ORF edges, gap edges, and overlap edges.

/// Score an overlap edge.
/// `length` can be negative (the formula handles it as in the reference).
/// `direction` is "same" or "diff" (strand switch).
/// `pstop` is the average P(stop) of the two flanking ORFs.
pub fn score_overlap(length: isize, direction: &str, pstop: f64) -> f64 {
    let o = 1.0 - pstop;
    let s = 0.05;
    let mut score = o.powi(length as i32);
    score = 1.0 / score;
    if direction == "diff" {
        score += 1.0 / s;
    }
    score
}

/// Score a gap edge.
/// `length` is the gap length in nucleotides (can be negative for overlaps).
/// `direction` is "same" or "diff".
/// `pgap` is the genome-wide average P(not_stop).
pub fn score_gap(length: isize, direction: &str, pgap: f64) -> f64 {
    let g = 1.0 - pgap;
    let s = 0.05;

    if length > 300 {
        return g.powi(100) + length as f64;
    }
    let mut score = g.powf(length as f64 / 3.0);
    score = 1.0 / score;
    if direction == "diff" {
        score += 1.0 / s;
    }
    score
}
