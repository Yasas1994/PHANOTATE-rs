//! Per-genome calibration of gap/overlap edge penalties for the ONNX model path.

use crate::orf::Orf;
use crate::weights::{score_gap, score_overlap};

const MODEL_GAP_TARGET_RATIO: f64 = 1.0;
const MODEL_OVERLAP_TARGET_RATIO: f64 = 1.0;
const MIN_ORFS_FOR_CALIBRATION: usize = 3;
const SCALE_MIN: f64 = 0.1;
const SCALE_MAX: f64 = 10.0;

/// Compute per-genome gap and overlap scales for the `--model` path.
///
/// The scales are chosen so that representative raw gap/overlap penalties are
/// proportional to the median absolute ORF reward on this genome.
///
/// On the default (non-model) path, callers should simply use `(1.0, 1.0)`.
pub fn compute_model_penalty_scales(orfs: &[Orf], pgap: f64) -> (f64, f64) {
    if orfs.len() < MIN_ORFS_FOR_CALIBRATION {
        return (1.0, 1.0);
    }

    let mut orf_rewards: Vec<f64> = orfs.iter().map(|o| o.weight.abs()).collect();
    orf_rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_orf = orf_rewards[orf_rewards.len() / 2];
    if !med_orf.is_finite() || med_orf <= 0.0 {
        return (1.0, 1.0);
    }

    let gap_target: f64 = std::env::var("PHANOTATE_MODEL_GAP_TARGET_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MODEL_GAP_TARGET_RATIO);
    let overlap_target: f64 = std::env::var("PHANOTATE_MODEL_OVERLAP_TARGET_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MODEL_OVERLAP_TARGET_RATIO);

    let mut gap_raws = Vec::new();
    for &len in &[0_isize, 10, 30, 100, 300] {
        gap_raws.push(score_gap(len, "same", pgap, 1.0));
        gap_raws.push(score_gap(len, "diff", pgap, 1.0));
    }
    gap_raws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_gap = gap_raws[gap_raws.len() / 2];

    let pstop_avg = if orfs.is_empty() {
        pgap
    } else {
        orfs.iter().map(|o| o.pstop).sum::<f64>() / orfs.len() as f64
    };
    let mut overlap_raws = Vec::new();
    for &len in &[1_isize, 5, 10, 20, 50] {
        overlap_raws.push(score_overlap(len, "same", pstop_avg, 1.0));
        overlap_raws.push(score_overlap(len, "diff", pstop_avg, 1.0));
    }
    overlap_raws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_overlap = overlap_raws[overlap_raws.len() / 2];

    let gap_scale = if med_gap.is_finite() && med_gap > 0.0 {
        (gap_target * med_orf / med_gap).clamp(SCALE_MIN, SCALE_MAX)
    } else {
        1.0
    };
    let overlap_scale = if med_overlap.is_finite() && med_overlap > 0.0 {
        (overlap_target * med_orf / med_overlap).clamp(SCALE_MIN, SCALE_MAX)
    } else {
        1.0
    };

    (gap_scale, overlap_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_orfs_falls_back_to_one() {
        assert_eq!(compute_model_penalty_scales(&[], 0.05), (1.0, 1.0));
    }
}
