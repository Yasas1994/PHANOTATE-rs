//! Post-hoc rescue of overlapping genes excluded by the primary shortest path.

use crate::graph::Node;
use crate::onnx_scorer::OnnxScorer;
use crate::orf::Orf;
use std::collections::HashSet;

/// Compute genomic interval for an ORF as (start, stop) in forward coordinates.
fn orf_interval(orf: &Orf) -> (usize, usize) {
    if orf.frame > 0 {
        (orf.start, orf.stop.saturating_add(2))
    } else {
        (orf.stop, orf.start.saturating_add(2))
    }
}

/// True if two closed intervals overlap by at least one base.
fn intervals_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Build a synthetic path edge tuple for an ORF so it can be fed to the output writers.
pub fn orf_to_path_edge(orf: &Orf, weight: f64) -> (Node, Node, f64) {
    if orf.frame > 0 {
        (
            Node::new("CDS", "start", orf.frame, orf.start),
            Node::new("CDS", "stop", orf.frame, orf.stop),
            weight,
        )
    } else {
        (
            Node::new("CDS", "stop", orf.frame, orf.stop),
            Node::new("CDS", "start", orf.frame, orf.start),
            weight,
        )
    }
}

/// Lightweight summary of a selected (primary-path) ORF for overlap scoring.
#[derive(Debug, Clone, Copy)]
pub struct RescueOrf {
    pub start: usize,
    pub stop: usize,
    pub strand: char,
    pub probability: f64,
    pub weight: f64,
}

impl RescueOrf {
    fn interval(&self) -> (usize, usize) {
        (self.start, self.stop)
    }

    fn len(&self) -> usize {
        self.stop.saturating_sub(self.start) + 1
    }
}

/// Compute a rescue score for an ORF that overlaps one or more primary genes.
///
/// Score = model_probability - penalty_weight * max_overlap_ratio
/// where max_overlap_ratio is the largest overlap length divided by the
/// shorter of the two overlapping ORFs.
pub fn compute_rescue_score(
    candidate: &RescueOrf,
    selected: &[&RescueOrf],
    penalty_weight: f64,
) -> f64 {
    let candidate_len = candidate.len();
    if candidate_len == 0 || candidate.start >= candidate.stop {
        return 0.0;
    }

    let max_overlap_ratio = selected
        .iter()
        .filter(|s| intervals_overlap(candidate.interval(), s.interval()))
        .map(|s| {
            let overlap_len = candidate
                .stop
                .min(s.stop)
                .saturating_sub(candidate.start.max(s.start))
                + 1;
            let min_len = candidate_len.min(s.len());
            if min_len == 0 {
                0.0
            } else {
                overlap_len as f64 / min_len as f64
            }
        })
        .fold(0.0, f64::max);

    candidate.probability - penalty_weight * max_overlap_ratio
}

/// Unique identifier for a rescued ORF and its rescue score.
pub type RescueResult = (usize, f64);

/// Find high-confidence ORFs that overlap the primary path but were not selected.
///
/// The second pass solves a weighted interval-scheduling problem: among
/// candidate ORFs that overlap primary genes, pick a non-overlapping subset
/// that maximises `sum(rescue_score - lambda)`.
///
/// * `primary_edges` — edges returned by `bellman_ford::shortest_path`.
/// * `orfs` — full list of ORFs found by `find_orfs_with_rc`.
/// * `scorer` — ONNX model scorer.
/// * `threshold` — minimum rescue score to include an ORF.
/// * `penalty_weight` — weight applied to the overlap-ratio penalty.
/// * `min_orf_len` — minimum ORF length (in bp) to be considered.
/// * `lambda` — per-rescued-gene DP penalty. A higher value suppresses more
///   rescues; only rescues with positive adjusted score (`score - lambda`) are kept.
pub fn find_overlapping_genes(
    primary_edges: &[(Node, Node, f64)],
    orfs: &[Orf],
    scorer: &OnnxScorer,
    threshold: f64,
    penalty_weight: f64,
    min_orf_len: usize,
    lambda: f64,
) -> Vec<RescueResult> {
    let selected_indices: HashSet<usize> = primary_edges
        .iter()
        .filter_map(|(left, right, _weight)| orf_index_from_edge(left, right, orfs))
        .collect();

    if selected_indices.is_empty() {
        return Vec::new();
    }

    let selected: Vec<RescueOrf> = selected_indices
        .iter()
        .map(|&idx| {
            let orf = &orfs[idx];
            let (start, stop) = orf_interval(orf);
            RescueOrf {
                start,
                stop,
                strand: if orf.frame > 0 { '+' } else { '-' },
                probability: scorer.probability_for_orf(orf),
                weight: orf.weight,
            }
        })
        .collect();

    let selected_refs: Vec<&RescueOrf> = selected.iter().collect();

    let rescues: Vec<(&Orf, f64)> = orfs
        .iter()
        .enumerate()
        .filter(|(idx, o)| {
            let (start, stop) = orf_interval(o);
            let len = stop.saturating_sub(start) + 1;
            len >= min_orf_len
                && selected_refs
                    .iter()
                    .any(|s| intervals_overlap((start, stop), s.interval()))
                && !selected_indices.contains(idx)
        })
        .map(|(_idx, o)| {
            let (start, stop) = orf_interval(o);
            let candidate = RescueOrf {
                start,
                stop,
                strand: if o.frame > 0 { '+' } else { '-' },
                probability: scorer.probability_for_orf(o),
                weight: o.weight,
            };
            let score = compute_rescue_score(&candidate, &selected_refs, penalty_weight);
            (o, score)
        })
        .filter(|(_, score)| *score >= threshold)
        .collect();

    // Second-pass DP: select a non-overlapping subset of candidates that
    // maximises sum(score - lambda).
    let mut candidates: Vec<(usize, f64, usize, usize)> = rescues
        .into_iter()
        .filter_map(|(orf, score)| {
            orfs.iter().position(|x| std::ptr::eq(x, orf)).map(|idx| {
                let (start, stop) = orf_interval(orf);
                (idx, score, start, stop)
            })
        })
        .collect();

    // Sort by stop coordinate for interval scheduling.
    candidates.sort_by_key(|(_, _, _, stop)| *stop);

    // dp[i] = best adjusted score using candidates[0..=i]
    // choice[i] = true if candidate i is taken in dp[i]
    // prev[i] = index of last compatible candidate before i
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut prev: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let (_, _, start_i, _) = candidates[i];
        prev[i] = candidates[..i]
            .iter()
            .rposition(|(_, _, _, stop_j)| *stop_j < start_i);
    }

    let mut dp: Vec<f64> = vec![0.0; n];
    let mut take: Vec<bool> = vec![false; n];
    for i in 0..n {
        let gain = candidates[i].1 - lambda;
        let include = gain + prev[i].map_or(0.0, |p| dp[p]);
        let exclude = if i == 0 { 0.0 } else { dp[i - 1] };
        if include > exclude && gain > 0.0 {
            dp[i] = include;
            take[i] = true;
        } else {
            dp[i] = exclude;
            take[i] = false;
        }
    }

    // Reconstruct selected candidates.
    let mut kept: Vec<RescueResult> = Vec::new();
    let mut i = n;
    while i > 0 {
        i -= 1;
        if take[i] {
            kept.push((candidates[i].0, candidates[i].1));
            if let Some(p) = prev[i] {
                i = p + 1; // skip to before the compatible predecessor
            } else {
                break;
            }
        }
    }
    kept.reverse();
    kept
}

/// Recover the index of the `Orf` corresponding to a primary-path edge, if any.
fn orf_index_from_edge(left: &Node, right: &Node, orfs: &[Orf]) -> Option<usize> {
    if left.gene != "CDS" || right.gene != "CDS" {
        return None;
    }
    if left.node_type == "start" && right.node_type == "stop" && left.frame > 0 {
        orfs.iter().position(|o| {
            o.start == left.position && o.stop == right.position && o.frame == left.frame
        })
    } else if left.node_type == "stop" && right.node_type == "start" && left.frame < 0 {
        orfs.iter().position(|o| {
            o.stop == left.position && o.start == right.position && o.frame == left.frame
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning;
    use std::path::Path;

    fn test_orf(start: usize, stop: usize, frame: i8) -> Orf {
        Orf {
            start,
            stop,
            frame,
            seq: vec![b'a'; 100],
            rbs_score: 0,
            pstop: 0.0,
            sd_rbs_score: 1.0,
            non_sd_rbs_score: 1.0,
            hold: 1.0,
            coding_potential: 1.0,
            weight: -1.0,
            rbs_spacer: 20.0,
            ..Orf::default()
        }
    }

    fn test_scorer() -> OnnxScorer {
        OnnxScorer::from_file(Path::new("tests/golden/orf_model.onnx")).unwrap()
    }

    #[test]
    fn test_interval_overlap() {
        let a = (100, 300);
        let b = (250, 400);
        assert!(intervals_overlap(a, b));
        assert!(intervals_overlap(b, a));
        let c = (350, 400);
        assert!(!intervals_overlap(a, c));
    }

    #[test]
    fn test_rescue_score_basic() {
        let selected = RescueOrf {
            start: 100,
            stop: 300,
            strand: '+',
            probability: 0.9,
            weight: -1.0,
        };
        let candidate = RescueOrf {
            start: 250,
            stop: 400,
            strand: '+',
            probability: 0.85,
            weight: -1.0,
        };
        let score = compute_rescue_score(&candidate, &[&selected], 0.3);
        // overlap_len = 51, min_len = 201, penalty = 51/201*0.3 ≈ 0.076
        assert!(score > 0.7 && score < 0.8);
    }

    #[test]
    fn test_redundant_rescues_removed() {
        let a = RescueOrf {
            start: 100,
            stop: 300,
            strand: '+',
            probability: 0.9,
            weight: -1.0,
        };
        let b = RescueOrf {
            start: 250,
            stop: 350,
            strand: '+',
            probability: 0.85,
            weight: -1.0,
        };
        let c = RescueOrf {
            start: 400,
            stop: 500,
            strand: '+',
            probability: 0.8,
            weight: -1.0,
        };
        let score_b = compute_rescue_score(&b, &[&a], 0.3);
        let score_c = compute_rescue_score(&c, &[&a], 0.3);
        // b overlaps a and is penalised; c does not overlap, so it retains its higher score.
        assert!(score_b < score_c);
    }

    #[test]
    fn test_empty_primary_edges_returns_empty() {
        let primary_edges: Vec<(Node, Node, f64)> = Vec::new();
        let orfs = vec![test_orf(100, 200, 1)];
        let scorer = test_scorer();
        let rescued = find_overlapping_genes(&primary_edges, &orfs, &scorer, -999.0, 0.0, 0, 0.0);
        assert!(rescued.is_empty());
    }

    #[test]
    fn test_empty_orfs_returns_empty() {
        let primary_edges = vec![(
            Node::new("CDS", "start", 1, 100),
            Node::new("CDS", "stop", 1, 200),
            -1.0,
        )];
        let orfs: Vec<Orf> = Vec::new();
        let scorer = test_scorer();
        let rescued = find_overlapping_genes(&primary_edges, &orfs, &scorer, -999.0, 0.0, 0, 0.0);
        assert!(rescued.is_empty());
    }

    #[test]
    fn test_zero_length_and_inverted_interval_score_is_zero() {
        let selected = RescueOrf {
            start: 100,
            stop: 200,
            strand: '+',
            probability: 0.9,
            weight: -1.0,
        };
        let zero_len = RescueOrf {
            start: 150,
            stop: 150,
            strand: '+',
            probability: 0.9,
            weight: -1.0,
        };
        let inverted = RescueOrf {
            start: 200,
            stop: 100,
            strand: '+',
            probability: 0.9,
            weight: -1.0,
        };
        assert_eq!(compute_rescue_score(&zero_len, &[&selected], 0.3), 0.0);
        assert_eq!(compute_rescue_score(&inverted, &[&selected], 0.3), 0.0);
    }

    #[test]
    fn test_no_overlap_returns_empty_rescue_set() {
        let primary = test_orf(100, 200, 1);
        let distant = test_orf(300, 400, 1);
        let orfs = vec![primary, distant];
        let primary_edges = vec![(
            Node::new("CDS", "start", 1, 100),
            Node::new("CDS", "stop", 1, 200),
            -1.0,
        )];
        let scorer = test_scorer();
        let rescued = find_overlapping_genes(
            &primary_edges,
            &orfs,
            &scorer,
            tuning::RESCUE_THRESHOLD,
            tuning::OVERLAP_PENALTY_WEIGHT,
            tuning::MIN_RESCUE_ORF_LEN,
            0.0,
        );
        assert!(rescued.is_empty());
    }
}
