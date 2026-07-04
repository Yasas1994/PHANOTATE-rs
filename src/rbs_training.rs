//! Shared RBS training for the annotation pipeline and `--export-features`.
//!
//! Centralises background RBS scoring, SD likelihood-ratio training, non-SD
//! motif training, and the SD/auto mode decision so that feature export and
//! gene prediction produce the same `sd_rbs_score` / `non_sd_rbs_score` values.

use crate::nonsd_motif::NonSdModel;
use crate::orf::Orf;
use crate::rbs_mode::RbsMode;
use crate::rbs_scanner::{score_rbs_for_mode, NUM_RBS_BINS};
use std::collections::HashMap;

/// Build the start-codon weight map used by the non-SD motif model.
pub fn build_start_weights(start_codons: &[Vec<u8>]) -> HashMap<Vec<u8>, f64> {
    let mut map = HashMap::new();
    for codon in start_codons {
        let w = match codon.as_slice() {
            b"atg" => 0.85,
            b"gtg" => 0.10,
            b"ttg" => 0.05,
            _ => 1.0,
        };
        map.insert(codon.clone(), w);
    }
    let max_w = map.values().cloned().fold(0.0, f64::max);
    if max_w > 0.0 {
        for v in map.values_mut() {
            *v /= max_w;
        }
    }
    map
}

/// Decide whether the genome uses Shine–Dalgarno RBS signalling by comparing
/// the high-score tail of the training distribution to the background.
pub fn detect_uses_sd(background: &[f64; NUM_RBS_BINS], training: &[f64; NUM_RBS_BINS]) -> bool {
    let top_bins = [27, 26, 25, 24, 22, 20];
    let signal: f64 = top_bins.iter().map(|&i| training[i] / background[i]).sum();
    signal >= 2.0
}

/// Compute the background RBS distribution over all 21-nt windows on both
/// strands, using the scoring scheme implied by `rbs_mode`.
pub fn compute_background_rbs(dna: &[u8], rc_dna: &[u8], rbs_mode: RbsMode) -> [f64; NUM_RBS_BINS] {
    let use_prodigal = rbs_mode == RbsMode::Prodigal;
    let mut background_rbs = [1.0f64; NUM_RBS_BINS];
    let len = dna.len();
    for i in 0..len {
        let window = if i + 21 <= len {
            &dna[i..i + 21]
        } else {
            &dna[i..]
        };
        background_rbs[score_rbs_for_mode(window, use_prodigal)] += 1.0;

        let rc_start = len.saturating_sub(i + 21);
        let rc_window = &rc_dna[rc_start..len - i];
        background_rbs[score_rbs_for_mode(rc_window, use_prodigal)] += 1.0;
    }
    let bg_sum: f64 = background_rbs.iter().sum();
    for v in &mut background_rbs {
        *v /= bg_sum;
    }
    background_rbs
}

/// Train per-ORF SD and non-SD RBS scores.
///
/// Returns `true` if the caller should treat the genome as non-SD mode for
/// output formatting.
pub fn train_rbs_scores(
    orfs: &mut [Orf],
    dna: &[u8],
    rc_dna: &[u8],
    rbs_mode: RbsMode,
    start_codons_map: &HashMap<Vec<u8>, f64>,
) -> bool {
    if orfs.is_empty() {
        return rbs_mode == RbsMode::NonSd;
    }

    let background_rbs = compute_background_rbs(dna, rc_dna, rbs_mode);

    let mut training_rbs = [1.0f64; NUM_RBS_BINS];
    for orf in orfs.iter() {
        training_rbs[orf.rbs_score] += 1.0;
    }
    let tr_sum: f64 = training_rbs.iter().sum();
    for v in &mut training_rbs {
        *v /= tr_sum;
    }
    for orf in orfs.iter_mut() {
        orf.sd_rbs_score = training_rbs[orf.rbs_score] / background_rbs[orf.rbs_score];
    }

    let use_non_sd = match rbs_mode {
        RbsMode::NonSd => true,
        RbsMode::Sd | RbsMode::Prodigal => false,
        RbsMode::Auto => !detect_uses_sd(&background_rbs, &training_rbs),
    };

    // Train the non-SD model once; it is used for scoring in non-SD mode
    // and for fallback motif labels in SD mode.
    let non_sd_model = NonSdModel::train(orfs, dna, rc_dna, start_codons_map);

    if use_non_sd {
        for orf in orfs.iter_mut() {
            orf.sd_rbs_score = 1.0;
        }
        for orf in orfs.iter_mut() {
            orf.non_sd_rbs_score = non_sd_model.score_orf(orf, dna, rc_dna);
            let (wseq, start) = crate::nonsd_motif::upstream_context(dna, rc_dna, orf);
            let hit = if start >= 18 + crate::nonsd_motif::MIN_MOTIF_LEN {
                crate::nonsd_motif::find_best_motif(
                    &non_sd_model.mot_wt,
                    wseq,
                    start,
                    non_sd_model.no_mot,
                )
            } else {
                crate::nonsd_motif::MotifHit::default()
            };
            orf.rbs_motif = crate::nonsd_motif::format_motif_hit(&hit);
        }
    } else {
        for orf in orfs.iter_mut() {
            if orf.rbs_motif.is_none() {
                orf.rbs_motif = non_sd_model
                    .best_motif_label(orf, dna, rc_dna)
                    .map(|m| format!("nonSD:{m}"));
            }
        }
    }

    use_non_sd
}
