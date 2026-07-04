//! Prodigal-style hexamer (6-mer) coding-potential scorer.

use crate::kmer::kmer_encode;
use crate::orf::Orf;

pub const NUM_HEXAMERS: usize = 4096;

/// Prodigal-style hexamer (6-mer) coding-potential model.
#[derive(Debug, Clone)]
pub struct HexamerModel {
    pub log_odds: [f64; NUM_HEXAMERS],
}

impl HexamerModel {
    /// Train a hexamer model from a set of seed ORFs.
    ///
    /// Seed genes are ORFs that are at least 80% of the mean ORF length and
    /// that have some start-codon signal (either an RBS or a non-SD motif).
    /// Gene hexamer counts are collected in the translational reading frame
    /// from each selected ORF; background counts are collected on both strands
    /// in all three forward frames.  Counts are smoothed with a pseudocount of
    /// one and converted to log-odds ratios.
    pub fn train(orfs: &[Orf], dna: &[u8], rc_dna: &[u8]) -> Self {
        let mut model = Self {
            log_odds: [0.0; NUM_HEXAMERS],
        };

        // Seed-gene selection: ORFs at least 80% of mean length with some start signal.
        let mean_len = if orfs.is_empty() {
            0.0
        } else {
            orfs.iter().map(|o| o.seq.len() as f64).sum::<f64>() / orfs.len() as f64
        };
        let threshold = mean_len * 0.8;

        let mut gene_counts = [0.0f64; NUM_HEXAMERS];
        let mut bg_counts = [0.0f64; NUM_HEXAMERS];
        let mut total_gene = 0.0;
        let mut total_bg = 0.0;

        // Background counts on both strands.
        for seq in [dna, rc_dna] {
            for i in 0..seq.len().saturating_sub(5) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    bg_counts[ndx] += 1.0;
                    total_bg += 1.0;
                }
            }
        }

        // Gene counts on selected ORFs.
        for orf in orfs {
            if (orf.seq.len() as f64) < threshold {
                continue;
            }
            if orf.sd_rbs_score <= 1.0 && orf.non_sd_rbs_score <= 1.0 {
                continue;
            }
            let seq = orf.sequence();
            for i in (0..seq.len().saturating_sub(5)).step_by(3) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    gene_counts[ndx] += 1.0;
                    total_gene += 1.0;
                }
            }
        }

        // Smooth and convert to log-odds.
        for ndx in 0..NUM_HEXAMERS {
            let g = (gene_counts[ndx] + 1.0) / (total_gene + NUM_HEXAMERS as f64);
            let b = (bg_counts[ndx] + 1.0) / (total_bg + NUM_HEXAMERS as f64);
            model.log_odds[ndx] = (g / b).ln().clamp(-4.0, 4.0);
        }

        model
    }

    /// Train an unsupervised hexamer model from all ORFs that are long enough
    /// to be plausible genes, without requiring RBS signal.
    ///
    /// This is useful for feature export on FASTA inputs where no trusted gene
    /// annotations are available.
    pub fn train_unsupervised(orfs: &[Orf], dna: &[u8], rc_dna: &[u8]) -> Self {
        let mut model = Self {
            log_odds: [0.0; NUM_HEXAMERS],
        };

        let mean_len = if orfs.is_empty() {
            0.0
        } else {
            orfs.iter().map(|o| o.seq.len() as f64).sum::<f64>() / orfs.len() as f64
        };
        let threshold = mean_len * 0.8;

        let mut gene_counts = [0.0f64; NUM_HEXAMERS];
        let mut bg_counts = [0.0f64; NUM_HEXAMERS];
        let mut total_gene = 0.0;
        let mut total_bg = 0.0;

        // Background counts on both strands.
        for seq in [dna, rc_dna] {
            for i in 0..seq.len().saturating_sub(5) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    bg_counts[ndx] += 1.0;
                    total_bg += 1.0;
                }
            }
        }

        // Gene counts from all long ORFs.
        for orf in orfs {
            if (orf.seq.len() as f64) < threshold {
                continue;
            }
            let seq = orf.sequence();
            for i in (0..seq.len().saturating_sub(5)).step_by(3) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    gene_counts[ndx] += 1.0;
                    total_gene += 1.0;
                }
            }
        }

        // Smooth and convert to log-odds.
        for ndx in 0..NUM_HEXAMERS {
            let g = (gene_counts[ndx] + 1.0) / (total_gene + NUM_HEXAMERS as f64);
            let b = (bg_counts[ndx] + 1.0) / (total_bg + NUM_HEXAMERS as f64);
            model.log_odds[ndx] = (g / b).ln().clamp(-4.0, 4.0);
        }

        model
    }

    /// Train a hexamer model from a supplied set of trusted ORFs (e.g. annotated
    /// CDS). Gene hexamer counts are taken from the ORF sequences in frame;
    /// background counts are collected from both genome strands in all frames.
    pub fn from_annotated_orfs(orfs: &[&Orf], dna: &[u8], rc_dna: &[u8]) -> Self {
        let mut model = Self {
            log_odds: [0.0; NUM_HEXAMERS],
        };

        let mut gene_counts = [0.0f64; NUM_HEXAMERS];
        let mut bg_counts = [0.0f64; NUM_HEXAMERS];
        let mut total_gene = 0.0;
        let mut total_bg = 0.0;

        // Background counts on both strands.
        for seq in [dna, rc_dna] {
            for i in 0..seq.len().saturating_sub(5) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    bg_counts[ndx] += 1.0;
                    total_bg += 1.0;
                }
            }
        }

        // Gene counts from annotated ORFs.
        for orf in orfs {
            let seq = orf.sequence();
            for i in (0..seq.len().saturating_sub(5)).step_by(3) {
                if let Some(ndx) = kmer_encode(seq, i, 6) {
                    gene_counts[ndx] += 1.0;
                    total_gene += 1.0;
                }
            }
        }

        // Smooth and convert to log-odds.
        for ndx in 0..NUM_HEXAMERS {
            let g = (gene_counts[ndx] + 1.0) / (total_gene + NUM_HEXAMERS as f64);
            let b = (bg_counts[ndx] + 1.0) / (total_bg + NUM_HEXAMERS as f64);
            model.log_odds[ndx] = (g / b).ln().clamp(-4.0, 4.0);
        }

        model
    }

    /// Compute the raw in-frame hexamer log-odds sum (Prodigal-style raw cscore).
    pub fn cscore(&self, orf: &Orf) -> f64 {
        let seq = orf.sequence();
        let mut sum = 0.0;
        let mut count = 0;
        for i in (0..seq.len().saturating_sub(5)).step_by(3) {
            if let Some(ndx) = kmer_encode(seq, i, 6) {
                sum += self.log_odds[ndx];
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        sum
    }

    /// Score an ORF by converting its average cscore to a positive multiplier.
    pub fn score_orf(&self, orf: &Orf) -> f64 {
        let raw = self.cscore(orf);
        let seq = orf.sequence();
        let mut count = 0;
        for i in (0..seq.len().saturating_sub(5)).step_by(3) {
            if kmer_encode(seq, i, 6).is_some() {
                count += 1;
            }
        }
        if count == 0 {
            return 1.0;
        }
        let avg = raw / count as f64;
        avg.exp().clamp(0.1, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmer_roundtrip_6() {
        for ndx in 0..NUM_HEXAMERS {
            let mut seq = Vec::with_capacity(6);
            for i in 0..6 {
                seq.push(b"ACGT"[(ndx >> (2 * i)) & 0x3]);
            }
            assert_eq!(kmer_encode(&seq, 0, 6), Some(ndx));
        }
    }

    #[test]
    fn kmer_encode_rejects_ambiguous() {
        assert!(kmer_encode(b"atgnat", 0, 6).is_none());
    }

    #[test]
    fn training_produces_log_odds() {
        let unit = b"atgaaaaaaaatgaaaaaaatgaaaaaaa";
        let seq = unit
            .iter()
            .cycle()
            .take(unit.len() * 20)
            .copied()
            .collect::<Vec<u8>>();
        let rc = crate::genome::rev_comp(&seq);
        let orfs = vec![Orf {
            start: 1,
            stop: seq.len() - 2,
            frame: 1,
            seq: seq.clone(),
            rbs_score: 0,
            rbs_motif: None,
            pstop: 0.01,
            sd_rbs_score: 2.0,
            hold: 100.0,
            non_sd_rbs_score: 1.0,
            coding_potential: 1.0,
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
        }];
        let model = HexamerModel::train(&orfs, &seq, &rc);
        assert!(model.log_odds.iter().all(|&s| s.is_finite()));
        assert!(model.log_odds.iter().any(|&s| s > 0.0));
    }

    #[test]
    fn score_orf_returns_positive_multiplier() {
        let unit = b"atgaaaaaaaatgaaaaaaatgaaaaaaa";
        let seq = unit
            .iter()
            .cycle()
            .take(unit.len() * 20)
            .copied()
            .collect::<Vec<u8>>();
        let rc = crate::genome::rev_comp(&seq);
        let orfs = vec![Orf {
            start: 1,
            stop: seq.len() - 2,
            frame: 1,
            seq: seq.clone(),
            rbs_score: 0,
            rbs_motif: None,
            pstop: 0.01,
            sd_rbs_score: 2.0,
            hold: 100.0,
            non_sd_rbs_score: 1.0,
            coding_potential: 1.0,
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
        }];
        let model = HexamerModel::train(&orfs, &seq, &rc);
        let s = model.score_orf(&orfs[0]);
        assert!(s > 0.0);
        assert!(s <= 10.0);
    }

    #[test]
    fn from_annotated_orfs_produces_log_odds() {
        let unit = b"atgaaaaaaaatgaaaaaaatgaaaaaaa";
        let seq = unit
            .iter()
            .cycle()
            .take(unit.len() * 20)
            .copied()
            .collect::<Vec<u8>>();
        let rc = crate::genome::rev_comp(&seq);
        let orf = Orf {
            start: 1,
            stop: seq.len() - 2,
            frame: 1,
            seq: seq.clone(),
            rbs_score: 0,
            rbs_motif: None,
            pstop: 0.01,
            sd_rbs_score: 2.0,
            hold: 100.0,
            non_sd_rbs_score: 1.0,
            coding_potential: 1.0,
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
        let model = HexamerModel::from_annotated_orfs(&[&orf], &seq, &rc);
        assert!(model.log_odds.iter().all(|&s| s.is_finite()));
        let s = model.score_orf(&orf);
        assert!(s > 0.0);
        assert!(s <= 10.0);
    }

    #[test]
    fn cscore_returns_raw_log_odds_sum() {
        let unit = b"atgaaaaaaaatgaaaaaaatgaaaaaaa";
        let seq = unit
            .iter()
            .cycle()
            .take(unit.len() * 20)
            .copied()
            .collect::<Vec<u8>>();
        let rc = crate::genome::rev_comp(&seq);
        let orf = Orf {
            start: 1,
            stop: seq.len() - 2,
            frame: 1,
            seq: seq.clone(),
            rbs_score: 0,
            rbs_motif: None,
            pstop: 0.01,
            sd_rbs_score: 2.0,
            hold: 100.0,
            non_sd_rbs_score: 1.0,
            coding_potential: 1.0,
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
        let model = HexamerModel::from_annotated_orfs(&[&orf], &seq, &rc);
        let s = model.cscore(&orf);
        assert!(s.is_finite());
        // score_orf is the exponentiated, clamped average.
        let m = model.score_orf(&orf);
        let count = if orf.seq.len() >= 6 {
            (orf.seq.len() - 3) / 3
        } else {
            0
        };
        assert!(count > 0);
        let avg = s / count as f64;
        assert!((m - avg.exp()).abs() < 1e-9);
        assert!(m > 0.0);
        assert!(m <= 10.0);
    }
}
