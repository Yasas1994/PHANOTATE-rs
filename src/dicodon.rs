//! Prodigal-style dicodon (6-mer) coding-potential scorer.

use crate::orf::Orf;

pub const NUM_DICODONS: usize = 4096;

/// 2-bit encode a single base: A=0, C=1, G=2, T=3.
fn encode_base(b: u8) -> Option<usize> {
    match b {
        b'a' | b'A' => Some(0),
        b'c' | b'C' => Some(1),
        b'g' | b'G' => Some(2),
        b't' | b'T' => Some(3),
        _ => None,
    }
}

/// Encode a DNA word of length `len` starting at `pos`.
/// Returns None if any base is ambiguous or out of range.
pub fn kmer_encode(seq: &[u8], pos: usize, len: usize) -> Option<usize> {
    if pos + len > seq.len() {
        return None;
    }
    let mut ndx = 0;
    for i in 0..len {
        ndx |= encode_base(seq[pos + i])? << (2 * i);
    }
    Some(ndx)
}

/// Prodigal-style dicodon (6-mer) coding-potential model.
#[derive(Debug, Clone)]
pub struct DicodonModel {
    pub scores: [f64; NUM_DICODONS],
}

impl DicodonModel {
    /// Train a dicodon model from a set of seed ORFs.
    ///
    /// Seed genes are ORFs that are at least 80% of the mean ORF length and
    /// that have some start-codon signal (either an RBS or a non-SD motif).
    /// Gene dicodon counts are collected in the translational reading frame
    /// from each selected ORF; background counts are collected on both strands
    /// in all three forward frames.  Counts are smoothed with a pseudocount of
    /// one and converted to log-likelihood ratios.
    pub fn train(orfs: &[Orf], dna: &[u8], rc_dna: &[u8]) -> Self {
        let mut model = Self {
            scores: [0.0; NUM_DICODONS],
        };

        // Seed-gene selection: ORFs at least 80% of mean length with some start signal.
        let mean_len = if orfs.is_empty() {
            0.0
        } else {
            orfs.iter().map(|o| o.seq.len() as f64).sum::<f64>() / orfs.len() as f64
        };
        let threshold = mean_len * 0.8;

        let mut gene_counts = [0.0f64; NUM_DICODONS];
        let mut bg_counts = [0.0f64; NUM_DICODONS];
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
            if orf.weight_rbs <= 1.0 && orf.motif_score <= 1.0 {
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

        // Smooth and convert to log-likelihoods.
        for ndx in 0..NUM_DICODONS {
            let g = (gene_counts[ndx] + 1.0) / (total_gene + NUM_DICODONS as f64);
            let b = (bg_counts[ndx] + 1.0) / (total_bg + NUM_DICODONS as f64);
            model.scores[ndx] = (g / b).ln().clamp(-4.0, 4.0);
        }

        model
    }

    /// Score an ORF by summing per-dicodon log-likelihoods in its translational
    /// frame and converting the average to a positive multiplier.
    pub fn score_orf(&self, orf: &Orf) -> f64 {
        let seq = orf.sequence();
        let mut sum = 0.0;
        let mut count = 0;
        for i in (0..seq.len().saturating_sub(5)).step_by(3) {
            if let Some(ndx) = kmer_encode(seq, i, 6) {
                sum += self.scores[ndx];
                count += 1;
            }
        }
        if count == 0 {
            return 1.0;
        }
        let avg = sum / count as f64;
        avg.exp().clamp(0.1, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmer_roundtrip_6() {
        for ndx in 0..NUM_DICODONS {
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
    fn training_produces_scores() {
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
            weight_rbs: 2.0,
            hold: 100.0,
            motif_score: 1.0,
            dicodon_score: 1.0,
            weight: 1.0,
        }];
        let model = DicodonModel::train(&orfs, &seq, &rc);
        assert!(model.scores.iter().all(|&s| s.is_finite()));
        assert!(model.scores.iter().any(|&s| s > 0.0));
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
            weight_rbs: 2.0,
            hold: 100.0,
            motif_score: 1.0,
            dicodon_score: 1.0,
            weight: 1.0,
        }];
        let model = DicodonModel::train(&orfs, &seq, &rc);
        let s = model.score_orf(&orfs[0]);
        assert!(s > 0.0);
        assert!(s <= 10.0);
    }
}
