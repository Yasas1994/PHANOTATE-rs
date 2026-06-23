//! Non-Shine-Dalgarno upstream motif finder.
//!
//! Discovers arbitrary 3-6 bp motifs enriched upstream of start codons,
//! mirroring Prodigal's train_starts_nonsd algorithm.

#[allow(unused_imports)]
use crate::orf::Orf;
#[allow(unused_imports)]
use std::collections::HashMap;

/// Number of possible spacer distance groups.
pub const NUM_SPACERS: usize = 4;
/// Minimum motif length (3 bp).
pub const MIN_MOTIF_LEN: usize = 3;
/// Maximum motif length (6 bp).
pub const MAX_MOTIF_LEN: usize = 6;
/// Maximum encoded motif index (4^6).
pub const MAX_MOTIF_INDEX: usize = 4096;

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

/// Decode a single 2-bit value to ASCII base.
fn decode_base(v: usize) -> u8 {
    match v {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => b'N',
    }
}

/// Encode a DNA word of length `len` starting at `pos` in `seq`.
/// Returns `None` if any base is ambiguous or out of range.
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

/// Decode a motif index back to a DNA string of length `len`.
pub fn kmer_decode(index: usize, len: usize) -> Vec<u8> {
    let mut seq = Vec::with_capacity(len);
    for i in 0..len {
        seq.push(decode_base((index >> (2 * i)) & 0x3));
    }
    seq
}

/// A single non-Shine-Dalgarno motif occurrence upstream of a start codon.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotifHit {
    pub len: usize,      // 3..6
    pub spacer: usize,   // 3..15
    pub spacendx: usize, // 0..3
    pub ndx: usize,      // encoded motif
    pub score: f64,
}

/// Classify a spacer (distance from motif start to coding start) into a group.
fn spacer_group(spacer: usize) -> usize {
    match spacer {
        3 | 4 => 1,
        5..=10 => 0,
        11 | 12 => 2,
        13..=15 => 3,
        _ => panic!("spacer out of range: {}", spacer),
    }
}

/// Scan positions `start-18-i .. start-6-i` for each motif length `i+3` and
/// return the highest scoring motif. If no valid motif is found, returns a
/// zeroed hit with score set to the caller's `no_mot` value.
#[allow(clippy::needless_range_loop)]
pub fn find_best_motif(
    mot_wt: &[[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    seq: &[u8],
    start: usize,
    no_mot: f64,
) -> MotifHit {
    let mut best = MotifHit {
        score: no_mot,
        ..Default::default()
    };
    for len_idx in 0..=(MAX_MOTIF_LEN - MIN_MOTIF_LEN) {
        let len = MIN_MOTIF_LEN + len_idx;
        let earliest = start.saturating_sub(18 + len);
        let latest = start.saturating_sub(6 + len);
        for pos in earliest..=latest {
            if pos + len > seq.len() {
                continue;
            }
            if let Some(ndx) = kmer_encode(seq, pos, len) {
                let spacer = start - pos - len;
                let spacendx = spacer_group(spacer);
                let score = mot_wt[len_idx][spacendx][ndx];
                if score > best.score {
                    best = MotifHit {
                        len,
                        spacer,
                        spacendx,
                        ndx,
                        score,
                    };
                }
            }
        }
    }
    best
}

/// Coverage map: 1 = exact match, 2 = mismatch-allowed, 0 = not covered.
pub type CoverageMap = [[[u8; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];

/// Build a coverage map. A motif is "good" if it contains a 3-mer subset
/// present in at least 20% of selected genes.
#[allow(clippy::needless_range_loop)]
pub fn build_coverage_map(
    real: &[[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    ngenes: f64,
) -> CoverageMap {
    let mut good = [[[0u8; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
    let thresh = 0.2;

    // 3-base motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..64 {
            if real[0][sp][j] / ngenes >= thresh {
                for k in 0..NUM_SPACERS {
                    good[0][k][j] = 1;
                }
            }
        }
    }

    // 4-base motifs need two valid 3-base sub-motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..256 {
            let d0 = (j & 0b11111100) >> 2;
            let d1 = j & 0b00111111;
            if good[0][sp][d0] == 0 || good[0][sp][d1] == 0 {
                continue;
            }
            good[1][sp][j] = 1;
        }
    }

    // 5-base motifs need three valid 3-base sub-motifs; allow interior mismatches
    for sp in 0..NUM_SPACERS {
        for j in 0..1024 {
            let d0 = (j & 0b1111110000) >> 4;
            let d1 = (j & 0b0000111100) >> 2;
            let d2 = j & 0b0000001111;
            if good[0][sp][d0] == 0 || good[0][sp][d1] == 0 || good[0][sp][d2] == 0 {
                continue;
            }
            good[2][sp][j] = 1;
            // flip bits 3 and 4 of the 5-mer (positions 2 and 3) to allow one mismatch
            let mut tmp = j;
            for k in [0, 16] {
                tmp ^= k;
                for l in [0, 32] {
                    tmp ^= l;
                    if good[2][sp][tmp] == 0 {
                        good[2][sp][tmp] = 2;
                    }
                }
            }
        }
    }

    // 6-base motifs need two valid 5-base sub-motifs
    for sp in 0..NUM_SPACERS {
        for j in 0..MAX_MOTIF_INDEX {
            let d0 = (j & 0b111111111100) >> 2;
            let d1 = j & 0b000000111111;
            if good[2][sp][d0] == 0 || good[2][sp][d1] == 0 {
                continue;
            }
            good[3][sp][j] = if good[2][sp][d0] == 1 && good[2][sp][d1] == 1 {
                1
            } else {
                2
            };
        }
    }

    good
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmer_roundtrip_3_to_6() {
        for len in 3..=6 {
            for ndx in 0..(1 << (2 * len)) {
                let seq = kmer_decode(ndx, len);
                assert_eq!(kmer_encode(&seq, 0, len), Some(ndx));
            }
        }
    }

    #[test]
    fn kmer_encode_rejects_ambiguous() {
        assert!(kmer_encode(b"atgnat", 3, 3).is_none());
    }

    #[test]
    fn coverage_map_marks_common_3mer_good() {
        let mut real = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
        let ngenes = 10.0;
        real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 3.0;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 1);
    }

    #[test]
    fn coverage_map_keeps_rare_3mer_bad() {
        let mut real = [[[0.0; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
        let ngenes = 10.0;
        real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 1.0;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 0);
    }
}
