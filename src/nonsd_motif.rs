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
}
