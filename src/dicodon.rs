//! Prodigal-style dicodon (6-mer) coding-potential scorer.

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
}
