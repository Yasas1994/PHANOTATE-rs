//! Shared 2-bit DNA k-mer encoding utilities.

/// 2-bit encode a single base: A=0, C=1, G=2, T=3.
/// Returns `None` for ambiguous/non-ACGT bases.
#[inline]
pub fn encode_base(b: u8) -> Option<usize> {
    match b {
        b'a' | b'A' => Some(0),
        b'c' | b'C' => Some(1),
        b'g' | b'G' => Some(2),
        b't' | b'T' => Some(3),
        _ => None,
    }
}

/// Encode a DNA word of length `len` starting at `pos`.
/// Returns `None` if any base is ambiguous or out of range.
#[inline]
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
