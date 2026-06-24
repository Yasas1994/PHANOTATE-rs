//! Non-Shine-Dalgarno upstream motif finder.
//!
//! Discovers arbitrary 3-6 bp motifs enriched upstream of start codons,
//! mirroring Prodigal's train_starts_nonsd algorithm.

use std::collections::HashMap;

use crate::orf::Orf;

/// Number of possible spacer distance groups.
pub const NUM_SPACERS: usize = 4;
/// Minimum motif length (3 bp).
pub const MIN_MOTIF_LEN: usize = 3;
/// Maximum motif length (6 bp).
pub const MAX_MOTIF_LEN: usize = 6;
/// Maximum encoded motif index (4^6).
pub const MAX_MOTIF_INDEX: usize = 4096;

/// Weight array indexed by `[motif_length_index][spacer_group][motif_index]`.
pub type MotifWeights =
    Box<[[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1]>;

/// Allocate a zeroed motif weight array directly on the heap.
///
/// The array is 4 * 4 * 4096 * 8 = 512 KiB, which is fine to allocate safely
/// via `Box::new` without resorting to raw allocation.
fn zero_motif_weights() -> MotifWeights {
    Box::new([[[0.0f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1])
}

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

/// Format a discovered non-SD motif hit as an uppercase DNA string.
pub fn format_motif_hit(hit: &MotifHit) -> Option<String> {
    if hit.len == 0 {
        None
    } else {
        Some(String::from_utf8(kmer_decode(hit.ndx, hit.len)).unwrap())
    }
}

/// A single non-Shine-Dalgarno motif occurrence upstream of a start codon.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotifHit {
    /// Motif length in bp (3..6).
    pub len: usize,
    /// Distance from motif start to coding start (3..18).
    pub spacer: usize,
    /// Spacer group index (0..3).
    pub spacendx: usize,
    /// Encoded motif index.
    pub ndx: usize,
    pub score: f64,
}

/// Classify a spacer (distance from motif start to coding start) into a group.
///
/// Returns `None` for distances outside the supported 3–18 bp window.
fn spacer_group(spacer: usize) -> Option<usize> {
    match spacer {
        3 | 4 => Some(1),
        5..=10 => Some(0),
        11 | 12 => Some(2),
        13..=18 => Some(3),
        _ => None,
    }
}

/// Scan positions `start-18-i .. start-6-i` for each motif length `i+3` and
/// return the highest scoring motif. If no valid motif is found, returns a
/// zeroed hit with score set to the caller's `no_mot` value.
#[allow(clippy::needless_range_loop)]
pub fn find_best_motif(mot_wt: &MotifWeights, seq: &[u8], start: usize, no_mot: f64) -> MotifHit {
    let mut best = MotifHit {
        score: no_mot,
        ..Default::default()
    };
    // Iterate from longest to shortest so that equally-scored longer motifs are
    // preferred, and use `>` so the earliest equally-scored hit wins.
    for len_idx in (0..=(MAX_MOTIF_LEN - MIN_MOTIF_LEN)).rev() {
        let len = MIN_MOTIF_LEN + len_idx;
        let earliest = start.saturating_sub(18 + len);
        let latest = start.saturating_sub(3 + len);
        for pos in earliest..=latest {
            if pos + len > seq.len() {
                continue;
            }
            if let Some(ndx) = kmer_encode(seq, pos, len) {
                let spacer = start - pos - len;
                let Some(spacendx) = spacer_group(spacer) else {
                    continue;
                };
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
pub fn build_coverage_map(real: &MotifWeights, ngenes: f64) -> CoverageMap {
    let mut good = [[[0u8; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1];
    if ngenes == 0.0 {
        return good;
    }
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
            let d1 = (j & 0b11111100) >> 2;
            let d2 = j & 0b00111111;
            if good[0][sp][d0] == 0 || good[0][sp][d1] == 0 || good[0][sp][d2] == 0 {
                continue;
            }
            good[2][sp][j] = 1;
            // flip the bits of the middle position (position 2) to allow one mismatch
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
            let d1 = j & 0b001111111111;
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

/// Compute a 0-order Markov background for all motif lengths.
///
/// Uses the genome-wide single-base frequencies (both strands) to estimate the
/// expected probability of every length-3..6 word, then spreads that
/// probability evenly across spacer groups.  This is more stable than an
/// ORF-upstream background when coding_score == 0, because it does not treat
/// the planted motif as background merely because it is common in the genome.
#[allow(clippy::needless_range_loop)]
fn markov_motif_background(dna: &[u8], rc_dna: &[u8]) -> MotifWeights {
    let mut base_counts = [0.0; 4];
    let mut total = 0.0;
    for seq in [dna, rc_dna] {
        for &b in seq {
            if let Some(idx) = encode_base(b) {
                base_counts[idx] += 1.0;
                total += 1.0;
            }
        }
    }
    let freqs: [f64; 4] = base_counts.map(|c| if total > 0.0 { c / total } else { 0.25 });

    let mut bg = zero_motif_weights();
    for len_idx in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
        let len = MIN_MOTIF_LEN + len_idx;
        for mi in 0..(1 << (2 * len)) {
            let mut prob = 1.0;
            for i in 0..len {
                prob *= freqs[(mi >> (2 * i)) & 0x3];
            }
            for sp in 0..NUM_SPACERS {
                bg[len_idx][sp][mi] = prob / NUM_SPACERS as f64;
            }
        }
    }
    bg
}

/// Trained non-SD motif model.
pub struct NonSdModel {
    pub mot_wt: MotifWeights,
    pub no_mot: f64,
    pub type_wt: [f64; 3],
    pub ups_comp: [[f64; 4]; 32],
}

impl Default for NonSdModel {
    fn default() -> Self {
        Self {
            mot_wt: zero_motif_weights(),
            no_mot: 0.0,
            type_wt: [0.0; 3],
            ups_comp: [[0.0; 4]; 32],
        }
    }
}

impl NonSdModel {
    /// Score a single ORF and return a positive multiplier.
    pub fn score_orf(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> f64 {
        let (wseq, start) = upstream_context(dna, rc_dna, orf);
        if start < 18 + MIN_MOTIF_LEN {
            return 1.0;
        }
        let hit = find_best_motif(&self.mot_wt, wseq, start, self.no_mot);
        let type_idx = start_codon_index(orf.start_codon()).unwrap_or(0);
        let type_bonus = self.type_wt[type_idx];
        let comp_bonus = score_upstream_composition(wseq, start, &self.ups_comp);
        let log_score = hit.score + type_bonus + comp_bonus;
        log_score.exp().clamp(0.25, 4.0)
    }

    /// Return the best non-SD motif label for an ORF, if one exists.
    pub fn best_motif_label(&self, orf: &Orf, dna: &[u8], rc_dna: &[u8]) -> Option<String> {
        let (wseq, start) = upstream_context(dna, rc_dna, orf);
        if start < 18 + MIN_MOTIF_LEN {
            return None;
        }
        let hit = find_best_motif(&self.mot_wt, wseq, start, self.no_mot);
        format_motif_hit(&hit)
    }

    /// Train a non-SD motif model from ORFs using a 20-iteration 3-stage EM loop.
    pub fn train(
        orfs: &[Orf],
        dna: &[u8],
        rc_dna: &[u8],
        start_weights: &HashMap<Vec<u8>, f64>,
    ) -> Self {
        let mut model = Self::default();
        let st_wt = 4.35;

        // Initialize type_wt from start codon weights.
        let atg_w = *start_weights.get(&b"atg".to_vec()).unwrap_or(&1.0);
        let gtg_w = *start_weights.get(&b"gtg".to_vec()).unwrap_or(&1.0);
        let ttg_w = *start_weights.get(&b"ttg".to_vec()).unwrap_or(&1.0);
        let max_w = atg_w.max(gtg_w).max(ttg_w);
        if max_w > 0.0 {
            model.type_wt[0] = (atg_w / max_w).ln();
            model.type_wt[1] = (gtg_w / max_w).ln();
            model.type_wt[2] = (ttg_w / max_w).ln();
        }

        // Background type frequencies across all ORFs.
        let mut tbg = [0.0; 3];
        for orf in orfs {
            if let Some(idx) = start_codon_index(orf.start_codon()) {
                tbg[idx] += 1.0;
            }
        }
        let tbg_sum: f64 = tbg.iter().sum();
        if tbg_sum > 0.0 {
            for v in &mut tbg {
                *v /= tbg_sum;
            }
        }

        // Markov background.  Unlike the previous ORF-upstream background, this
        // stays stable when coding_score == 0 and lets the EM bootstrap from a
        // shared upstream motif even when that motif is repeated in the genome.
        let genome_bg = markov_motif_background(dna, rc_dna);

        let mut sthresh = 10.0;

        for iter in 0..20 {
            let stage = if iter < 4 {
                0
            } else if iter < 12 {
                1
            } else {
                2
            };

            // Background motif counts: use the genomic distribution.
            let mbg = &genome_bg;
            let zbg = 0.0;

            // Real counts: group ORFs by (stop, frame), pick best start per group.
            let mut mreal = zero_motif_weights();
            let mut zreal = 0.0;
            let mut treal = [0.0; 3];
            let mut ngenes = 0.0;

            let groups = group_orfs_by_stop(orfs);
            for group in groups.values() {
                let best = group
                    .iter()
                    .filter_map(|&orf| {
                        let (wseq, start) = upstream_context(dna, rc_dna, orf);
                        if start < 18 + MIN_MOTIF_LEN {
                            return None;
                        }
                        let hit = find_best_motif(&model.mot_wt, wseq, start, model.no_mot);
                        let type_idx = start_codon_index(orf.start_codon()).unwrap_or(0);
                        let coding_score = 0.0;
                        let score = coding_score + st_wt * (hit.score + model.type_wt[type_idx]);
                        Some((orf, wseq, start, hit, score, type_idx))
                    })
                    .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap());

                if let Some((_orf, wseq, start, hit, score, type_idx)) = best {
                    // In stage 0, seed the model with every ORF that has enough
                    // upstream context.  Without this, the EM cannot bootstrap
                    // when the coding signal has been neutralised (coding_score=0).
                    if stage == 0 || score >= sthresh {
                        ngenes += 1.0;
                        treal[type_idx] += 1.0;
                        update_motif_counts(&mut mreal, &mut zreal, wseq, start, &hit, stage);
                        if iter == 19 {
                            count_upstream_composition(wseq, start, &mut model.ups_comp);
                        }
                    }
                }
            }

            // Coverage filter (stages 0 and 1 only).
            if stage < 2 {
                let mgood = build_coverage_map(&mreal, ngenes);
                for li in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                    for si in 0..NUM_SPACERS {
                        for mi in 0..MAX_MOTIF_INDEX {
                            if mgood[li][si][mi] == 0 {
                                zreal += mreal[li][si][mi];
                                mreal[li][si][mi] = 0.0;
                            }
                        }
                    }
                }
            }

            // Weight update (all stages).
            let mreal_sum = mreal
                .iter()
                .flat_map(|a| a.iter())
                .flat_map(|b| b.iter())
                .sum::<f64>()
                + zreal;
            if mreal_sum == 0.0 {
                model.mot_wt = zero_motif_weights();
                model.no_mot = -4.0;
            } else if stage == 0 {
                // Stage 0 seeds the model using per-gene occurrence frequency
                // scaled by motif length.  This lets the EM bootstrap from a
                // shared upstream motif even when coding_score == 0.
                for li in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                    let len = MIN_MOTIF_LEN + li;
                    let len_bonus = (len * len) as f64;
                    for si in 0..NUM_SPACERS {
                        for mi in 0..MAX_MOTIF_INDEX {
                            let per_gene = mreal[li][si][mi] / ngenes;
                            model.mot_wt[li][si][mi] = (per_gene * len_bonus)
                                .max(f64::MIN_POSITIVE)
                                .ln()
                                .clamp(-4.0, 4.0);
                        }
                    }
                }
                model.no_mot = -4.0;
            } else {
                for li in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                    for si in 0..NUM_SPACERS {
                        for mi in 0..MAX_MOTIF_INDEX {
                            mreal[li][si][mi] /= mreal_sum;
                            model.mot_wt[li][si][mi] = if mbg[li][si][mi] != 0.0 {
                                (mreal[li][si][mi] / mbg[li][si][mi]).ln()
                            } else {
                                -4.0
                            }
                            .clamp(-4.0, 4.0);
                        }
                    }
                }
                model.no_mot = if zbg != 0.0 {
                    (zreal / mreal_sum / zbg).ln()
                } else {
                    -4.0
                }
                .clamp(-4.0, 4.0);
            }

            // Update type weights.
            let treal_sum: f64 = treal.iter().sum();
            if treal_sum == 0.0 {
                model.type_wt = [0.0; 3];
            } else {
                for i in 0..3 {
                    let real = treal[i] / treal_sum;
                    model.type_wt[i] = if tbg[i] != 0.0 {
                        (real / tbg[i]).ln()
                    } else {
                        -4.0
                    }
                    .clamp(-4.0, 4.0);
                }
            }

            if treal_sum <= orfs.len() as f64 / 2000.0 {
                sthresh /= 2.0;
            }
        }

        finalize_upstream_composition(&mut model.ups_comp, dna, rc_dna);

        model
    }
}

/// Map a start codon to its type index (ATG=0, GTG=1, TTG=2).
pub fn start_codon_index(codon: &[u8]) -> Option<usize> {
    match codon {
        b"atg" | b"ATG" => Some(0),
        b"gtg" | b"GTG" => Some(1),
        b"ttg" | b"TTG" => Some(2),
        _ => None,
    }
}

/// Return the sequence and 0-based start-codon index to scan upstream of `orf`.
pub fn upstream_context<'a>(dna: &'a [u8], rc_dna: &'a [u8], orf: &Orf) -> (&'a [u8], usize) {
    if orf.frame > 0 {
        (dna, orf.start - 1)
    } else {
        let start = dna.len() - orf.start;
        (rc_dna, start)
    }
}

/// Group ORFs by their stop coordinate and frame.
fn group_orfs_by_stop(orfs: &[Orf]) -> HashMap<(usize, i8), Vec<&Orf>> {
    let mut map: HashMap<(usize, i8), Vec<&Orf>> = HashMap::new();
    for orf in orfs {
        map.entry((orf.stop, orf.frame)).or_default().push(orf);
    }
    map
}

/// Update motif counts for one ORF according to the EM stage.
///
/// Stage 0 counts all possible motifs in the upstream window to seed the model.
/// Stages 1 and 2 count the chosen hit (and its sub-motifs in stage 1).
#[allow(clippy::needless_range_loop)]
fn update_motif_counts(
    mcnt: &mut [[[f64; MAX_MOTIF_INDEX]; NUM_SPACERS]; MAX_MOTIF_LEN - MIN_MOTIF_LEN + 1],
    zero: &mut f64,
    seq: &[u8],
    start: usize,
    hit: &MotifHit,
    stage: usize,
) {
    if stage != 0 && hit.len == 0 {
        *zero += 1.0;
        return;
    }

    match stage {
        0 => {
            for len_idx in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
                let len = MIN_MOTIF_LEN + len_idx;
                let earliest = start.saturating_sub(18 + len);
                let latest = start.saturating_sub(3 + len);
                for pos in earliest..=latest {
                    if pos + len > seq.len() {
                        continue;
                    }
                    if let Some(ndx) = kmer_encode(seq, pos, len) {
                        for sp in 0..NUM_SPACERS {
                            mcnt[len_idx][sp][ndx] += 1.0;
                        }
                    }
                }
            }
        }
        1 => {
            mcnt[hit.len - MIN_MOTIF_LEN][hit.spacendx][hit.ndx] += 1.0;
            for sub_len_idx in 0..hit.len - MIN_MOTIF_LEN {
                let sub_len = MIN_MOTIF_LEN + sub_len_idx;
                // Sub-motifs must stay inside the 18 bp upstream window, so
                // clamp the earliest position to keep spacer <= 18.
                let earliest = start
                    .saturating_sub(hit.spacer + hit.len)
                    .max(start.saturating_sub(18 + sub_len));
                let latest = start.saturating_sub(hit.spacer + sub_len);
                for pos in earliest..=latest {
                    if pos + sub_len > seq.len() {
                        continue;
                    }
                    if let Some(ndx) = kmer_encode(seq, pos, sub_len) {
                        let spacer = start - pos - sub_len;
                        let Some(sp) = spacer_group(spacer) else {
                            continue;
                        };
                        mcnt[sub_len_idx][sp][ndx] += 1.0;
                    }
                }
            }
        }
        _ => {
            mcnt[hit.len - MIN_MOTIF_LEN][hit.spacendx][hit.ndx] += 1.0;
        }
    }
}

/// Accumulate upstream composition counts around a selected start.
fn count_upstream_composition(seq: &[u8], start: usize, ups_comp: &mut [[f64; 4]; 32]) {
    let mut count = 0;
    for i in 1..45 {
        if i > 2 && i < 15 {
            continue;
        }
        if start >= i {
            if let Some(base) = encode_base(seq[start - i]) {
                ups_comp[count][base] += 1.0;
            }
        }
        count += 1;
    }
}

/// Score upstream composition using trained log-odds weights.
fn score_upstream_composition(seq: &[u8], start: usize, ups_comp: &[[f64; 4]; 32]) -> f64 {
    let mut score = 0.0;
    let mut count = 0;
    for i in 1..45 {
        if i > 2 && i < 15 {
            continue;
        }
        if start >= i {
            if let Some(base) = encode_base(seq[start - i]) {
                score += 0.4 * 4.35 * ups_comp[count][base];
            }
        }
        count += 1;
    }
    score
}

/// Convert raw upstream composition counts to log-odds weights.
#[allow(clippy::needless_range_loop)]
fn finalize_upstream_composition(ups_comp: &mut [[f64; 4]; 32], dna: &[u8], rc_dna: &[u8]) {
    let mut gc_count = 0.0;
    let mut at_count = 0.0;
    for &b in dna.iter().chain(rc_dna.iter()) {
        match b {
            b'g' | b'G' | b'c' | b'C' => gc_count += 1.0,
            b'a' | b'A' | b't' | b'T' => at_count += 1.0,
            _ => {}
        }
    }
    let gc = if gc_count + at_count > 0.0 {
        gc_count / (gc_count + at_count)
    } else {
        0.5
    };

    for row in ups_comp.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum == 0.0 {
            continue;
        }
        for j in 0..4 {
            row[j] /= sum;
            let expected = if j == 0 || j == 3 {
                if gc > 0.1 && gc < 0.9 {
                    1.0 - gc
                } else if gc <= 0.1 {
                    0.90
                } else {
                    0.10
                }
            } else if gc > 0.1 && gc < 0.9 {
                gc
            } else if gc <= 0.1 {
                0.10
            } else {
                0.90
            };
            row[j] = (row[j] * 2.0 / expected).ln().clamp(-4.0, 4.0);
        }
    }
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
        let mut real = zero_motif_weights();
        let ngenes = 10.0;
        real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 3.0;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 1);
    }

    #[test]
    fn coverage_map_keeps_rare_3mer_bad() {
        let mut real = zero_motif_weights();
        let ngenes = 10.0;
        real[0][0][kmer_encode(b"aaa", 0, 3).unwrap()] = 1.0;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[0][0][kmer_encode(b"aaa", 0, 3).unwrap()], 0);
    }

    #[test]
    fn coverage_map_4mer_propagates_from_3mers() {
        let mut real = zero_motif_weights();
        let ngenes = 10.0;
        let sp = 0;
        // 4-mer "AACG" needs 3-mers "AAC" (positions 0-2) and "ACG" (positions 1-3).
        real[0][sp][kmer_encode(b"AAC", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"ACG", 0, 3).unwrap()] = ngenes;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[1][sp][kmer_encode(b"AACG", 0, 4).unwrap()], 1);
    }

    #[test]
    fn coverage_map_5mer_propagates_from_3mers() {
        let mut real = zero_motif_weights();
        let ngenes = 10.0;
        let sp = 0;
        // 5-mer "AACGT" needs 3-mers "AAC", "ACG", and "CGT".
        real[0][sp][kmer_encode(b"AAC", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"ACG", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"CGT", 0, 3).unwrap()] = ngenes;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[2][sp][kmer_encode(b"AACGT", 0, 5).unwrap()], 1);
    }

    #[test]
    fn coverage_map_6mer_propagates_from_5mers() {
        let mut real = zero_motif_weights();
        let ngenes = 10.0;
        let sp = 0;
        // 6-mer "AACGTT" needs 5-mers "AACGT" and "ACGTT".
        // "AACGT" needs 3-mers AAC, ACG, CGT; "ACGTT" needs ACG, CGT, GTT.
        real[0][sp][kmer_encode(b"AAC", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"ACG", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"CGT", 0, 3).unwrap()] = ngenes;
        real[0][sp][kmer_encode(b"GTT", 0, 3).unwrap()] = ngenes;
        let good = build_coverage_map(&real, ngenes);
        assert_eq!(good[3][sp][kmer_encode(b"AACGTT", 0, 6).unwrap()], 1);
    }

    #[test]
    fn find_best_motif_spacer_three() {
        // Motif "AAAAAA" placed 3 bp upstream of the start codon.
        // 12 bp upstream spacer + 6 bp motif + 3 bp gap + ATG = 24 bp.
        // The start codon begins at 0-based index 21.
        let seq = b"ccccccccccccaaaaaacccatg";
        let mut weights = zero_motif_weights();
        let ndx = kmer_encode(b"aaaaaa", 0, 6).unwrap();
        let sp = spacer_group(3).unwrap();
        weights[3][sp][ndx] = 5.0;

        let hit = find_best_motif(&weights, seq, 21, 0.0);
        assert_eq!(hit.len, 6);
        assert_eq!(hit.spacer, 3);
        assert_eq!(kmer_decode(hit.ndx, 6), b"AAAAAA".to_vec());
    }

    #[test]
    fn training_runs_without_panic_and_produces_bounded_model() {
        let motif = b"aaaaaa";
        let mut seq = Vec::new();
        let mut orfs = Vec::new();

        // 40 ORFs with a planted upstream motif.
        // Each block is: 9 bp spacer + 6 bp motif + 6 bp gap + 3 bp start +
        // 3 bp filler + 3 bp stop + 3 bp inter-ORF spacer = 33 bp.
        // The motif is 9 bp from the block start and 6 bp upstream of the start codon.
        for i in 0..40 {
            seq.extend_from_slice(b"ccccccccc"); // 9 bp upstream spacer
            seq.extend_from_slice(motif); // 6 bp planted motif
            seq.extend_from_slice(b"cccccc"); // 6 bp gap to start codon
            seq.extend_from_slice(b"ttg"); // start
            seq.extend_from_slice(b"atg"); // filler
            seq.extend_from_slice(b"taa"); // stop
            seq.extend_from_slice(b"ccc"); // inter-ORF spacer
            let hold = 1.0;
            orfs.push(Orf {
                start: i * 33 + 22,
                stop: i * 33 + 30,
                frame: 1,
                seq: seq[i * 33 + 21..i * 33 + 30].to_vec(),
                rbs_score: 0,
                pstop: 0.01,
                weight_rbs: 1.0,
                hold,
                dicodon_score: 1.0 / hold,
                motif_score: 1.0,
                rbs_motif: None,
                weight: 1.0,
            });
        }

        // 10 decoy ORFs with a shorter 5-mer.
        // The decoys share 3-5-mers with the real motif but not the full 6-mer.
        for i in 40..50 {
            seq.extend_from_slice(b"ccccccccc");
            seq.extend_from_slice(b"aaaaac"); // 5 A's, no full AAAAAA
            seq.extend_from_slice(b"cccccc");
            seq.extend_from_slice(b"ttg");
            seq.extend_from_slice(b"atg");
            seq.extend_from_slice(b"taa");
            seq.extend_from_slice(b"ccc");
            let hold = 1.0;
            orfs.push(Orf {
                start: i * 33 + 22,
                stop: i * 33 + 30,
                frame: 1,
                seq: seq[i * 33 + 21..i * 33 + 30].to_vec(),
                rbs_score: 0,
                pstop: 0.01,
                weight_rbs: 1.0,
                hold,
                dicodon_score: 1.0 / hold,
                motif_score: 1.0,
                rbs_motif: None,
                weight: 1.0,
            });
        }

        let rc = crate::genome::rev_comp(&seq);
        let mut weights = HashMap::new();
        weights.insert(b"ttg".to_vec(), 1.0);
        let model = NonSdModel::train(&orfs, &seq, &rc, &weights);

        // The trained weights are clamped to [-4, 4] during EM.
        assert!(model.no_mot >= -4.0 && model.no_mot <= 4.0);
        for &w in &model.type_wt {
            assert!(w >= -4.0 && w <= 4.0);
        }
        for li in 0..=MAX_MOTIF_LEN - MIN_MOTIF_LEN {
            for si in 0..NUM_SPACERS {
                for mi in 0..MAX_MOTIF_INDEX {
                    let w = model.mot_wt[li][si][mi];
                    assert!(
                        w >= -4.0 && w <= 4.0,
                        "weight out of bounds at [{li}][{si}][{mi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn training_finds_planted_motif_without_hold_signal() {
        let motif = b"aaaaaa";
        let prefix = vec![b'c'; 600];
        let gap = vec![b'c'; 6];
        let start_codon = b"ttg";
        let filler = b"atg";
        let stop = b"taa";
        let unit_len = motif.len() + gap.len() + start_codon.len() + filler.len() + stop.len();

        let mut seq: Vec<u8> = prefix.clone();
        let n = 100;
        for _ in 0..n {
            seq.extend_from_slice(motif);
            seq.extend_from_slice(&gap);
            seq.extend_from_slice(start_codon);
            seq.extend_from_slice(filler);
            seq.extend_from_slice(stop);
        }

        let mut orfs = Vec::new();
        for i in 0..n {
            let block_start = prefix.len() + i * unit_len;
            let start_1based = block_start + motif.len() + gap.len() + 1;
            let stop_1based =
                block_start + motif.len() + gap.len() + start_codon.len() + filler.len() + 1;
            let orf_seq_start = start_1based - 1;
            let orf_seq_end = stop_1based - 1 + stop.len();
            let hold = 1.0;
            orfs.push(Orf {
                start: start_1based,
                stop: stop_1based,
                frame: 1,
                seq: seq[orf_seq_start..orf_seq_end].to_vec(),
                rbs_score: 0,
                pstop: 0.01,
                weight_rbs: 1.0,
                hold,
                dicodon_score: 1.0 / hold,
                motif_score: 1.0,
                rbs_motif: None,
                weight: 1.0,
            });
        }

        let rc = crate::genome::rev_comp(&seq);
        let mut weights = HashMap::new();
        weights.insert(b"ttg".to_vec(), 1.0);
        let model = NonSdModel::train(&orfs, &seq, &rc, &weights);

        // Query the best motif at the first ORF start with enough upstream context.
        let query_start = prefix.len() + motif.len() + gap.len();
        let hit = find_best_motif(&model.mot_wt, &seq, query_start, model.no_mot);
        assert_eq!(hit.len, 6);
        assert_eq!(kmer_decode(hit.ndx, 6), b"AAAAAA".to_vec());
    }

    #[test]
    fn score_orf_returns_reasonable_multiplier() {
        let model = NonSdModel::default();
        let hold = 100.0;
        let orf = Orf {
            start: 25,
            stop: 60,
            frame: 1,
            seq: vec![b'a'; 36],
            rbs_score: 0,
            pstop: 0.01,
            weight_rbs: 1.0,
            hold,
            dicodon_score: 1.0 / hold,
            motif_score: 1.0,
            rbs_motif: None,
            weight: 1.0,
        };
        let dna = vec![b'a'; 100];
        let rc = dna.clone();
        let s = model.score_orf(&orf, &dna, &rc);
        assert!(s > 0.0);
        assert!(s <= 4.0);
    }

    #[test]
    fn best_motif_label_finds_planted_motif_with_enough_context() {
        let mut model = NonSdModel::default();
        let motif = b"AAAAAA";
        let len_idx = 6 - MIN_MOTIF_LEN;
        let spacer = 6;
        let sp = spacer_group(spacer).unwrap();
        let ndx = kmer_encode(motif, 0, 6).unwrap();
        model.mot_wt[len_idx][sp][ndx] = 5.0;
        model.no_mot = -4.0;

        // Build: 21 bp prefix + motif + 6 bp gap + ATG, placing the start codon
        // 1-based so there is enough upstream context.
        let prefix = vec![b'c'; 21];
        let gap = vec![b'c'; spacer];
        let mut seq = prefix;
        seq.extend_from_slice(motif);
        seq.extend_from_slice(&gap);
        seq.extend_from_slice(b"ATGAAA");

        let orf = Orf {
            start: 34,
            stop: 39,
            frame: 1,
            seq: seq[33..39].to_vec(),
            rbs_score: 0,
            pstop: 0.01,
            weight_rbs: 1.0,
            hold: 100.0,
            dicodon_score: 0.01,
            motif_score: 1.0,
            rbs_motif: None,
            weight: 1.0,
        };
        let rc = crate::genome::rev_comp(&seq);

        let label = model.best_motif_label(&orf, &seq, &rc);
        assert!(label.is_some());
        assert_eq!(label.unwrap(), "AAAAAA");
    }

    #[test]
    fn best_motif_label_returns_none_without_upstream_context() {
        let mut model = NonSdModel::default();
        let motif = b"AAAAAA";
        let len_idx = 6 - MIN_MOTIF_LEN;
        let sp = spacer_group(6).unwrap();
        let ndx = kmer_encode(motif, 0, 6).unwrap();
        model.mot_wt[len_idx][sp][ndx] = 5.0;
        model.no_mot = -4.0;

        let seq = b"ATGAAA".to_vec();
        let orf = Orf {
            start: 1,
            stop: 6,
            frame: 1,
            seq: seq.clone(),
            rbs_score: 0,
            pstop: 0.01,
            weight_rbs: 1.0,
            hold: 100.0,
            dicodon_score: 0.01,
            motif_score: 1.0,
            rbs_motif: None,
            weight: 1.0,
        };
        let rc = crate::genome::rev_comp(&seq);

        assert!(model.best_motif_label(&orf, &seq, &rc).is_none());
    }
}
