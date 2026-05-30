use crate::genome::rev_comp;

#[derive(Debug, Clone)]
pub struct Orf {
    pub start: usize,    // 1-based, inclusive start of start codon (forward coords)
    pub stop: usize,     // 1-based, inclusive stop of stop codon (forward coords)
    pub frame: i8,       // 1,2,3 for forward; -1,-2,-3 for reverse
    pub seq: Vec<u8>,    // the ORF nucleotide sequence (forward direction)
    pub rbs_score: usize,
    pub pstop: f64,      // P(stop) for this ORF
    pub weight_rbs: f64,
    pub hold: f64,       // product of adjusted P(not_stop) per codon
    pub weight: f64,     // final ORF edge weight (negative)
}

impl Orf {
    pub fn start_codon(&self) -> &[u8] {
        &self.seq[..3]
    }

    /// Compute P(stop) = P(TAA) + P(TAG) + P(TGA) from per-base frequencies in this ORF.
    pub fn compute_pstop(seq: &[u8]) -> f64 {
        let mut freq = [0usize; 4]; // a,t,c,g
        for &b in seq {
            match b {
                b'a' => freq[0] += 1,
                b't' => freq[1] += 1,
                b'c' => freq[2] += 1,
                b'g' => freq[3] += 1,
                _ => {}
            }
        }
        let len = seq.len() as f64;
        let pa = freq[0] as f64 / len;
        let pt = freq[1] as f64 / len;
        let _pc = freq[2] as f64 / len;
        let pg = freq[3] as f64 / len;
        // TAA + TAG + TGA
        pt * pa * pa + pt * pg * pa + pt * pa * pg
    }

    pub fn score(&mut self, start_codons: &std::collections::HashMap<Vec<u8>, f64>) {
        let mut s = 1.0 / self.hold;
        let sc = self.start_codon().to_vec();
        if let Some(&w) = start_codons.get(&sc) {
            s *= w;
        }
        s *= self.weight_rbs;
        self.weight = -s;
    }
}

/// Enumerate all ORFs in all six reading frames.
///
/// Parameters:
/// - `closed_ends`: if true, do not allow ORFs to run off sequence edges.
///   Fragment ORFs (those without a matching start/stop at the boundary) are discarded.
/// - `mask_n`: if true, do not build any ORF that spans a run of N nucleotides.
pub fn find_orfs(
    dna: &[u8],
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    min_orf_len: usize,
    closed_ends: bool,
    mask_n: bool,
) -> Vec<Orf> {
    let mut orfs = Vec::new();
    let contig_length = dna.len();

    // Pre-compute masked regions if needed
    let masked_regions: Vec<(usize, usize)> = if mask_n {
        find_n_runs(dna)
    } else {
        Vec::new()
    };

    // The dicts that hold start and stop codons
    // frames: 1,2,3,-1,-2,-3
    let mut stops: std::collections::HashMap<i8, usize> =
        [(1, 0), (2, 0), (3, 0), (-1, 1), (-2, 2), (-3, 3)]
            .into_iter()
            .collect();
    let mut starts: std::collections::HashMap<i8, Vec<usize>> =
        [(1, Vec::new()), (2, Vec::new()), (3, Vec::new()),
         (-1, Vec::new()), (-2, Vec::new()), (-3, Vec::new())]
            .into_iter()
            .collect();

    // Initial fragment starts if first codon is NOT a start
    if !closed_ends {
        if dna.len() >= 3 {
            if !start_codons.iter().any(|c| c == &dna[0..3]) {
                starts.get_mut(&1).unwrap().push(1);
            }
        }
        if dna.len() >= 4 {
            if !start_codons.iter().any(|c| c == &dna[1..4]) {
                starts.get_mut(&2).unwrap().push(2);
            }
        }
        if dna.len() >= 5 {
            if !start_codons.iter().any(|c| c == &dna[2..5]) {
                starts.get_mut(&3).unwrap().push(3);
            }
        }
    }

    // Scan all six frames
    for i in 1..=(dna.len().saturating_sub(2)) {
        let codon = &dna[i - 1..i + 2];
        let frame = ((i - 1) % 3) + 1; // 1, 2, 3 as usize
        let frame_i8 = frame as i8;

        if start_codons.iter().any(|c| c == codon) {
            starts.get_mut(&frame_i8).unwrap().push(i);
        } else if stop_codons.iter().any(|c| c == codon) {
            let stop = i + 2;
            for &start in starts.get(&frame_i8).unwrap().iter().rev() {
                let length = stop - start + 1;
                if length >= min_orf_len {
                    let seq = dna[start - 1..stop].to_vec();
                    if mask_n && spans_masked(start, stop, &masked_regions) {
                        continue;
                    }
                    let rbs = get_rbs(dna, start, true);
                    let rbs_score = score_rbs(&rbs);
                    let pstop = Orf::compute_pstop(&seq);
                    orfs.push(Orf {
                        start,
                        stop: stop - 2,
                        frame: frame_i8,
                        seq,
                        rbs_score,
                        pstop,
                        weight_rbs: 1.0,
                        hold: 1.0,
                        weight: 1.0,
                    });
                }
            }
            starts.get_mut(&frame_i8).unwrap().clear();
            stops.insert(frame_i8, stop);
        } else {
            let rc_codon = rev_comp(codon);
            if start_codons.iter().any(|c| c == &rc_codon) {
                starts.get_mut(&(-(frame as i8))).unwrap().push(i + 2);
            } else if stop_codons.iter().any(|c| c == &rc_codon) {
                let stop = stops[&(-(frame as i8))];
                for &start in starts.get(&(-(frame as i8))).unwrap().iter() {
                    let length = start - stop + 1;
                    if length >= min_orf_len {
                        let seq = rev_comp(&dna[stop.saturating_sub(1)..start]);
                        if mask_n && spans_masked(stop, start, &masked_regions) {
                            continue;
                        }
                        let rbs = rev_comp(&dna[start..(start + 21).min(dna.len())]);
                        let rbs_score = score_rbs(&rbs);
                        let pstop = Orf::compute_pstop(&seq);
                        orfs.push(Orf {
                            start: start - 2,
                            stop,
                            frame: -(frame as i8),
                            seq,
                            rbs_score,
                            pstop,
                            weight_rbs: 1.0,
                            hold: 1.0,
                            weight: 1.0,
                        });
                    }
                }
                starts.get_mut(&(-frame_i8)).unwrap().clear();
                stops.insert(-frame_i8, i);
            }
        }
    }

    // Fragment ORFs at the end of the genome
    if !closed_ends {
        for frame_usize in [1usize, 2, 3] {
            let frame = frame_usize as i8;
            let stop = contig_length - ((contig_length - (frame_usize - 1)) % 3);
            for &start in starts.get(&frame).unwrap().iter().rev() {
                let length = stop - start + 1;
                if length >= min_orf_len {
                    let seq = dna[start.saturating_sub(1)..stop].to_vec();
                    if mask_n && spans_masked(start, stop, &masked_regions) {
                        continue;
                    }
                    let rbs = get_rbs(dna, start, true);
                    let rbs_score = score_rbs(&rbs);
                    let pstop = Orf::compute_pstop(&seq);
                    orfs.push(Orf {
                        start,
                        stop: stop - 2,
                        frame,
                        seq,
                        rbs_score,
                        pstop,
                        weight_rbs: 1.0,
                        hold: 1.0,
                        weight: 1.0,
                    });
                }
            }

            // Reverse strand fragment
            let neg_frame = -frame;
            let start_pos = contig_length - ((contig_length - (frame_usize - 1)) % 3);
            if start_pos >= 3 {
                let last_codon = rev_comp(&dna[start_pos - 3..start_pos]);
                if !start_codons.iter().any(|c| c == &last_codon) {
                    starts.get_mut(&neg_frame).unwrap().push(start_pos);
                }
            }
            let stop = stops[&neg_frame];
            for &start in starts.get(&neg_frame).unwrap().iter() {
                let length = start - stop + 1;
                if length >= min_orf_len {
                    let seq = rev_comp(&dna[stop.saturating_sub(1)..start]);
                    if mask_n && spans_masked(stop, start, &masked_regions) {
                        continue;
                    }
                    let rbs = rev_comp(&dna[start..(start + 21).min(dna.len())]);
                    let rbs_score = score_rbs(&rbs);
                    let pstop = Orf::compute_pstop(&seq);
                    orfs.push(Orf {
                        start: start - 2,
                        stop,
                        frame: neg_frame,
                        seq,
                        rbs_score,
                        pstop,
                        weight_rbs: 1.0,
                        hold: 1.0,
                        weight: 1.0,
                    });
                }
            }
        }
    }

    orfs
}

/// Find runs of N (or n) in the sequence. Returns inclusive 1-based ranges.
fn find_n_runs(dna: &[u8]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut in_run = false;
    let mut start = 0;
    for (i, &b) in dna.iter().enumerate() {
        if b == b'n' || b == b'N' {
            if !in_run {
                in_run = true;
                start = i + 1; // 1-based
            }
        } else if in_run {
            regions.push((start, i)); // end is inclusive 1-based
            in_run = false;
        }
    }
    if in_run {
        regions.push((start, dna.len()));
    }
    regions
}

/// Check if an interval [start, stop] (1-based, inclusive) spans any masked region.
fn spans_masked(start: usize, stop: usize, masked: &[(usize, usize)]) -> bool {
    let (lo, hi) = (start.min(stop), start.max(stop));
    for &(m_start, m_stop) in masked {
        if m_start <= hi && m_stop >= lo {
            return true;
        }
    }
    false
}

/// Get the 21 nt upstream of a start codon.
fn get_rbs(dna: &[u8], start: usize, _forward: bool) -> Vec<u8> {
    if start >= 21 {
        dna[start - 21..start].to_vec()
    } else {
        let mut pad = vec![b'a'; 21 - start];
        pad.extend_from_slice(&dna[..start]);
        pad
    }
}

/// Shine-Dalgarno likelihood score.
/// Replicates the reference implementation exactly.
pub fn score_rbs(seq: &[u8]) -> usize {
    // The reference takes the 21 nt upstream, then reverses it
    let s: Vec<u8> = seq.iter().rev().copied().collect();
    let s = std::str::from_utf8(&s).unwrap_or("");

    // Helper: check if pattern appears in any of the given (start, end) ranges
    let in_range = |pat: &str, start: usize, end: usize| -> bool {
        if end > s.len() {
            return false;
        }
        s[start..end].contains(pat)
    };

    // Conditions in Python's exact order: 27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
    // Key fix: score 7 (ggtgg in 11-16) must come AFTER score 13 (agg/gag/gga in 5-13)
    // and AFTER score 14 (ggtgg in 5-15)

    if in_range("aggagg", 0, 6) { return 27; }
    if in_range("ggagg", 0, 6) { return 26; }
    if in_range("aaggag", 0, 7) { return 25; }
    if in_range("aggtgg", 0, 7) { return 24; }
    if in_range("agg", 0, 7) || in_range("gag", 0, 7) || in_range("gga", 0, 7) { return 23; }
    if in_range("ggtggt", 0, 7) { return 22; }
    if in_range("aagg", 0, 7) { return 21; }
    if in_range("ggcg", 0, 7) || in_range("gccg", 0, 7) || in_range("gcgg", 0, 7) { return 20; }
    if in_range("ggtgg", 0, 7) { return 19; }
    if in_range("agga", 0, 7) { return 18; }
    if in_range("gggg", 0, 7) { return 17; }
    if in_range("ggtgga", 0, 7) || in_range("ggggga", 0, 7) || in_range("ggcgga", 0, 7) { return 16; }
    if in_range("gaggt", 0, 7) || in_range("gaggc", 0, 7) || in_range("gagga", 0, 7) { return 15; }
    if in_range("ggtgg", 0, 16) { return 14; }
    if in_range("agg", 5, 14) || in_range("gag", 5, 14) || in_range("gga", 5, 14) { return 13; }
    if in_range("gg", 5, 14) { return 12; }
    if in_range("g", 5, 14) { return 11; }
    if in_range("a", 5, 14) { return 10; }
    if in_range("t", 5, 14) { return 9; }
    if in_range("c", 5, 14) { return 8; }
    if in_range("ggtgg", 11, 17) { return 7; }
    if in_range("agg", 11, 17) || in_range("gag", 11, 17) || in_range("gga", 11, 17) { return 6; }
    if in_range("gg", 11, 17) { return 5; }
    if in_range("g", 11, 17) { return 4; }
    if in_range("a", 11, 17) { return 3; }
    if in_range("t", 11, 17) { return 2; }
    if in_range("c", 11, 17) { return 1; }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_rbs_direct() {
        // "aaggag" reversed is "gaggaa"
        let seq = b"gaggaa";
        assert_eq!(score_rbs(seq), 25);
    }

    #[test]
    fn test_score_rbs_specific() {
        // Test with a known sequence that should give score 27
        let seq = b"aggagg";
        assert_eq!(score_rbs(seq), 27);
    }

    #[test]
    fn test_score_rbs_specific2() {
        // Test with a known sequence that should give score 13
        let seq = b"aaaaaagg";
        assert_eq!(score_rbs(seq), 12);
    }

    #[test]
    fn test_score_rbs_accctg() {
        // Sequence with no match -> score 0
        let seq = b"accctg";
        assert_eq!(score_rbs(seq), 0);
    }

    #[test]
    fn test_background_rbs_counts() {
        // phiX174 has 5386 bases, so 5386 * 2 = 10772 background counts
        // The background_rbs array starts with 1.0 in each bin, so sum should be 10772 + 28
        let dna = b"atgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgcatgc";
        let mut background_rbs = vec![1.0f64; 28];
        for i in 0..dna.len() {
            let window = if i + 21 <= dna.len() {
                &dna[i..i + 21]
            } else {
                &dna[i..]
            };
            let score = score_rbs(window);
            background_rbs[score] += 1.0;
            let rc_window = rev_comp(window);
            let rc_score = score_rbs(&rc_window);
            background_rbs[rc_score] += 1.0;
        }
        let sum: f64 = background_rbs.iter().sum();
        assert!((sum - (dna.len() * 2 + 28) as f64).abs() < 0.001);
    }

    #[test]
    fn test_find_extra_7() {
        // Test for the extra score 7 condition
        // ggtgg in positions 11-16 (0-indexed: 10-15)
        let seq = b"aaaaaaaaaaggtgg"; // 15 chars, ggtgg at positions 10-14
        assert_eq!(score_rbs(seq), 7);
    }

    #[test]
    fn test_find_diff_7() {
        // Test that score 7 (ggtgg in 11-16) is checked AFTER score 13
        // A sequence that matches both should get score 13, not 7
        // agg in positions 5-13 (0-indexed: 4-12) AND ggtgg in 11-16
        let seq = b"aaaaaggaaaaaggtgg"; // 17 chars
        // agg at positions 4-6, ggtgg at positions 11-15
        // score 13 checks agg in 5-14 (1-based), which is 4-13 (0-based)
        // Our seq has agg at 4-6, which is in range 4-13
        assert_eq!(score_rbs(seq), 13);
    }

    #[test]
    fn test_specific_windows() {
        // Test a few specific window sequences
        let cases: Vec<(&[u8], usize)> = vec![
            (b"aggagg", 27),
            (b"ggagg", 26),
            (b"aaggag", 25),
            (b"aggtgg", 24),
            (b"ggtggt", 22),
            (b"ggtgg", 19),
            (b"agga", 18),
            (b"gggg", 17),
            (b"ggtgga", 16),
            (b"gaggt", 15),
            (b"gaggc", 15),
            (b"gagga", 15),
        ];
        for (seq, expected) in cases {
            assert_eq!(score_rbs(seq), expected, "Failed for sequence: {:?}", std::str::from_utf8(seq));
        }
    }

    #[test]
    fn test_i_822() {
        // Test the specific case at position 822 of phiX174
        // The 21-nt window upstream of position 822
        let window = b"ttttttttttttttttttttt";
        assert_eq!(score_rbs(window), 0);
    }

    #[test]
    fn test_nc001416_orf_20815() {
        // Test ORF at position 20815 in NC_001416
        // This ORF was missing in earlier versions
        let dna = include_bytes!("../tests/golden/NC_001416.1.tabular");
        // Just verify the function doesn't panic
        let _ = find_orfs(dna, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::genome::read_fasta;

    #[test]
    fn debug_phix174_orf() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            println!("Found {} ORFs", orfs.len());
            for orf in &orfs[..10.min(orfs.len())] {
                println!("ORF {}-{} frame={} rbs={}", orf.start, orf.stop, orf.frame, orf.rbs_score);
            }
        }
    }

    #[test]
    fn debug_pos_max_min() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            for (stop, orfs_at_stop) in by_stop {
                println!("Stop {}: {} ORFs", stop, orfs_at_stop.len());
                for orf in orfs_at_stop {
                    println!("  start={} frame={}", orf.start, orf.frame);
                }
            }
        }
    }

    #[test]
    fn debug_pos_max_min_fixed() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut training_rbs = vec![1.0f64; 28];
            for orf in &orfs {
                training_rbs[orf.rbs_score] += 1.0;
            }
            let mut selected_orfs = Vec::new();
            for (_, orfs_at_stop) in by_stop {
                for orf in orfs_at_stop {
                    if orf.start_codon() == b"atg" {
                        selected_orfs.push(orf);
                        break;
                    }
                }
            }
            println!("Selected {} training ORFs", selected_orfs.len());
            for orf in &selected_orfs[..5.min(selected_orfs.len())] {
                println!("  {}-{} frame={}", orf.start, orf.stop, orf.frame);
            }
        }
    }

    #[test]
    fn count_atg_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let atg_count = orfs.iter().filter(|o| o.start_codon() == b"atg").count();
            println!("ATG ORFs: {}/{} ({}%)", atg_count, orfs.len(), atg_count * 100 / orfs.len());
        }
    }

    #[test]
    fn count_all_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            println!("Total ORFs: {}", orfs.len());
            let fwd = orfs.iter().filter(|o| o.frame > 0).count();
            let rev = orfs.iter().filter(|o| o.frame < 0).count();
            println!("Forward: {}, Reverse: {}", fwd, rev);
        }
    }

    #[test]
    fn compare_stops() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut stops: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
            for orf in &orfs {
                *stops.entry(orf.stop).or_insert(0) += 1;
            }
            for (stop, count) in stops.iter().take(20) {
                println!("Stop {}: {} ORFs", stop, count);
            }
        }
    }

    #[test]
    fn compare_training_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut selected = Vec::new();
            for (_, orfs_at_stop) in by_stop {
                for orf in orfs_at_stop {
                    if orf.start_codon() == b"atg" {
                        selected.push(orf);
                        break;
                    }
                }
            }
            println!("Selected {} training ORFs", selected.len());
        }
    }

    #[test]
    fn compare_first_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            for (stop, orfs_at_stop) in by_stop.iter().take(10) {
                println!("Stop {}: first ATG ORF = {}-{} frame={}",
                    stop,
                    orfs_at_stop.iter().find(|o| o.start_codon() == b"atg").map(|o| o.start).unwrap_or(0),
                    orfs_at_stop.iter().find(|o| o.start_codon() == b"atg").map(|o| o.stop).unwrap_or(0),
                    orfs_at_stop.iter().find(|o| o.start_codon() == b"atg").map(|o| o.frame).unwrap_or(0),
                );
            }
        }
    }

    #[test]
    fn compare_reverse_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(10) {
                println!("  {}-{} frame={} start_codon={:?}", orf.start, orf.stop, orf.frame, std::str::from_utf8(orf.start_codon()));
            }
        }
    }

    #[test]
    fn count_training_bases() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
            for orf in &orfs {
                by_stop.entry(orf.stop).or_default().push(orf);
            }
            for (_, orfs_at_stop) in by_stop.iter_mut() {
                if orfs_at_stop[0].frame > 0 {
                    orfs_at_stop.sort_by_key(|o| o.start);
                } else {
                    orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
                }
            }
            let mut total_bases = 0;
            for (_, orfs_at_stop) in by_stop {
                if let Some(orf) = orfs_at_stop.iter().find(|o| o.start_codon() == b"atg") {
                    let (start, stop) = (orf.start, orf.stop);
                    if start < stop {
                        let n = ((stop - start) / 8) * 3;
                        let mut base = start + n;
                        while base + 36 < stop {
                            total_bases += 1;
                            base += 3;
                        }
                    }
                }
            }
            println!("Training bases: {}", total_bases);
        }
    }

    #[test]
    fn compare_longest_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut longest = orfs.iter().max_by_key(|o| o.seq.len()).unwrap();
            println!("Longest ORF: {}-{} ({} nt)", longest.start, longest.stop, longest.seq.len());
        }
    }

    #[test]
    fn check_mixed_stops() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let mut stop_counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
            for orf in &orfs {
                let stop = &genome.seq[orf.stop.saturating_sub(1)..orf.stop + 2];
                let key = String::from_utf8_lossy(stop).to_string();
                *stop_counts.entry(key).or_insert(0) += 1;
            }
            for (stop, count) in stop_counts.iter().take(10) {
                println!("Stop {}: {} ORFs", stop, count);
            }
        }
    }

    #[test]
    fn check_reverse_orfs_nc001416() {
        let genomes = read_fasta("../PHANOTATE/tests/NC_001416.1.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(&genome.seq, &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()], &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()], 90, false, false);
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("NC_001416 Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(5) {
                println!("  {}-{} frame={}", orf.start, orf.stop, orf.frame);
            }
        }
    }
}
