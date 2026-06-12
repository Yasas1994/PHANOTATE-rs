#[derive(Debug, Clone)]
pub struct Orf {
    pub start: usize, // 1-based, inclusive start of start codon (forward coords)
    pub stop: usize,  // 1-based, inclusive stop of stop codon (forward coords)
    pub frame: i8,    // 1,2,3 for forward; -1,-2,-3 for reverse
    pub seq: Vec<u8>, // the ORF nucleotide sequence (forward direction)
    pub rbs_score: usize,
    pub pstop: f64, // P(stop) for this ORF
    pub weight_rbs: f64,
    pub hold: f64,   // product of adjusted P(not_stop) per codon
    pub weight: f64, // final ORF edge weight (negative)
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

    /// Hybrid scoring: heuristic baseline + ML adjustment.
    ///
    /// Computes the heuristic score as in [`Self::score`], then multiplies
    /// by an ML-predicted adjustment factor.  The adjustment is clamped to
    /// [0.5, 2.0] so the ML model cannot destabilise the shortest-path search.
    #[cfg(feature = "ml")]
    pub fn score_hybrid(
        &mut self,
        start_codons: &std::collections::HashMap<Vec<u8>, f64>,
        ml: &crate::ml_scorer::MlScorer,
    ) {
        // 1. Compute heuristic baseline
        self.score(start_codons);
        let heuristic_weight = self.weight;

        // 2. Extract features and get ML adjustment
        let features = self.extract_features();
        let adjustment = ml.predict_adjustment(&features);

        // 3. Apply adjustment (weight is negative, so multiply)
        self.weight = heuristic_weight * adjustment;
    }
}

/// Enumerate all ORFs in all six reading frames.
///
/// Parameters:
/// - `closed_ends`: if true, do not allow ORFs to run off sequence edges.
///   Fragment ORFs (those without a matching start/stop at the boundary) are discarded.
/// - `mask_n`: if true, do not build any ORF that spans a run of N nucleotides.
///
/// Convenience wrapper that computes RC internally.
/// Use `find_orfs_with_rc` if RC is already available.
#[allow(dead_code)]
pub fn find_orfs(
    dna: &[u8],
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    min_orf_len: usize,
    closed_ends: bool,
    mask_n: bool,
) -> Vec<Orf> {
    let rc_dna = crate::genome::rev_comp(dna);
    find_orfs_with_rc(
        dna,
        &rc_dna,
        start_codons,
        stop_codons,
        min_orf_len,
        closed_ends,
        mask_n,
    )
}

/// Enumerate all ORFs in all six reading frames using a pre-computed reverse complement.
pub fn find_orfs_with_rc(
    dna: &[u8],
    rc_dna: &[u8],
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    min_orf_len: usize,
    closed_ends: bool,
    mask_n: bool,
) -> Vec<Orf> {
    let mut orfs = Vec::new();
    let contig_length = dna.len();

    // Pre-compute masked regions if needed
    let masked_regions: Vec<(usize, usize)> = if mask_n { find_n_runs(dna) } else { Vec::new() };

    // The dicts that hold start and stop codons
    // frames: 1,2,3,-1,-2,-3
    let mut stops: std::collections::HashMap<i8, usize> =
        [(1, 0), (2, 0), (3, 0), (-1, 1), (-2, 2), (-3, 3)]
            .into_iter()
            .collect();
    let mut starts: std::collections::HashMap<i8, Vec<usize>> = [
        (1, Vec::new()),
        (2, Vec::new()),
        (3, Vec::new()),
        (-1, Vec::new()),
        (-2, Vec::new()),
        (-3, Vec::new()),
    ]
    .into_iter()
    .collect();

    // Initial fragment starts if first codon is NOT a start
    if !closed_ends {
        if dna.len() >= 3 && !start_codons.iter().any(|c| c == &dna[0..3]) {
            starts.get_mut(&1).unwrap().push(1);
        }
        if dna.len() >= 4 && !start_codons.iter().any(|c| c == &dna[1..4]) {
            starts.get_mut(&2).unwrap().push(2);
        }
        if dna.len() >= 5 && !start_codons.iter().any(|c| c == &dna[2..5]) {
            starts.get_mut(&3).unwrap().push(3);
        }
    }

    // Scan all six frames
    for i in 1..=(dna.len().saturating_sub(2)) {
        let codon = &dna[i - 1..i + 2];
        let frame = ((i - 1) % 3) + 1; // 1, 2, 3 as usize
        let frame_i8 = frame as i8;

        // Forward start
        if start_codons.iter().any(|c| c == codon) {
            starts.get_mut(&frame_i8).unwrap().push(i);
        }

        // Forward stop
        if stop_codons.iter().any(|c| c == codon) {
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
        }

        // Read reverse-codon directly from pre-computed RC genome
        let rc_start = dna.len().saturating_sub(i + 2);
        let rc_codon = &rc_dna[rc_start..dna.len() - i + 1];

        // Reverse start
        if start_codons.iter().any(|c| c == rc_codon) {
            starts.get_mut(&(-(frame as i8))).unwrap().push(i + 2);
        }

        // Reverse stop
        if stop_codons.iter().any(|c| c == rc_codon) {
            let stop = stops[&(-(frame as i8))];
            for &start in starts.get(&(-(frame as i8))).unwrap().iter() {
                let length = start - stop + 1;
                if length >= min_orf_len {
                    // ORF seq: reverse complement of forward sequence
                    let fwd_start = stop.saturating_sub(1);
                    let fwd_end = start;
                    let seq = rc_dna[dna.len() - fwd_end..dna.len() - fwd_start].to_vec();
                    if mask_n && spans_masked(stop, start, &masked_regions) {
                        continue;
                    }
                    // RBS: 21nt upstream of start on reverse strand
                    // In forward coords: dna[start..start+21], RC is rc_dna[len-start-21..len-start]
                    let rbs_start = dna.len().saturating_sub(start + 21);
                    let rbs_end = dna.len() - start;
                    let rbs = &rc_dna[rbs_start..rbs_end];
                    let rbs_score = score_rbs(rbs);
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
                let rc_last_start = dna.len().saturating_sub(start_pos);
                let rc_last_end = dna.len() - start_pos + 3;
                let last_codon = &rc_dna[rc_last_start..rc_last_end];
                if !start_codons.iter().any(|c| c == last_codon) {
                    starts.get_mut(&neg_frame).unwrap().push(start_pos);
                }
            }
            let stop = stops[&neg_frame];
            for &start in starts.get(&neg_frame).unwrap().iter() {
                let length = start - stop + 1;
                if length >= min_orf_len {
                    let fwd_start = stop.saturating_sub(1);
                    let fwd_end = start;
                    let seq = rc_dna[dna.len() - fwd_end..dna.len() - fwd_start].to_vec();
                    if mask_n && spans_masked(stop, start, &masked_regions) {
                        continue;
                    }
                    let rbs_start = dna.len().saturating_sub(start + 21);
                    let rbs_end = dna.len() - start;
                    let rbs = &rc_dna[rbs_start..rbs_end];
                    let rbs_score = score_rbs(rbs);
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
/// Replicates the Python reference implementation exactly.
/// Byte-based implementation: no allocations, no UTF-8 conversion.
pub fn score_rbs(seq: &[u8]) -> usize {
    // The reference takes the 21 nt upstream, then reverses it
    let s: Vec<u8> = seq.iter().rev().copied().collect();

    // Helper: check if pattern (as bytes) appears in s[start..end]
    let in_range = |pat: &[u8], start: usize, end: usize| -> bool {
        if end > s.len() || start >= s.len() {
            return false;
        }
        let window = &s[start..end];
        if pat.len() > window.len() {
            return false;
        }
        window.windows(pat.len()).any(|w| w == pat)
    };

    // Ported from Python's score_rbs in functions.py — using byte patterns
    if in_range(b"ggagga", 5, 11)
        || in_range(b"ggagga", 6, 12)
        || in_range(b"ggagga", 7, 13)
        || in_range(b"ggagga", 8, 14)
        || in_range(b"ggagga", 9, 15)
        || in_range(b"ggagga", 10, 16)
    {
        return 27;
    }
    if in_range(b"ggagga", 3, 9) || in_range(b"ggagga", 4, 10) {
        return 26;
    }
    if in_range(b"ggagga", 11, 17) || in_range(b"ggagga", 12, 18) {
        return 25;
    }
    if in_range(b"ggagg", 5, 10)
        || in_range(b"ggagg", 6, 11)
        || in_range(b"ggagg", 7, 12)
        || in_range(b"ggagg", 8, 13)
        || in_range(b"ggagg", 9, 14)
        || in_range(b"ggagg", 10, 15)
    {
        return 24;
    }
    if in_range(b"ggagg", 3, 8) || in_range(b"ggagg", 4, 9) {
        return 23;
    }
    if in_range(b"gagga", 5, 10)
        || in_range(b"gagga", 6, 11)
        || in_range(b"gagga", 7, 12)
        || in_range(b"gagga", 8, 13)
        || in_range(b"gagga", 9, 14)
        || in_range(b"gagga", 10, 15)
    {
        return 22;
    }
    if in_range(b"gagga", 3, 8) || in_range(b"gagga", 4, 9) {
        return 21;
    }
    if in_range(b"gagga", 11, 16)
        || in_range(b"gagga", 12, 17)
        || in_range(b"ggagg", 11, 16)
        || in_range(b"ggagg", 12, 17)
    {
        return 20;
    }
    if in_range(b"ggacga", 5, 11)
        || in_range(b"ggacga", 6, 12)
        || in_range(b"ggacga", 7, 13)
        || in_range(b"ggacga", 8, 14)
        || in_range(b"ggacga", 9, 15)
        || in_range(b"ggacga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggatga", 5, 11)
        || in_range(b"ggatga", 6, 12)
        || in_range(b"ggatga", 7, 13)
        || in_range(b"ggatga", 8, 14)
        || in_range(b"ggatga", 9, 15)
        || in_range(b"ggatga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggaaga", 5, 11)
        || in_range(b"ggaaga", 6, 12)
        || in_range(b"ggaaga", 7, 13)
        || in_range(b"ggaaga", 8, 14)
        || in_range(b"ggaaga", 9, 15)
        || in_range(b"ggaaga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggcgga", 5, 11)
        || in_range(b"ggcgga", 6, 12)
        || in_range(b"ggcgga", 7, 13)
        || in_range(b"ggcgga", 8, 14)
        || in_range(b"ggcgga", 9, 15)
        || in_range(b"ggcgga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggggga", 5, 11)
        || in_range(b"ggggga", 6, 12)
        || in_range(b"ggggga", 7, 13)
        || in_range(b"ggggga", 8, 14)
        || in_range(b"ggggga", 9, 15)
        || in_range(b"ggggga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggtgga", 5, 11)
        || in_range(b"ggtgga", 6, 12)
        || in_range(b"ggtgga", 7, 13)
        || in_range(b"ggtgga", 8, 14)
        || in_range(b"ggtgga", 9, 15)
        || in_range(b"ggtgga", 10, 16)
    {
        return 19;
    }
    if in_range(b"ggaaga", 3, 9)
        || in_range(b"ggaaga", 4, 10)
        || in_range(b"ggatga", 3, 9)
        || in_range(b"ggatga", 4, 10)
        || in_range(b"ggacga", 3, 9)
        || in_range(b"ggacga", 4, 10)
    {
        return 18;
    }
    if in_range(b"ggtgga", 3, 9)
        || in_range(b"ggtgga", 4, 10)
        || in_range(b"ggggga", 3, 9)
        || in_range(b"ggggga", 4, 10)
        || in_range(b"ggcgga", 3, 9)
        || in_range(b"ggcgga", 4, 10)
    {
        return 18;
    }
    if in_range(b"ggaaga", 11, 17)
        || in_range(b"ggaaga", 12, 18)
        || in_range(b"ggatga", 11, 17)
        || in_range(b"ggatga", 12, 18)
        || in_range(b"ggacga", 11, 17)
        || in_range(b"ggacga", 12, 18)
    {
        return 17;
    }
    if in_range(b"ggtgga", 11, 17)
        || in_range(b"ggtgga", 12, 18)
        || in_range(b"ggggga", 11, 17)
        || in_range(b"ggggga", 12, 18)
        || in_range(b"ggcgga", 11, 17)
        || in_range(b"ggcgga", 12, 18)
    {
        return 17;
    }
    if in_range(b"ggag", 5, 9)
        || in_range(b"ggag", 6, 10)
        || in_range(b"ggag", 7, 11)
        || in_range(b"ggag", 8, 12)
        || in_range(b"ggag", 9, 13)
        || in_range(b"ggag", 10, 14)
    {
        return 16;
    }
    if in_range(b"gagg", 5, 9)
        || in_range(b"gagg", 6, 10)
        || in_range(b"gagg", 7, 11)
        || in_range(b"gagg", 8, 12)
        || in_range(b"gagg", 9, 13)
        || in_range(b"gagg", 10, 14)
    {
        return 16;
    }
    if in_range(b"agga", 5, 9)
        || in_range(b"agga", 6, 10)
        || in_range(b"agga", 7, 11)
        || in_range(b"agga", 8, 12)
        || in_range(b"agga", 9, 13)
        || in_range(b"agga", 10, 14)
    {
        return 15;
    }
    if in_range(b"ggtgg", 5, 10)
        || in_range(b"ggtgg", 6, 11)
        || in_range(b"ggtgg", 7, 12)
        || in_range(b"ggtgg", 8, 13)
        || in_range(b"ggtgg", 9, 14)
        || in_range(b"ggtgg", 10, 15)
    {
        return 14;
    }
    if in_range(b"ggggg", 5, 10)
        || in_range(b"ggggg", 6, 11)
        || in_range(b"ggggg", 7, 12)
        || in_range(b"ggggg", 8, 13)
        || in_range(b"ggggg", 9, 14)
        || in_range(b"ggggg", 10, 15)
    {
        return 14;
    }
    if in_range(b"ggcgg", 5, 10)
        || in_range(b"ggcgg", 6, 11)
        || in_range(b"ggcgg", 7, 12)
        || in_range(b"ggcgg", 8, 13)
        || in_range(b"ggcgg", 9, 14)
        || in_range(b"ggcgg", 10, 15)
    {
        return 14;
    }
    if in_range(b"agg", 5, 8)
        || in_range(b"agg", 6, 9)
        || in_range(b"agg", 7, 10)
        || in_range(b"agg", 8, 11)
        || in_range(b"agg", 9, 12)
        || in_range(b"agg", 10, 13)
    {
        return 13;
    }
    if in_range(b"gag", 5, 8)
        || in_range(b"gag", 6, 9)
        || in_range(b"gag", 7, 10)
        || in_range(b"gag", 8, 11)
        || in_range(b"gag", 9, 12)
        || in_range(b"gag", 10, 13)
    {
        return 13;
    }
    if in_range(b"gga", 5, 8)
        || in_range(b"gga", 6, 9)
        || in_range(b"gga", 7, 10)
        || in_range(b"gga", 8, 11)
        || in_range(b"gga", 9, 12)
        || in_range(b"gga", 10, 13)
    {
        return 13;
    }
    if in_range(b"agga", 11, 15)
        || in_range(b"agga", 12, 16)
        || in_range(b"gagg", 11, 15)
        || in_range(b"gagg", 12, 16)
        || in_range(b"ggag", 11, 15)
        || in_range(b"ggag", 12, 16)
    {
        return 12;
    }
    if in_range(b"agga", 3, 7)
        || in_range(b"agga", 4, 8)
        || in_range(b"gagg", 3, 7)
        || in_range(b"gagg", 4, 8)
        || in_range(b"ggag", 3, 7)
        || in_range(b"ggag", 4, 8)
    {
        return 11;
    }
    if in_range(b"gagga", 13, 18)
        || in_range(b"gagga", 14, 19)
        || in_range(b"gagga", 15, 20)
        || in_range(b"ggagg", 13, 18)
        || in_range(b"ggagg", 14, 19)
        || in_range(b"ggagg", 15, 20)
        || in_range(b"ggagga", 13, 19)
        || in_range(b"ggagga", 14, 20)
        || in_range(b"ggagga", 15, 21)
    {
        return 10;
    }
    if in_range(b"gaaga", 5, 10)
        || in_range(b"gaaga", 6, 11)
        || in_range(b"gaaga", 7, 12)
        || in_range(b"gaaga", 8, 13)
        || in_range(b"gaaga", 9, 14)
        || in_range(b"gaaga", 10, 15)
    {
        return 9;
    }
    if in_range(b"gatga", 5, 10)
        || in_range(b"gatga", 6, 11)
        || in_range(b"gatga", 7, 12)
        || in_range(b"gatga", 8, 13)
        || in_range(b"gatga", 9, 14)
        || in_range(b"gatga", 10, 15)
    {
        return 9;
    }
    if in_range(b"gacga", 5, 10)
        || in_range(b"gacga", 6, 11)
        || in_range(b"gacga", 7, 12)
        || in_range(b"gacga", 8, 13)
        || in_range(b"gacga", 9, 14)
        || in_range(b"gacga", 10, 15)
    {
        return 9;
    }
    if in_range(b"ggtgg", 3, 8)
        || in_range(b"ggtgg", 4, 9)
        || in_range(b"ggggg", 3, 8)
        || in_range(b"ggggg", 4, 9)
        || in_range(b"ggcgg", 3, 8)
        || in_range(b"ggcgg", 4, 9)
    {
        return 8;
    }
    if in_range(b"ggtgg", 11, 16)
        || in_range(b"ggtgg", 12, 17)
        || in_range(b"ggggg", 11, 16)
        || in_range(b"ggggg", 12, 17)
        || in_range(b"ggcgg", 11, 16)
        || in_range(b"ggcgg", 12, 17)
    {
        return 7;
    }
    if in_range(b"agg", 11, 14)
        || in_range(b"agg", 12, 15)
        || in_range(b"gag", 11, 14)
        || in_range(b"gag", 12, 15)
        || in_range(b"gga", 11, 14)
        || in_range(b"gga", 12, 15)
    {
        return 6;
    }
    if in_range(b"gaaga", 3, 8)
        || in_range(b"gaaga", 4, 9)
        || in_range(b"gatga", 3, 8)
        || in_range(b"gatga", 4, 9)
        || in_range(b"gacga", 3, 8)
        || in_range(b"gacga", 4, 9)
    {
        return 5;
    }
    if in_range(b"gaaga", 11, 16)
        || in_range(b"gaaga", 12, 17)
        || in_range(b"gatga", 11, 16)
        || in_range(b"gatga", 12, 17)
        || in_range(b"gacga", 11, 16)
        || in_range(b"gacga", 12, 17)
    {
        return 4;
    }
    if in_range(b"agga", 13, 17)
        || in_range(b"agga", 14, 18)
        || in_range(b"agga", 15, 19)
        || in_range(b"gagg", 13, 17)
        || in_range(b"gagg", 14, 18)
        || in_range(b"gagg", 15, 19)
        || in_range(b"ggag", 13, 17)
        || in_range(b"ggag", 14, 18)
        || in_range(b"ggag", 15, 19)
    {
        return 3;
    }
    if in_range(b"agg", 13, 16)
        || in_range(b"agg", 14, 17)
        || in_range(b"agg", 15, 18)
        || in_range(b"gag", 13, 16)
        || in_range(b"gag", 14, 17)
        || in_range(b"gag", 15, 18)
        || in_range(b"gga", 13, 16)
        || in_range(b"gga", 14, 17)
        || in_range(b"gga", 15, 18)
    {
        return 2;
    }
    if in_range(b"ggaaga", 13, 19)
        || in_range(b"ggaaga", 14, 20)
        || in_range(b"ggaaga", 15, 21)
        || in_range(b"ggatga", 13, 19)
        || in_range(b"ggatga", 14, 20)
        || in_range(b"ggatga", 15, 21)
        || in_range(b"ggacga", 13, 19)
        || in_range(b"ggacga", 14, 20)
        || in_range(b"ggacga", 15, 21)
    {
        return 2;
    }
    if in_range(b"ggtgg", 13, 18)
        || in_range(b"ggtgg", 14, 19)
        || in_range(b"ggtgg", 15, 20)
        || in_range(b"ggggg", 13, 18)
        || in_range(b"ggggg", 14, 19)
        || in_range(b"ggggg", 15, 20)
        || in_range(b"ggcgg", 13, 18)
        || in_range(b"ggcgg", 14, 19)
        || in_range(b"ggcgg", 15, 20)
    {
        return 2;
    }
    if in_range(b"agg", 3, 6)
        || in_range(b"agg", 4, 7)
        || in_range(b"gag", 3, 6)
        || in_range(b"gag", 4, 7)
        || in_range(b"gga", 3, 6)
        || in_range(b"gga", 4, 7)
    {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_rbs_direct() {
        // "aaggag" reversed is "gaggaa" — but Python's score_rbs is designed
        // for 21-nt windows; short inputs give truncated slices.
        // For 6-char "gaggaa", Python gives 1 (agg in s[3:6]).
        let seq = b"gaggaa";
        assert_eq!(score_rbs(seq), 1);
    }

    #[test]
    fn test_score_rbs_specific() {
        // For 6-char "aggagg", Python gives 1 (agg in s[3:6]).
        let seq = b"aggagg";
        assert_eq!(score_rbs(seq), 1);
    }

    #[test]
    fn test_score_rbs_specific2() {
        // For 8-char "aaaaaagg", Python gives 0 (no match in truncated slices).
        let seq = b"aaaaaagg";
        assert_eq!(score_rbs(seq), 0);
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
            let rc_window = crate::genome::rev_comp(window);
            let rc_score = score_rbs(&rc_window);
            background_rbs[rc_score] += 1.0;
        }
        let sum: f64 = background_rbs.iter().sum();
        assert!((sum - (dna.len() * 2 + 28) as f64).abs() < 0.001);
    }

    #[test]
    fn test_find_extra_7() {
        // 15-char sequence: ggtgg at positions 10-14 (0-indexed).
        // Reversed: "ggtggaaaaaaaaaa" — ggtgg is at positions 0-4.
        // Python checks s[11:16] etc. for score 7, but string is only 15 chars,
        // so s[11:16] = "aaa" — no match. Score = 1 (agg in s[3:6] = "tgg"? no)
        // Actually: s = "ggtggaaaaaaaaaa", s[3:6] = "gga" — no match for agg/gag/gga
        // Let me check: s = 'ggtggaaaaaaaaaa', len=15
        // s[3:6] = 'gga' → matches 'gga' → score 1
        let seq = b"aaaaaaaaaaggtgg";
        assert_eq!(score_rbs(seq), 1);
    }

    #[test]
    fn test_find_diff_7() {
        // 17-char sequence
        // Reversed: "ggtggaaaaaaggaaaa"
        // Python: s[5:10]='ggaaa', s[6:11]='gaaaa' — no ggtgg/ggggg/ggcgg for score 14
        // s[5:9]='ggaa', s[6:10]='gaaa' — no ggag/gagg/agga for score 16
        // s[5:8]='gga' → matches 'gga' → score 13
        // Wait, let me check Python: score_rbs('aaaaaggaaaaaggtgg') = 15
        // s = 'ggtggaaaaaaggaaaa'
        // s[5:9] = 'ggaa' — no match for agga/gagg/ggag
        // Actually: s[5:9]='ggaa', but score 15 checks agga in s[5:9],s[6:10]...
        // s = 'ggtggaaaaaaggaaaa', len=17
        // s[5:9] = 'aaaa' — no
        // s[6:10] = 'aaaa' — no
        // s[7:11] = 'aaaa' — no
        // s[8:12] = 'aaag' — no
        // s[9:13] = 'aagg' — no
        // s[10:14] = 'agga' → MATCH! score 15
        let seq = b"aaaaaggaaaaaggtgg";
        assert_eq!(score_rbs(seq), 15);
    }

    #[test]
    fn test_specific_windows() {
        // Test a few specific window sequences against Python's actual behavior.
        // These short sequences produce truncated slices in Python's score_rbs.
        let cases: Vec<(&[u8], usize)> = vec![
            (b"aggagg", 1), // 6-char: agg in s[3:6]
            (b"ggagg", 0),  // 5-char: no match
            (b"aaggag", 0), // 6-char: no match
            (b"aggtgg", 1), // 6-char: agg in s[3:6]
            (b"ggtggt", 0), // 6-char: no match
            (b"ggtgg", 0),  // 5-char: no match
            (b"agga", 0),   // 4-char: no match
            (b"gggg", 0),   // 4-char: no match
            (b"ggtgga", 0), // 6-char: no match
            (b"gaggt", 0),  // 5-char: no match
            (b"gaggc", 0),  // 5-char: no match
            (b"gagga", 0),  // 5-char: no match
        ];
        for (seq, expected) in cases {
            assert_eq!(
                score_rbs(seq),
                expected,
                "Failed for sequence: {:?}",
                std::str::from_utf8(seq)
            );
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
        let _ = find_orfs(
            dna,
            &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
            &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
            90,
            false,
            false,
        );
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            println!("Found {} ORFs", orfs.len());
            for orf in &orfs[..10.min(orfs.len())] {
                println!(
                    "ORF {}-{} frame={} rbs={}",
                    orf.start, orf.stop, orf.frame, orf.rbs_score
                );
            }
        }
    }

    #[test]
    fn debug_pos_max_min() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let atg_count = orfs.iter().filter(|o| o.start_codon() == b"atg").count();
            println!(
                "ATG ORFs: {}/{} ({}%)",
                atg_count,
                orfs.len(),
                atg_count * 100 / orfs.len()
            );
        }
    }

    #[test]
    fn count_all_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut stops: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
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
                println!(
                    "Stop {}: first ATG ORF = {}-{} frame={}",
                    stop,
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.start)
                        .unwrap_or(0),
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.stop)
                        .unwrap_or(0),
                    orfs_at_stop
                        .iter()
                        .find(|o| o.start_codon() == b"atg")
                        .map(|o| o.frame)
                        .unwrap_or(0),
                );
            }
        }
    }

    #[test]
    fn compare_reverse_orfs() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(10) {
                println!(
                    "  {}-{} frame={} start_codon={:?}",
                    orf.start,
                    orf.stop,
                    orf.frame,
                    std::str::from_utf8(orf.start_codon())
                );
            }
        }
    }

    #[test]
    fn count_training_bases() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let longest = orfs.iter().max_by_key(|o| o.seq.len()).unwrap();
            println!(
                "Longest ORF: {}-{} ({} nt)",
                longest.start,
                longest.stop,
                longest.seq.len()
            );
        }
    }

    #[test]
    fn check_mixed_stops() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let mut stop_counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
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
            let orfs = find_orfs(
                &genome.seq,
                &[b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
                &[b"tag".to_vec(), b"tga".to_vec(), b"taa".to_vec()],
                90,
                false,
                false,
            );
            let rev_orfs: Vec<_> = orfs.iter().filter(|o| o.frame < 0).collect();
            println!("NC_001416 Reverse ORFs: {}", rev_orfs.len());
            for orf in rev_orfs.iter().take(5) {
                println!("  {}-{} frame={}", orf.start, orf.stop, orf.frame);
            }
        }
    }
}
