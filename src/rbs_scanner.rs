//! RBS motif scanning.
//!
//! Contains the legacy PHANOTATE pattern matcher and a Prodigal-style
//! Shine-Dalgarno consensus scanner with mismatch support.

/// Number of RBS score bins (0 = no motif, 1-27 = increasing SD signal).
pub const NUM_RBS_BINS: usize = 28;

/// Legacy PHANOTATE Shine-Dalgarno likelihood score.
/// Replicates the Python reference implementation exactly.
/// `seq` is the 21-nt upstream window (original orientation).  The function
/// reverses it internally, matching the original convention.
pub fn score_rbs_legacy(seq: &[u8]) -> usize {
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

/// Detect the matching Shine-Dalgarno motif in the upstream window.
///
/// Mirrors the priority order and position ranges of `score_rbs_legacy`. The returned
/// name is the motif as it appears in the original upstream window (i.e. the
/// reverse of the pattern checked in the reversed window `s`).
pub fn detect_rbs_motif_legacy(seq: &[u8]) -> Option<String> {
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

    // Check a pattern against one or more ranges and return its motif name.
    macro_rules! detect {
        ($pat:expr, $name:expr, $ranges:expr) => {
            if $ranges
                .iter()
                .any(|&(start, end)| in_range($pat, start, end))
            {
                return Some($name.to_string());
            }
        };
    }

    // Tiers are listed in the same priority order as `score_rbs_legacy`.
    detect!(
        b"ggagga",
        "AGGAGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(b"ggagga", "AGGAGG", &[(3, 9), (4, 10)]);
    detect!(b"ggagga", "AGGAGG", &[(11, 17), (12, 18)]);

    detect!(
        b"ggagg",
        "GGAGG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(b"ggagg", "GGAGG", &[(3, 8), (4, 9)]);

    detect!(
        b"gagga",
        "AGGAG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(b"gagga", "AGGAG", &[(3, 8), (4, 9)]);
    detect!(b"gagga", "AGGAG", &[(11, 16), (12, 17)]);
    detect!(b"ggagg", "GGAGG", &[(11, 16), (12, 17)]);

    detect!(
        b"ggacga",
        "AGCAGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(
        b"ggatga",
        "AGTAGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(
        b"ggaaga",
        "AGAAGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(
        b"ggcgga",
        "AGGCGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(
        b"ggggga",
        "AGGGGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );
    detect!(
        b"ggtgga",
        "AGGTGG",
        &[(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]
    );

    detect!(b"ggaaga", "AGAAGG", &[(3, 9), (4, 10)]);
    detect!(b"ggatga", "AGTAGG", &[(3, 9), (4, 10)]);
    detect!(b"ggacga", "AGCAGG", &[(3, 9), (4, 10)]);
    detect!(b"ggtgga", "AGGTGG", &[(3, 9), (4, 10)]);
    detect!(b"ggggga", "AGGGGG", &[(3, 9), (4, 10)]);
    detect!(b"ggcgga", "AGGCGG", &[(3, 9), (4, 10)]);

    detect!(b"ggaaga", "AGAAGG", &[(11, 17), (12, 18)]);
    detect!(b"ggatga", "AGTAGG", &[(11, 17), (12, 18)]);
    detect!(b"ggacga", "AGCAGG", &[(11, 17), (12, 18)]);
    detect!(b"ggtgga", "AGGTGG", &[(11, 17), (12, 18)]);
    detect!(b"ggggga", "AGGGGG", &[(11, 17), (12, 18)]);
    detect!(b"ggcgga", "AGGCGG", &[(11, 17), (12, 18)]);

    detect!(
        b"ggag",
        "GAGG",
        &[(5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14)]
    );
    detect!(
        b"gagg",
        "GGAG",
        &[(5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14)]
    );

    detect!(
        b"agga",
        "AGGA",
        &[(5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14)]
    );

    detect!(
        b"ggtgg",
        "GGTGG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(
        b"ggggg",
        "GGGGG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(
        b"ggcgg",
        "GGCGG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );

    detect!(
        b"agg",
        "GGA",
        &[(5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (10, 13)]
    );
    detect!(
        b"gag",
        "GAG",
        &[(5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (10, 13)]
    );
    detect!(
        b"gga",
        "AGG",
        &[(5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (10, 13)]
    );

    detect!(b"agga", "AGGA", &[(11, 15), (12, 16)]);
    detect!(b"gagg", "GGAG", &[(11, 15), (12, 16)]);
    detect!(b"ggag", "GAGG", &[(11, 15), (12, 16)]);

    detect!(b"agga", "AGGA", &[(3, 7), (4, 8)]);
    detect!(b"gagg", "GGAG", &[(3, 7), (4, 8)]);
    detect!(b"ggag", "GAGG", &[(3, 7), (4, 8)]);

    detect!(b"gagga", "AGGAG", &[(13, 18), (14, 19), (15, 20)]);
    detect!(b"ggagg", "GGAGG", &[(13, 18), (14, 19), (15, 20)]);
    detect!(b"ggagga", "AGGAGG", &[(13, 19), (14, 20), (15, 21)]);

    detect!(
        b"gaaga",
        "AGAAG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(
        b"gatga",
        "AGTAG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    detect!(
        b"gacga",
        "AGCAG",
        &[(5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );

    detect!(b"ggtgg", "GGTGG", &[(3, 8), (4, 9)]);
    detect!(b"ggggg", "GGGGG", &[(3, 8), (4, 9)]);
    detect!(b"ggcgg", "GGCGG", &[(3, 8), (4, 9)]);

    detect!(b"ggtgg", "GGTGG", &[(11, 16), (12, 17)]);
    detect!(b"ggggg", "GGGGG", &[(11, 16), (12, 17)]);
    detect!(b"ggcgg", "GGCGG", &[(11, 16), (12, 17)]);

    detect!(b"agg", "GGA", &[(11, 14), (12, 15)]);
    detect!(b"gag", "GAG", &[(11, 14), (12, 15)]);
    detect!(b"gga", "AGG", &[(11, 14), (12, 15)]);

    detect!(b"gaaga", "AGAAG", &[(3, 8), (4, 9)]);
    detect!(b"gatga", "AGTAG", &[(3, 8), (4, 9)]);
    detect!(b"gacga", "AGCAG", &[(3, 8), (4, 9)]);

    detect!(b"gaaga", "AGAAG", &[(11, 16), (12, 17)]);
    detect!(b"gatga", "AGTAG", &[(11, 16), (12, 17)]);
    detect!(b"gacga", "AGCAG", &[(11, 16), (12, 17)]);

    detect!(b"agga", "AGGA", &[(13, 17), (14, 18), (15, 19)]);
    detect!(b"gagg", "GGAG", &[(13, 17), (14, 18), (15, 19)]);
    detect!(b"ggag", "GAGG", &[(13, 17), (14, 18), (15, 19)]);

    detect!(b"agg", "GGA", &[(13, 16), (14, 17), (15, 18)]);
    detect!(b"gag", "GAG", &[(13, 16), (14, 17), (15, 18)]);
    detect!(b"gga", "AGG", &[(13, 16), (14, 17), (15, 18)]);
    detect!(b"ggaaga", "AGAAGG", &[(13, 19), (14, 20), (15, 21)]);
    detect!(b"ggatga", "AGTAGG", &[(13, 19), (14, 20), (15, 21)]);
    detect!(b"ggacga", "AGCAGG", &[(13, 19), (14, 20), (15, 21)]);
    detect!(b"ggtgg", "GGTGG", &[(13, 18), (14, 19), (15, 20)]);
    detect!(b"ggggg", "GGGGG", &[(13, 18), (14, 19), (15, 20)]);
    detect!(b"ggcgg", "GGCGG", &[(13, 18), (14, 19), (15, 20)]);

    detect!(b"agg", "GGA", &[(3, 6), (4, 7)]);
    detect!(b"gag", "GAG", &[(3, 6), (4, 7)]);
    detect!(b"gga", "AGG", &[(3, 6), (4, 7)]);

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_no_motif() {
        assert_eq!(score_rbs_legacy(b"aaaaaaaaaaaaaaaaaaaaa"), 0);
        assert_eq!(detect_rbs_motif_legacy(b"aaaaaaaaaaaaaaaaaaaaa"), None);
    }

    #[test]
    fn legacy_aggagg_detected() {
        // 21-nt window ending in AGGAGG 5-10 bp upstream
        let seq = b"aaaaaaaaaaaaggaggaaaa"; // reversed AGGAGG at positions 3-9
        assert!(score_rbs_legacy(seq) > 0);
        assert_eq!(detect_rbs_motif_legacy(seq), Some("AGGAGG".to_string()));
    }
}
