//! NCBI Translation Tables relevant to bacteriophages.
//!
//! All functions accept a lowercase nucleotide byte slice and return a protein
//! string where '*' denotes a stop codon and 'X' denotes an unknown/incomplete
//! codon.
//!
//! Tables included and their single-sentence rationale:
//!
//!  Table  1  Standard                    — baseline; some eukaryotic phages
//!  Table  4  Mold/Mycoplasma/Spiroplasma — Mycoplasma & Spiroplasma phages; TGA→Trp
//!  Table  6  Ciliate nuclear             — phages infecting Tetrahymena/Paramecium; TAA/TAG→Gln
//!  Table 11  Bacterial/Archaeal          — default for all bacteriophages
//!  Table 15  Blepharisma nuclear         — some Crassvirales; TAG→Gln
//!  Table 25  SR1/Gracilibacteria         — Gracilibacteria phages; TGA→Gly
//!
//! Differences from Table 11 are marked with // [DIFF] comments.

// ---------------------------------------------------------------------------
// Table 1 — The Standard Code
// ---------------------------------------------------------------------------
// Differences from Table 11:
//   Identical amino-acid assignments. Narrower start-codon set (ATG, TTG, CTG
//   vs. the broader set in Table 11). Included for completeness when annotating
//   eukaryotic viruses.
pub fn translate_table1(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" | b"tag" | b"tga" => '*',
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Table 4 — Mold, Protozoan, Coelenterate Mitochondrial +
//            Mycoplasma / Spiroplasma Code
// ---------------------------------------------------------------------------
// Differences from Table 11:
//   TGA → Trp  (instead of stop)
//
// Phage relevance:
//   All phages infecting Mycoplasma and Spiroplasma species use this table.
//   These are the genomes explicitly excluded from PHANOTATE's 2133-genome
//   benchmark. Using Table 11 on these genomes causes TGA codons in the middle
//   of real genes to be misread as stops, producing hundreds of spuriously
//   truncated CDSs.
pub fn translate_table4(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" | b"tag" => '*',
                b"tga" => 'W', // [DIFF] TGA is Trp, not a stop
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Table 6 — Ciliate, Dasycladacean and Hexamita Nuclear Code
// ---------------------------------------------------------------------------
// Differences from Table 11:
//   TAA → Gln  (instead of stop)
//   TAG → Gln  (instead of stop)
//   TGA remains the only stop codon.
//
// Phage relevance:
//   Phages that infect ciliates such as Tetrahymena and Paramecium. Rare in
//   typical phage surveys but present in metagenomic datasets from aquatic
//   environments where ciliates are abundant.
pub fn translate_table6(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" => 'Q', // [DIFF] TAA is Gln, not a stop
                b"tag" => 'Q', // [DIFF] TAG is Gln, not a stop
                b"tga" => '*',
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Table 11 — Bacterial, Archaeal and Plant Plastid Code
// ---------------------------------------------------------------------------
// This is the default table for all bacteriophages.
// Amino-acid assignments are identical to Table 1; the difference is a broader
// set of allowed start codons handled separately by the gene caller
// (TTG, CTG, ATT, ATC, ATA, ATG, GTG).
pub fn translate_table11(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" | b"tag" | b"tga" => '*',
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Table 15 — Blepharisma Nuclear Code
// ---------------------------------------------------------------------------
// Differences from Table 11:
//   TAG → Gln  (instead of stop)
//   TAA and TGA remain stops.
//
// Phage relevance:
//   Used by a subset of Crassvirales (crAss-like phages), a globally abundant
//   gut phage family. TAG reassignment has been confirmed by tRNA anticodon
//   analysis in several crAss-like genomes. Annotating these with Table 11
//   causes TAG-terminated ORFs to appear spuriously short.
pub fn translate_table15(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" | b"tga" => '*',
                b"tag" => 'Q', // [DIFF] TAG is Gln, not a stop
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Table 25 — Candidate Division SR1 and Gracilibacteria Code
// ---------------------------------------------------------------------------
// Differences from Table 11:
//   TGA → Gly  (instead of stop)
//   TAA and TAG remain stops.
//
// Phage relevance:
//   Phages infecting Gracilibacteria and SR1 bacteria, found primarily in oral
//   and gut microbiomes. TGA recoding to Gly is confirmed by tRNA anticodon
//   analysis. These genomes are increasingly appearing in human microbiome
//   phage surveys and are mis-annotated at scale when Table 11 is used.
pub fn translate_table25(seq: &[u8]) -> String {
    seq.chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon {
                b"ttt" | b"ttc" => 'F',
                b"tta" | b"ttg" | b"ctt" | b"ctc" | b"cta" | b"ctg" => 'L',
                b"att" | b"atc" | b"ata" => 'I',
                b"atg" => 'M',
                b"gtt" | b"gtc" | b"gta" | b"gtg" => 'V',
                b"tct" | b"tcc" | b"tca" | b"tcg" | b"agt" | b"agc" => 'S',
                b"cct" | b"ccc" | b"cca" | b"ccg" => 'P',
                b"act" | b"acc" | b"aca" | b"acg" => 'T',
                b"gct" | b"gcc" | b"gca" | b"gcg" => 'A',
                b"tat" | b"tac" => 'Y',
                b"taa" | b"tag" => '*',
                b"tga" => 'G', // [DIFF] TGA is Gly, not a stop
                b"cat" | b"cac" => 'H',
                b"caa" | b"cag" => 'Q',
                b"aat" | b"aac" => 'N',
                b"aaa" | b"aag" => 'K',
                b"gat" | b"gac" => 'D',
                b"gaa" | b"gag" => 'E',
                b"tgt" | b"tgc" => 'C',
                b"tgg" => 'W',
                b"cgt" | b"cgc" | b"cga" | b"cgg" | b"aga" | b"agg" => 'R',
                b"ggt" | b"ggc" | b"gga" | b"ggg" => 'G',
                _ => 'X',
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

/// Select a translation function by NCBI table number and translate `seq`.
///
/// Returns `Err` for any table number not implemented here. The caller should
/// validate the table number at CLI-parse time (before any pipeline work) using
/// `is_supported_table()` so that errors surface immediately.
pub fn translate(seq: &[u8], table: u8) -> Result<String, String> {
    match table {
        1  => Ok(translate_table1(seq)),
        4  => Ok(translate_table4(seq)),
        6  => Ok(translate_table6(seq)),
        11 => Ok(translate_table11(seq)),
        15 => Ok(translate_table15(seq)),
        25 => Ok(translate_table25(seq)),
        n  => Err(format!(
            "Translation table {} is not implemented. \
             Supported tables for phage annotation: 1, 4, 6, 11, 15, 25.",
            n
        )),
    }
}

/// Return true if `table` is supported by this module.
pub fn is_supported_table(table: u8) -> bool {
    matches!(table, 1 | 4 | 6 | 11 | 15 | 25)
}

/// Return the canonical NCBI name for a table number.
#[allow(dead_code)]
pub fn table_name(table: u8) -> &'static str {
    match table {
        1  => "Standard",
        4  => "Mold/Protozoan/Coelenterate Mitochondrial + Mycoplasma/Spiroplasma",
        6  => "Ciliate, Dasycladacean and Hexamita Nuclear",
        11 => "Bacterial, Archaeal and Plant Plastid",
        15 => "Blepharisma Nuclear",
        25 => "Candidate Division SR1 and Gracilibacteria",
        _  => "Unknown",
    }
}

/// Return the set of stop codons (lowercase) for a given table.
pub fn stop_codons(table: u8) -> &'static [&'static [u8]] {
    match table {
        1 | 11                => &[b"taa", b"tag", b"tga"],
        4                     => &[b"taa", b"tag"],           // TGA is Trp
        6                     => &[b"tga"],                   // TAA/TAG are Gln
        15                    => &[b"taa", b"tga"],           // TAG is Gln
        25                    => &[b"taa", b"tag"],           // TGA is Gly
        _                     => &[b"taa", b"tag", b"tga"],
    }
}

/// Return the set of start codons (lowercase) for a given table.
/// These are the codons that PHANOTATE should consider as valid ORF starts.
pub fn start_codons(table: u8) -> &'static [&'static [u8]] {
    match table {
        1  => &[b"atg", b"ttg", b"ctg"],
        4  => &[b"tta", b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg"],
        6  => &[b"atg"],
        11 => &[b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg"],
        15 => &[b"atg"],
        25 => &[b"ttg", b"atg", b"gtg"],
        _  => &[b"atg"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_atg() {
        assert_eq!(translate(b"atg", 1).unwrap(), "M");
        assert_eq!(translate(b"atg", 11).unwrap(), "M");
    }

    #[test]
    fn test_translate_stop_table1() {
        assert_eq!(translate(b"taa", 1).unwrap(), "*");
        assert_eq!(translate(b"tag", 1).unwrap(), "*");
        assert_eq!(translate(b"tga", 1).unwrap(), "*");
    }

    #[test]
    fn test_translate_stop_table11() {
        assert_eq!(translate(b"taa", 11).unwrap(), "*");
        assert_eq!(translate(b"tag", 11).unwrap(), "*");
        assert_eq!(translate(b"tga", 11).unwrap(), "*");
    }

    #[test]
    fn test_translate_table4_tga_is_trp() {
        assert_eq!(translate(b"tga", 4).unwrap(), "W");
        assert_eq!(translate(b"taa", 4).unwrap(), "*");
        assert_eq!(translate(b"tag", 4).unwrap(), "*");
    }

    #[test]
    fn test_translate_table6_taa_tag_are_gln() {
        assert_eq!(translate(b"taa", 6).unwrap(), "Q");
        assert_eq!(translate(b"tag", 6).unwrap(), "Q");
        assert_eq!(translate(b"tga", 6).unwrap(), "*");
    }

    #[test]
    fn test_translate_table15_tag_is_gln() {
        assert_eq!(translate(b"tag", 15).unwrap(), "Q");
        assert_eq!(translate(b"taa", 15).unwrap(), "*");
        assert_eq!(translate(b"tga", 15).unwrap(), "*");
    }

    #[test]
    fn test_translate_table25_tga_is_gly() {
        assert_eq!(translate(b"tga", 25).unwrap(), "G");
        assert_eq!(translate(b"taa", 25).unwrap(), "*");
        assert_eq!(translate(b"tag", 25).unwrap(), "*");
    }

    #[test]
    fn test_translate_short() {
        assert_eq!(translate(b"at", 1).unwrap(), "X");
    }

    #[test]
    fn test_unsupported_table() {
        assert!(translate(b"atg", 99).is_err());
    }

    #[test]
    fn test_is_supported_table() {
        assert!(is_supported_table(1));
        assert!(is_supported_table(4));
        assert!(is_supported_table(6));
        assert!(is_supported_table(11));
        assert!(is_supported_table(15));
        assert!(is_supported_table(25));
        assert!(!is_supported_table(99));
    }

    #[test]
    fn test_stop_codons() {
        assert_eq!(stop_codons(1), &[b"taa", b"tag", b"tga"]);
        assert_eq!(stop_codons(4), &[b"taa", b"tag"]);
        assert_eq!(stop_codons(6), &[b"tga"]);
        assert_eq!(stop_codons(15), &[b"taa", b"tga"]);
        assert_eq!(stop_codons(25), &[b"taa", b"tag"]);
    }

    #[test]
    fn test_start_codons() {
        assert_eq!(start_codons(1), &[b"atg", b"ttg", b"ctg"]);
        assert_eq!(start_codons(11), &[b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg"]);
    }
}
