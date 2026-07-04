//! NCBI Translation Tables.
//!
//! All functions accept a byte slice (lowercase or uppercase) and return a protein
//! string where '*' denotes a stop codon and 'X' denotes an unknown/incomplete
//! codon.
//!
//! Supports NCBI translation tables 1-6, 9-16, and 21-31. Tables 0, 7, 8, 17-20,
//! and >31 are not supported.

// ---------------------------------------------------------------------------
// Codon lookup table
// ---------------------------------------------------------------------------

/// 2-bit base encoding: A=00, C=01, G=10, T=11.  Any other base maps to 4.
const BASE_INDEX: [u8; 256] = {
    let mut t = [4u8; 256];
    t[b'A' as usize] = 0;
    t[b'C' as usize] = 1;
    t[b'G' as usize] = 2;
    t[b'T' as usize] = 3;
    t[b'a' as usize] = 0;
    t[b'c' as usize] = 1;
    t[b'g' as usize] = 2;
    t[b't' as usize] = 3;
    t
};

/// Lookup table indexed by `[table][codon]`, where codon index is
/// `base1 * 16 + base2 * 4 + base3` using the encoding above.
///
/// Invalid table rows are filled with `b'?'`; they are never accessed because
/// `is_supported_table` rejects them.
const CODONS: [[u8; 64]; 32] = [
    [b'?'; 64],                                                           // 0 invalid
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF", // 1 Standard
    *b"KNKNTTTT*S*SMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 2 Vertebrate Mitochondrial
    *b"KNKNTTTTRSRSMIMIQHQHPPPPRRRRTTTTEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 3 Yeast Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 4 Mold/Protozoan/Coelenterate Mitochondrial + Mycoplasma/Spiroplasma
    *b"KNKNTTTTSSSSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 5 Invertebrate Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSS*CWCLFLF", // 6 Ciliate Nuclear
    [b'?'; 64],                                                           // 7 invalid
    [b'?'; 64],                                                           // 8 invalid
    *b"NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 9 Echinoderm and Flatworm Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSCCWCLFLF", // 10 Euplotid Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF", // 11 Bacterial/Archaeal/Plant Plastid
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLSLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF", // 12 Alternative Yeast Nuclear
    *b"KNKNTTTTGSGSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 13 Ascidian Mitochondrial
    *b"NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYY*YSSSSWCWCLFLF", // 14 Alternative Flatworm Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YQYSSSS*CWCLFLF", // 15 Blepharisma Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YLYSSSS*CWCLFLF", // 16 Chlorophycean Mitochondrial
    [b'?'; 64],                                                           // 17 invalid
    [b'?'; 64],                                                           // 18 invalid
    [b'?'; 64],                                                           // 19 invalid
    [b'?'; 64],                                                           // 20 invalid
    *b"NNKNTTTTSSSSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 21 Trematode Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YLY*SSS*CWCLFLF", // 22 Scenedesmus obliquus Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWC*FLF", // 23 Thraustochytrium Mitochondrial
    *b"KNKNTTTTSSKSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF", // 24 Rhabdopleuridae Mitochondrial
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSGCWCLFLF", // 25 Candidate Division SR1 and Gracilibacteria
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLALEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF", // 26 Pachysolen tannophilus Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSSWCWCLFLF", // 27 Karyorelict Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSSWCWCLFLF", // 28 Condylostoma Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYYYYSSSS*CWCLFLF", // 29 Mesodinium Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVEYEYSSSS*CWCLFLF", // 30 Peritrich Nuclear
    *b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVEYEYSSSSWCWCLFLF", // 31 Blastocrithidia Nuclear
];

/// Map a 3-mer to its CODONS table index, or `None` if it contains an
/// invalid base.
fn codon_index(codon: &[u8]) -> Option<usize> {
    if codon.len() < 3 {
        return None;
    }
    let mut idx = 0usize;
    for i in 0..3 {
        let v = BASE_INDEX[codon[i] as usize];
        if v == 4 {
            return None;
        }
        idx = idx * 4 + v as usize;
    }
    Some(idx)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Translate `seq` using the NCBI translation table `table`.
pub fn translate(seq: &[u8], table: u8) -> Result<String, String> {
    if !is_supported_table(table) {
        return Err(format!(
            "Translation table {} is not supported. Supported tables: 1-6, 9-16, 21-31.",
            table
        ));
    }
    let table_aa = &CODONS[table as usize];
    let protein: String = seq
        .chunks(3)
        .map(|codon| {
            if codon.len() < 3 {
                return 'X';
            }
            match codon_index(codon) {
                Some(idx) => table_aa[idx] as char,
                None => 'X',
            }
        })
        .collect();
    Ok(protein)
}

/// Return true if `table` is supported by this module.
pub fn is_supported_table(table: u8) -> bool {
    matches!(table, 1..=6 | 9..=16 | 21..=31)
}

/// Return the canonical NCBI name for a table number.
pub fn table_name(table: u8) -> &'static str {
    match table {
        1 => "The Standard Code",
        2 => "The Vertebrate Mitochondrial Code",
        3 => "The Yeast Mitochondrial Code",
        4 => "The Mold, Protozoan, and Coelenterate Mitochondrial Code and the Mycoplasma/Spiroplasma Code",
        5 => "The Invertebrate Mitochondrial Code",
        6 => "The Ciliate, Dasycladacean and Hexamita Nuclear Code",
        9 => "The Echinoderm and Flatworm Mitochondrial Code",
        10 => "The Euplotid Nuclear Code",
        11 => "The Bacterial, Archaeal and Plant Plastid Code",
        12 => "The Alternative Yeast Nuclear Code",
        13 => "The Ascidian Mitochondrial Code",
        14 => "The Alternative Flatworm Mitochondrial Code",
        15 => "Blepharisma Nuclear Code",
        16 => "Chlorophycean Mitochondrial Code",
        21 => "Trematode Mitochondrial Code",
        22 => "Scenedesmus obliquus Mitochondrial Code",
        23 => "Thraustochytrium Mitochondrial Code",
        24 => "Rhabdopleuridae Mitochondrial Code",
        25 => "Candidate Division SR1 and Gracilibacteria Code",
        26 => "Pachysolen tannophilus Nuclear Code",
        27 => "Karyorelict Nuclear Code",
        28 => "Condylostoma Nuclear Code",
        29 => "Mesodinium Nuclear Code",
        30 => "Peritrich Nuclear Code",
        31 => "Blastocrithidia Nuclear Code",
        _ => "Unknown",
    }
}

/// Return the set of stop codons (lowercase) for a given table.
pub fn stop_codons(table: u8) -> &'static [&'static [u8]] {
    match table {
        1 | 11 => &[b"taa", b"tag", b"tga"],
        2 => &[b"taa", b"tag", b"aga", b"agg"],
        3 | 4 | 5 | 9 | 10 | 21 | 24 | 25 => &[b"taa", b"tag"],
        6 | 27 | 29 | 30 => &[b"tga"],
        12 | 26 => &[b"taa", b"tag", b"tga"],
        13 => &[b"taa", b"tag"],
        14 | 15 | 16 | 31 => &[b"taa", b"tga"],
        22 => &[b"tca", b"taa", b"tga"],
        23 => &[b"tta", b"taa", b"tag", b"tga"],
        28 => &[b"taa", b"tag", b"tga"],
        _ => &[],
    }
}

/// Return the set of start codons (lowercase) for a given table.
pub fn start_codons(table: u8) -> &'static [&'static [u8]] {
    match table {
        1 => &[b"ttg", b"ctg", b"atg"],
        2 => &[b"att", b"atc", b"ata", b"atg", b"gtg"],
        3 => &[b"ata", b"atg", b"gtg"],
        4 => &[
            b"tta", b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg",
        ],
        5 => &[b"ttg", b"att", b"atc", b"ata", b"atg", b"gtg"],
        6 | 10 | 14 | 15 | 16 | 22 | 27 | 28 | 29 | 30 | 31 => &[b"atg"],
        9 | 21 => &[b"atg", b"gtg"],
        11 => &[b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg"],
        12 | 26 => &[b"ctg", b"atg"],
        13 => &[b"ttg", b"ata", b"atg", b"gtg"],
        23 => &[b"att", b"atg", b"gtg"],
        24 | 25 => &[b"ttg", b"ctg", b"atg", b"gtg"],
        _ => &[b"atg"],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_atg() {
        assert_eq!(translate(b"atg", 1).unwrap(), "M");
        assert_eq!(translate(b"atg", 11).unwrap(), "M");
        assert_eq!(translate(b"ATG", 11).unwrap(), "M");
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
    fn test_translate_table2() {
        assert_eq!(translate(b"tga", 2).unwrap(), "W"); // UGA -> Trp
        assert_eq!(translate(b"aga", 2).unwrap(), "*"); // AGA -> stop
        assert_eq!(translate(b"agg", 2).unwrap(), "*"); // AGG -> stop
        assert_eq!(translate(b"ata", 2).unwrap(), "M"); // AUA -> Met
    }

    #[test]
    fn test_translate_table3_cun_is_thr() {
        assert_eq!(translate(b"ctt", 3).unwrap(), "T");
        assert_eq!(translate(b"ctc", 3).unwrap(), "T");
        assert_eq!(translate(b"cta", 3).unwrap(), "T");
        assert_eq!(translate(b"ctg", 3).unwrap(), "T");
    }

    #[test]
    fn test_translate_table5_aga_agg_are_ser() {
        assert_eq!(translate(b"aga", 5).unwrap(), "S");
        assert_eq!(translate(b"agg", 5).unwrap(), "S");
    }

    #[test]
    fn test_translate_table22_tag_is_leu_and_tca_is_stop() {
        assert_eq!(translate(b"tag", 22).unwrap(), "L");
        assert_eq!(translate(b"tca", 22).unwrap(), "*");
    }

    #[test]
    fn test_translate_table26_ctg_is_ala() {
        assert_eq!(translate(b"ctg", 26).unwrap(), "A");
    }

    #[test]
    fn test_translate_short() {
        assert_eq!(translate(b"at", 1).unwrap(), "X");
    }

    #[test]
    fn test_translate_invalid_base() {
        assert_eq!(translate(b"atn", 1).unwrap(), "X");
    }

    #[test]
    fn test_unsupported_table() {
        for &t in &[0u8, 7, 8, 17, 18, 19, 20, 32, 99] {
            assert!(
                translate(b"atg", t).is_err(),
                "table {} should be rejected",
                t
            );
        }
    }

    #[test]
    fn test_is_supported_table() {
        for t in [
            1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31,
        ] {
            assert!(is_supported_table(t), "table {} should be supported", t);
        }
        for t in [0, 7, 8, 17, 18, 19, 20, 32, 99] {
            assert!(
                !is_supported_table(t),
                "table {} should not be supported",
                t
            );
        }
    }

    #[test]
    fn test_stop_codons() {
        assert_eq!(stop_codons(1), &[b"taa", b"tag", b"tga"]);
        assert_eq!(stop_codons(4), &[b"taa", b"tag"]);
        assert_eq!(stop_codons(6), &[b"tga"]);
        assert_eq!(stop_codons(15), &[b"taa", b"tga"]);
        assert_eq!(stop_codons(25), &[b"taa", b"tag"]);
        assert_eq!(stop_codons(2), &[b"taa", b"tag", b"aga", b"agg"]);
        assert_eq!(stop_codons(22), &[b"tca", b"taa", b"tga"]);
    }

    #[test]
    fn test_start_codons() {
        assert_eq!(start_codons(1), &[b"ttg", b"ctg", b"atg"]);
        assert_eq!(
            start_codons(11),
            &[b"ttg", b"ctg", b"att", b"atc", b"ata", b"atg", b"gtg"]
        );
        assert_eq!(start_codons(2), &[b"att", b"atc", b"ata", b"atg", b"gtg"]);
    }
}
