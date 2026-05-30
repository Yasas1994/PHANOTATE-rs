//! NCBI genetic code translation tables.

/// Translate a DNA sequence using the specified NCBI genetic code table.
/// Supported tables: 1 (Standard), 11 (Bacterial/Archaeal/Plastid).
pub fn translate(seq: &[u8], table: u8) -> Result<String, String> {
    let table_fn = match table {
        1 => translate_table1,
        11 => translate_table11,
        _ => return Err(format!("Unsupported translation table: {}. Supported: 1, 11", table)),
    };
    Ok(table_fn(seq))
}

fn translate_table1(seq: &[u8]) -> String {
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

fn translate_table11(seq: &[u8]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_atg() {
        assert_eq!(translate(b"atg", 1).unwrap(), "M");
        assert_eq!(translate(b"atg", 11).unwrap(), "M");
    }

    #[test]
    fn test_translate_stop() {
        assert_eq!(translate(b"taa", 1).unwrap(), "*");
        assert_eq!(translate(b"tag", 1).unwrap(), "*");
        assert_eq!(translate(b"tga", 1).unwrap(), "*");
    }

    #[test]
    fn test_translate_short() {
        assert_eq!(translate(b"at", 1).unwrap(), "X");
    }

    #[test]
    fn test_unsupported_table() {
        assert!(translate(b"atg", 99).is_err());
    }
}
