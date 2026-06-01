use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Genome {
    pub id: String,
    pub seq: Vec<u8>,    // lowercase ASCII nucleotides
    pub rc_seq: Vec<u8>, // reverse complement, pre-computed
}

/// Read a FASTA file, returning one or more Genome records.
/// Ambiguous bases are handled as in the reference:
///   s, b, v -> g; everything else -> a
#[cfg_attr(not(test), allow(dead_code))]
pub fn read_fasta<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Genome>> {
    let data = fs::read_to_string(path)?;
    read_fasta_data(&data)
}

/// Parse FASTA data from a string.
pub fn read_fasta_data(data: &str) -> anyhow::Result<Vec<Genome>> {
    let mut genomes = Vec::new();
    let mut current_id = String::new();
    let mut current_seq = String::new();

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('>') {
            if !current_id.is_empty() {
                let seq = normalize_seq(&current_seq);
                let rc_seq = rev_comp(&seq);
                genomes.push(Genome {
                    id: current_id.clone(),
                    seq,
                    rc_seq,
                });
            }
            current_id = line.split_whitespace().next().unwrap_or("").to_string();
            current_seq.clear();
        } else {
            current_seq.push_str(line);
        }
    }

    if !current_id.is_empty() {
        let seq = normalize_seq(&current_seq);
        let rc_seq = rev_comp(&seq);
        genomes.push(Genome {
            id: current_id,
            seq,
            rc_seq,
        });
    }

    Ok(genomes)
}

/// Parse GenBank data from a string.
pub fn read_genbank(data: &str) -> anyhow::Result<Vec<Genome>> {
    let mut genomes = Vec::new();
    let mut current_id = String::new();
    let mut current_seq = String::new();
    let mut in_origin = false;

    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("LOCUS") {
            if !current_id.is_empty() {
                let seq = normalize_seq(&current_seq);
                let rc_seq = rev_comp(&seq);
                genomes.push(Genome {
                    id: current_id.clone(),
                    seq,
                    rc_seq,
                });
            }
            current_id = trimmed.split_whitespace().nth(1).unwrap_or("").to_string();
            current_seq.clear();
            in_origin = false;
        } else if trimmed.starts_with("ORIGIN") {
            in_origin = true;
        } else if trimmed.starts_with("//") {
            in_origin = false;
        } else if in_origin {
            // GenBank ORIGIN lines: "     1 gatcctccat atacaacggt ..."
            // Skip the position number and extract sequence
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            for part in parts.iter().skip(1) {
                current_seq.push_str(part);
            }
        }
    }

    if !current_id.is_empty() {
        let seq = normalize_seq(&current_seq);
        let rc_seq = rev_comp(&seq);
        genomes.push(Genome {
            id: current_id,
            seq,
            rc_seq,
        });
    }

    Ok(genomes)
}

/// Convert to lowercase, then map ambiguous bases.
/// Preserves 'n'/'N' so that `-m` (mask runs of N) can detect them.
pub fn normalize_seq(seq: &str) -> Vec<u8> {
    // 256-byte lookup table: map every possible byte to its normalized form.
    // Lowercase letters are mapped directly; uppercase letters are mapped
    // via their lowercase counterparts. Everything else maps to 'a'.
    const LUT: [u8; 256] = {
        let mut lut = [b'a'; 256];
        let mut i = 0u16;
        while i < 256 {
            lut[i as usize] = match i as u8 {
                b'A' | b'a' => b'a',
                b'C' | b'c' => b'c',
                b'T' | b't' | b'U' | b'u' => b't',
                b'G' | b'g' => b'g',
                b'N' | b'n' => b'n',
                b'S' | b's' | b'B' | b'b' | b'V' | b'v' => b'g',
                _ => b'a',
            };
            i += 1;
        }
        lut
    };
    seq.bytes().map(|b| LUT[b as usize]).collect()
}

/// Reverse complement a DNA sequence.
pub fn rev_comp(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'a' => b't',
            b't' => b'a',
            b'g' => b'c',
            b'c' => b'g',
            _ => b'a',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rev_comp() {
        assert_eq!(rev_comp(b"atgc"), vec![b'g', b'c', b'a', b't']);
        assert_eq!(rev_comp(b"nnnx"), vec![b'a', b'a', b'a', b'a']);
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize_seq("ATGCSBVX"), vec![b'a', b't', b'g', b'c', b'g', b'g', b'g', b'a']);
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn check_phix_seq() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            println!("Rust len: {}", genome.seq.len());
            println!("Rust first 50: {}", String::from_utf8_lossy(&genome.seq[..50]));
            println!("Rust last 50: {}", String::from_utf8_lossy(&genome.seq[genome.seq.len()-50..]));
        }
    }
}
