use crate::graph::Node;
use crate::orf::Orf;

/// Primary output format enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Gbk,
    Gff,
    Sco,
}

/// Check if a path edge corresponds to an actual ORF.
fn is_orf_edge(left: &Node, right: &Node, orfs: &[Orf]) -> bool {
    if left.gene != "CDS" || right.gene != "CDS" {
        return false;
    }
    if left.node_type == "start" && right.node_type == "stop" && left.frame > 0 {
        orfs.iter().any(|o| o.start == left.position && o.stop == right.position && o.frame == left.frame)
    } else if left.node_type == "stop" && right.node_type == "start" && left.frame < 0 {
        orfs.iter().any(|o| o.stop == left.position && o.start == right.position && o.frame == left.frame)
    } else {
        false
    }
}

/// Extract (start, stop, strand, orf_ref) for each ORF edge in the path.
fn collect_orf_edges<'a>(
    path: &[(Node, Node, f64)],
    orfs: &'a [Orf],
) -> Vec<(usize, usize, char, f64, &'a Orf)> {
    let mut result = Vec::new();
    for (left, right, weight) in path {
        if !is_orf_edge(left, right, orfs) {
            continue;
        }
        let (start, stop, strand) = if left.node_type == "start" && right.node_type == "stop" {
            (left.position, right.position + 2, '+')
        } else {
            (right.position + 2, left.position, '-')
        };

        let orf = if strand == '+' {
            orfs.iter()
                .find(|o| o.start == start && o.stop == stop - 2 && o.frame > 0)
        } else {
            orfs.iter()
                .find(|o| o.stop == start && o.start == stop - 2 && o.frame < 0)
        };

        if let Some(orf) = orf {
            result.push((start, stop, strand, *weight, orf));
        }
    }
    result
}

/// Primary output dispatcher.
pub fn write_primary(
    id: &str,
    seq: &[u8],
    path: &[(Node, Node, f64)],
    orfs: &[Orf],
    last_position: usize,
    format: Format,
) -> String {
    match format {
        Format::Gbk => write_gbk(id, seq, path, orfs, last_position),
        Format::Gff => write_gff(id, path, orfs),
        Format::Sco => write_sco(id, path, orfs),
    }
}

// ---------------------------------------------------------------------------
// GBK (GenBank)
// ---------------------------------------------------------------------------
fn write_gbk(
    id: &str,
    seq: &[u8],
    path: &[(Node, Node, f64)],
    orfs: &[Orf],
    last_position: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LOCUS       {} {:>10} bp    DNA             PHG\n",
        id.trim_start_matches('>'),
        last_position
    ));
    out.push_str(&format!("DEFINITION  {}\n", id.trim_start_matches('>')));
    out.push_str("FEATURES             Location/Qualifiers\n");
    out.push_str(&format!("     source          1..{}\n", last_position));

    for (start, stop, strand, weight, _orf) in collect_orf_edges(path, orfs) {
        out.push_str("     CDS             ");
        if strand == '+' {
            out.push_str(&format!("{}..{}\n", start, stop));
        } else {
            out.push_str(&format!("complement({}..{})\n", start, stop));
        }
        out.push_str(&format!(
            "                     /note=\"score={:.2E}\"\n",
            weight
        ));
    }

    out.push_str("ORIGIN\n");
    for (i, chunk) in seq.chunks(10).enumerate() {
        let pos = i * 10 + 1;
        if i % 6 == 0 {
            out.push('\n');
            out.push_str(&format!("{:>9} ", pos));
        } else {
            out.push(' ');
        }
        out.push_str(&String::from_utf8_lossy(chunk).to_lowercase());
    }
    out.push('\n');
    out.push_str("//\n");

    out
}

// ---------------------------------------------------------------------------
// GFF (GFF3)
// ---------------------------------------------------------------------------
fn write_gff(id: &str, path: &[(Node, Node, f64)], orfs: &[Orf]) -> String {
    let mut out = String::new();
    out.push_str("##gff-version 3\n");

    for (start, stop, strand, weight, _orf) in collect_orf_edges(path, orfs) {
        out.push_str(&format!(
            "{}\tphanotate\tCDS\t{}\t{}\t{:.2E}\t{}\t0\tID=CDS_{}_{};score={:.2E}\n",
            id.trim_start_matches('>'),
            start,
            stop,
            weight,
            strand,
            start,
            stop,
            weight
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// SCO (Simple Coordinate Output)
// ---------------------------------------------------------------------------
fn write_sco(id: &str, path: &[(Node, Node, f64)], orfs: &[Orf]) -> String {
    let mut out = String::new();
    for (start, stop, strand, weight, _orf) in collect_orf_edges(path, orfs) {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2E}\n",
            start, stop, strand, weight
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Protein FASTA (for -a flag)
// ---------------------------------------------------------------------------
pub fn write_protein_fasta(
    id: &str,
    path: &[(Node, Node, f64)],
    orfs: &[Orf],
    table: u8,
) -> String {
    use crate::codon_table::translate;

    let mut out = String::new();
    for (start, stop, strand, _weight, orf) in collect_orf_edges(path, orfs) {
        let protein = match translate(&orf.seq, table) {
            Ok(p) => p,
            Err(_) => continue,
        };
        out.push_str(&format!(
            ">{}_{}_{} {}\n",
            id.trim_start_matches('>'),
            start,
            stop,
            strand
        ));
        for chunk in protein.as_bytes().chunks(60) {
            out.push_str(&String::from_utf8_lossy(chunk));
            out.push('\n');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Nucleotide FASTA (for -d flag)
// ---------------------------------------------------------------------------
pub fn write_nucleotide_fasta(
    id: &str,
    path: &[(Node, Node, f64)],
    orfs: &[Orf],
) -> String {
    let mut out = String::new();
    for (start, stop, strand, _weight, orf) in collect_orf_edges(path, orfs) {
        out.push_str(&format!(
            ">{}_{}_{} {}\n",
            id.trim_start_matches('>'),
            start,
            stop,
            strand
        ));
        for chunk in orf.seq.chunks(60) {
            out.push_str(&String::from_utf8_lossy(chunk));
            out.push('\n');
        }
    }
    out
}
