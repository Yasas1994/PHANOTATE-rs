use crate::graph::Node;
use crate::orf::Orf;
use crate::overlap::OverlapInfo;

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
                .find(|o| o.start == left.position && o.stop == right.position && o.frame > 0)
        } else {
            orfs.iter()
                .find(|o| o.stop == left.position && o.start == right.position && o.frame < 0)
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
    overlaps: &[OverlapInfo],
) -> String {
    match format {
        Format::Gbk => write_gbk(id, seq, path, orfs, last_position, overlaps),
        Format::Gff => write_gff(id, path, orfs, overlaps),
        Format::Sco => write_sco(id, path, orfs, overlaps),
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
    overlaps: &[OverlapInfo],
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

    // Append overlapping genes
    for ov in overlaps {
        let (start, stop, strand) = overlap_coords(ov);
        out.push_str("     CDS             ");
        if strand == '+' {
            out.push_str(&format!("{}..{}\n", start, stop));
        } else {
            out.push_str(&format!("complement({}..{})\n", start, stop));
        }
        out.push_str(&format!(
            "                     /note=\"overlapping_gene;score={:.2E}\"\n",
            ov.score
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
fn write_gff(id: &str, path: &[(Node, Node, f64)], orfs: &[Orf], overlaps: &[OverlapInfo]) -> String {
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

    for ov in overlaps {
        let (start, stop, strand) = overlap_coords(ov);
        out.push_str(&format!(
            "{}\tphanotate\tCDS\t{}\t{}\t{:.2E}\t{}\t0\tID=CDS_{}_{}_ovl;overlap_score={:.2E}\n",
            id.trim_start_matches('>'),
            start,
            stop,
            ov.score,
            strand,
            start,
            stop,
            ov.score
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// SCO (Simple Coordinate Output)
// ---------------------------------------------------------------------------
fn write_sco(_id: &str, path: &[(Node, Node, f64)], orfs: &[Orf], _overlaps: &[OverlapInfo]) -> String {
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
    overlaps: &[OverlapInfo],
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

    // Append overlapping gene proteins
    for ov in overlaps {
        let (start, stop, strand) = overlap_coords(ov);
        let protein = match translate(&ov.orf.seq, table) {
            Ok(p) => p,
            Err(_) => continue,
        };
        out.push_str(&format!(
            ">{}_{}_{}_ovl {}\n",
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
    overlaps: &[OverlapInfo],
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

    for ov in overlaps {
        let (start, stop, strand) = overlap_coords(ov);
        out.push_str(&format!(
            ">{}_{}_{}_ovl {}\n",
            id.trim_start_matches('>'),
            start,
            stop,
            strand
        ));
        for chunk in ov.orf.seq.chunks(60) {
            out.push_str(&String::from_utf8_lossy(chunk));
            out.push('\n');
        }
    }

    out
}

/// Compute output coordinates for an overlapping gene.
/// Returns (start, stop, strand) in the same format as collect_orf_edges.
fn overlap_coords(ov: &OverlapInfo) -> (usize, usize, char) {
    if ov.orf.frame > 0 {
        (ov.orf.start, ov.orf.stop + 2, '+')
    } else {
        (ov.orf.start + 2, ov.orf.stop, '-')
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal Orf for testing output formats.
    fn make_orf(start: usize, stop: usize, frame: i8, seq: Vec<u8>) -> Orf {
        Orf {
            start,
            stop,
            frame,
            seq,
            rbs_score: 0,
            pstop: 0.0,
            weight_rbs: 1.0,
            hold: 1.0,
            weight: -1.0,
        }
    }

    /// Helper: create a forward-strand ORF edge path.
    fn fwd_path(start: usize, stop: usize) -> (Node, Node, f64) {
        (
            Node::new("CDS", "start", 1, start),
            Node::new("CDS", "stop", 1, stop),
            -42.0,
        )
    }

    /// Helper: create a reverse-strand ORF edge path.
    fn rev_path(start: usize, stop: usize) -> (Node, Node, f64) {
        (
            Node::new("CDS", "stop", -1, stop),
            Node::new("CDS", "start", -1, start),
            -99.0,
        )
    }

    // -----------------------------------------------------------------------
    // is_orf_edge tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_orf_edge_forward() {
        let orf = make_orf(10, 30, 1, b"atg".to_vec());
        let left = Node::new("CDS", "start", 1, 10);
        let right = Node::new("CDS", "stop", 1, 30);
        assert!(is_orf_edge(&left, &right, &[orf]));
    }

    #[test]
    fn test_is_orf_edge_reverse() {
        let orf = make_orf(10, 30, -1, b"atg".to_vec());
        let left = Node::new("CDS", "stop", -1, 30);
        let right = Node::new("CDS", "start", -1, 10);
        assert!(is_orf_edge(&left, &right, &[orf]));
    }

    #[test]
    fn test_is_orf_edge_non_cds() {
        let orf = make_orf(10, 30, 1, b"atg".to_vec());
        let left = Node::new("source", "source", 0, 0);
        let right = Node::new("CDS", "stop", 1, 30);
        assert!(!is_orf_edge(&left, &right, &[orf]));
    }

    #[test]
    fn test_is_orf_edge_wrong_positions() {
        let orf = make_orf(10, 30, 1, b"atg".to_vec());
        let left = Node::new("CDS", "start", 1, 99);
        let right = Node::new("CDS", "stop", 1, 30);
        assert!(!is_orf_edge(&left, &right, &[orf]));
    }

    // -----------------------------------------------------------------------
    // collect_orf_edges tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_collect_orf_edges_empty_path() {
        let orfs = vec![make_orf(10, 30, 1, b"atg".to_vec())];
        let path: Vec<(Node, Node, f64)> = vec![];
        let edges = collect_orf_edges(&path, &orfs);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_collect_orf_edges_single_forward() {
        let orfs = vec![make_orf(10, 30, 1, b"atg".to_vec())];
        let path = vec![fwd_path(10, 30)];
        let edges = collect_orf_edges(&path, &orfs);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, 10);  // start
        assert_eq!(edges[0].1, 32);  // stop + 2
        assert_eq!(edges[0].2, '+'); // strand
        assert_eq!(edges[0].3, -42.0); // weight
    }

    #[test]
    fn test_collect_orf_edges_single_reverse() {
        let orfs = vec![make_orf(10, 30, -1, b"atg".to_vec())];
        let path = vec![rev_path(10, 30)];
        let edges = collect_orf_edges(&path, &orfs);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, 12);  // right.position + 2
        assert_eq!(edges[0].1, 30);  // left.position
        assert_eq!(edges[0].2, '-'); // strand
        assert_eq!(edges[0].3, -99.0); // weight
    }

    #[test]
    fn test_collect_orf_edges_mixed() {
        let orfs = vec![
            make_orf(10, 30, 1, b"atg".to_vec()),
            make_orf(50, 70, -1, b"atg".to_vec()),
        ];
        let path = vec![fwd_path(10, 30), rev_path(50, 70)];
        let edges = collect_orf_edges(&path, &orfs);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].2, '+');
        assert_eq!(edges[1].2, '-');
    }

    #[test]
    fn test_collect_orf_edges_skips_non_orf() {
        let orfs = vec![make_orf(10, 30, 1, b"atg".to_vec())];
        let source = Node::new("source", "source", 0, 0);
        let start = Node::new("CDS", "start", 1, 10);
        let path = vec![
            (source, start, 0.0),
            fwd_path(10, 30),
        ];
        let edges = collect_orf_edges(&path, &orfs);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, '+');
    }

    // -----------------------------------------------------------------------
    // write_gbk tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_gbk_basic() {
        let seq = b"atgcatgcatgc";
        let orfs = vec![make_orf(1, 12, 1, b"atgcatgcatgc".to_vec())];
        let path = vec![fwd_path(1, 12)];
        let out = write_gbk(">test", seq, &path, &orfs, seq.len(), &[]);
        assert!(out.contains("LOCUS       test"));
        assert!(out.contains("DEFINITION  test"));
        assert!(out.contains("FEATURES"));
        assert!(out.contains("source          1..12"));
        assert!(out.contains("CDS             1..14"));
        assert!(out.contains("/note=\"score="));
        assert!(out.contains("ORIGIN"));
        assert!(out.contains("//"));
    }

    #[test]
    fn test_write_gbk_reverse_strand() {
        let seq = b"atgcatgcatgc";
        let orfs = vec![make_orf(1, 12, -1, b"atgcatgcatgc".to_vec())];
        let path = vec![rev_path(1, 12)];
        let out = write_gbk(">test", seq, &path, &orfs, seq.len(), &[]);
        // For rev_path(1, 12): left=stop@-1@12, right=start@-1@1
        // display: start=right.position+2=3, stop=left.position=12
        assert!(out.contains("complement(3..12)"));
    }

    #[test]
    fn test_write_gbk_no_orfs() {
        let seq = b"atgcatgcatgc";
        let out = write_gbk(">test", seq, &[], &[], seq.len(), &[]);
        assert!(out.contains("LOCUS       test"));
        assert!(!out.contains("CDS"));
    }

    #[test]
    fn test_write_gbk_sequence_formatting() {
        let seq = b"atgc";
        let out = write_gbk(">test", seq, &[], &[], seq.len(), &[]);
        // Should have position number and sequence
        assert!(out.contains("        1 atgc"));
    }

    // -----------------------------------------------------------------------
    // write_gff tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_gff_basic() {
        let orfs = vec![make_orf(10, 30, 1, b"atg".to_vec())];
        let path = vec![fwd_path(10, 30)];
        let out = write_gff(">test", &path, &orfs, &[]);
        assert!(out.starts_with("##gff-version 3\n"));
        assert!(out.contains("test\tphanotate\tCDS\t10\t32\t"));
        assert!(out.contains("\t+\t0\tID=CDS_10_32"));
    }

    #[test]
    fn test_write_gff_reverse() {
        let orfs = vec![make_orf(10, 30, -1, b"atg".to_vec())];
        let path = vec![rev_path(10, 30)];
        let out = write_gff(">test", &path, &orfs, &[]);
        assert!(out.contains("\t-\t0\t"));
    }

    #[test]
    fn test_write_gff_empty() {
        let out = write_gff(">test", &[], &[], &[]);
        assert_eq!(out, "##gff-version 3\n");
    }

    // -----------------------------------------------------------------------
    // write_sco tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_sco_basic() {
        let orfs = vec![make_orf(10, 30, 1, b"atg".to_vec())];
        let path = vec![fwd_path(10, 30)];
        let out = write_sco(">test", &path, &orfs, &[]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1);
        let cols: Vec<&str> = lines[0].split('\t').collect();
        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0], "10");
        assert_eq!(cols[1], "32");
        assert_eq!(cols[2], "+");
    }

    #[test]
    fn test_write_sco_empty() {
        let out = write_sco(">test", &[], &[], &[]);
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // write_protein_fasta tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_protein_fasta_basic() {
        // atg tgg taa -> M W *
        let orfs = vec![make_orf(1, 9, 1, b"atgtggtaa".to_vec())];
        let path = vec![fwd_path(1, 9)];
        let out = write_protein_fasta(">test", &path, &orfs, 11, &[]);
        assert!(out.starts_with(">test_1_11 +\n"));
        assert!(out.contains("MW*\n"));
    }

    #[test]
    fn test_write_protein_fasta_long_lines() {
        // 90 bases = 30 amino acids, should wrap at 60 chars
        let seq = b"atg".repeat(30);
        let orfs = vec![make_orf(1, 90, 1, seq.to_vec())];
        let path = vec![fwd_path(1, 90)];
        let out = write_protein_fasta(">test", &path, &orfs, 11, &[]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], ">test_1_92 +");
        // Should have 2 sequence lines (30 aa = 60 chars would be one line...)
        // Actually 30 aa is exactly 30 chars, so one line
        assert_eq!(lines[1].len(), 30);
    }

    #[test]
    fn test_write_protein_fasta_unsupported_table() {
        let orfs = vec![make_orf(1, 9, 1, b"atgtggtaa".to_vec())];
        let path = vec![fwd_path(1, 9)];
        let out = write_protein_fasta(">test", &path, &orfs, 99, &[]);
        // Should skip ORFs with unsupported table
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // write_nucleotide_fasta tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_nucleotide_fasta_basic() {
        let orfs = vec![make_orf(1, 12, 1, b"atgcatgcatgc".to_vec())];
        let path = vec![fwd_path(1, 12)];
        let out = write_nucleotide_fasta(">test", &path, &orfs, &[]);
        assert!(out.starts_with(">test_1_14 +\n"));
        assert!(out.contains("atgcatgcatgc\n"));
    }

    #[test]
    fn test_write_nucleotide_fasta_wrap() {
        let seq = b"atgc".repeat(20); // 80 chars
        let orfs = vec![make_orf(1, 80, 1, seq.to_vec())];
        let path = vec![fwd_path(1, 80)];
        let out = write_nucleotide_fasta(">test", &path, &orfs, &[]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], ">test_1_82 +");
        assert_eq!(lines[1].len(), 60);
        assert_eq!(lines[2].len(), 20);
    }

    // -----------------------------------------------------------------------
    // write_primary dispatcher tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_primary_gbk() {
        let seq = b"atgc";
        let out = write_primary(">test", seq, &[], &[], 4, Format::Gbk, &[]);
        assert!(out.contains("LOCUS"));
    }

    #[test]
    fn test_write_primary_gff() {
        let out = write_primary(">test", b"atgc", &[], &[], 4, Format::Gff, &[]);
        assert!(out.starts_with("##gff-version 3"));
    }

    #[test]
    fn test_write_primary_sco() {
        let out = write_primary(">test", b"atgc", &[], &[], 4, Format::Sco, &[]);
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // Format enum tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_format_eq() {
        assert_eq!(Format::Gbk, Format::Gbk);
        assert_ne!(Format::Gbk, Format::Gff);
    }

    #[test]
    fn test_format_copy() {
        let f = Format::Gff;
        let f2 = f;
        assert_eq!(f, f2); // Copy trait allows this
    }
}
