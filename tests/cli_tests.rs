use std::process::Command;

const PHANOTATE_RS: &str = env!("CARGO_BIN_EXE_phanotate-rs");
const PHIX174: &str = "../PHANOTATE/tests/phiX174.fasta";

/// Small synthetic genome with N runs for -m testing.
/// Two ORFs separated by an N-run. Each ORF is ~90 bp with start/stop.
const MASKED_FASTA: &str = ">masked_test\n\
    ATGAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCTAA\
    NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\
    ATGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCTAA\n";

fn run(args: &[&str], stdin: Option<&str>) -> (String, String, i32) {
    let mut cmd = Command::new(PHANOTATE_RS);
    cmd.args(args);
    if let Some(input) = stdin {
        use std::io::Write;
        let mut child = cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to spawn");
        {
            let stdin = child.stdin.as_mut().unwrap();
            stdin.write_all(input.as_bytes()).unwrap();
            // Close stdin so the child sees EOF
        }
        drop(child.stdin.take());
        let output = child.wait_with_output().unwrap();
        (
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
            output.status.code().unwrap_or(-1),
        )
    } else {
        let output = cmd.output().expect("failed to execute");
        (
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
            output.status.code().unwrap_or(-1),
        )
    }
}

// ---------------------------------------------------------------------------
// Help flag
// ---------------------------------------------------------------------------
#[test]
fn test_help_flag() {
    let (stdout, _stderr, code) = run(&["-h"], None);
    assert_eq!(code, 0, "help should exit 0");
    assert!(stdout.contains("-a"), "help should mention -a");
    assert!(stdout.contains("-c"), "help should mention -c");
    assert!(stdout.contains("-d"), "help should mention -d");
    assert!(stdout.contains("-f"), "help should mention -f");
    assert!(stdout.contains("-g"), "help should mention -g");
    assert!(stdout.contains("-i"), "help should mention -i");
    assert!(stdout.contains("-m"), "help should mention -m");
}

// ---------------------------------------------------------------------------
// Format flag (-f)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_f_gbk() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "gbk"], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("LOCUS"));
    assert!(stdout.contains("FEATURES"));
    assert!(stdout.contains("CDS"));
    assert!(stdout.contains("ORIGIN"));
    assert!(stdout.contains("//"));
}

#[test]
fn test_flag_f_gff() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "gff"], None);
    assert_eq!(code, 0);
    assert!(stdout.starts_with("##gff-version 3\n"));
    for line in stdout.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 9, "GFF line should have 9 columns: {}", line);
        assert_eq!(cols[1], "phanotate");
        assert_eq!(cols[2], "CDS");
    }
}

#[test]
fn test_flag_f_sco() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "sco"], None);
    assert_eq!(code, 0);
    for line in stdout.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "SCO line should have 4 columns: {}", line);
    }
}

#[test]
fn test_flag_f_invalid() {
    let (_stdout, stderr, code) = run(&["-i", PHIX174, "-f", "xyz"], None);
    assert_ne!(code, 0, "invalid format should fail");
    assert!(
        stderr.contains("gbk") || stderr.contains("gff") || stderr.contains("sco"),
        "error should mention valid formats: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// Translation table (-g)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_g_table1() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "1", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_table11() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "11", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_table4() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "4", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_table6() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "6", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_table15() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "15", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_table25() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-g", "25", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
}

#[test]
fn test_flag_g_invalid() {
    let (_stdout, stderr, code) = run(&["-i", PHIX174, "-g", "99"], None);
    assert_ne!(code, 0, "invalid table should fail");
    assert!(stderr.contains("table"), "error should mention table: {}", stderr);
}

// ---------------------------------------------------------------------------
// Protein output (-a)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_a_protein() {
    let tmpfile = tempfile::NamedTempFile::new().unwrap();
    let path = tmpfile.path().to_str().unwrap();
    let (_stdout, _stderr, code) = run(&["-i", PHIX174, "-a", path, "-f", "sco"], None);
    assert_eq!(code, 0);
    let protein = std::fs::read_to_string(path).unwrap();
    assert!(protein.starts_with('>'));
    for chunk in protein.split('>').skip(1) {
        let lines: Vec<&str> = chunk.lines().collect();
        assert!(!lines.is_empty(), "each protein record needs a header");
        assert!(lines.len() >= 2, "each protein record needs sequence: {:?}", lines);
    }
}

// ---------------------------------------------------------------------------
// Nucleotide output (-d)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_d_nucleotide() {
    let tmpfile = tempfile::NamedTempFile::new().unwrap();
    let path = tmpfile.path().to_str().unwrap();
    let (_stdout, _stderr, code) = run(&["-i", PHIX174, "-d", path, "-f", "sco"], None);
    assert_eq!(code, 0);
    let nuc = std::fs::read_to_string(path).unwrap();
    assert!(nuc.starts_with('>'));
    for chunk in nuc.split('>').skip(1) {
        let lines: Vec<&str> = chunk.lines().collect();
        assert!(!lines.is_empty());
        assert!(lines.len() >= 2, "each nuc record needs sequence");
    }
}

// ---------------------------------------------------------------------------
// Stdin input (no -i)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_i_stdin() {
    let fasta = std::fs::read_to_string(PHIX174).unwrap();
    let (stdout1, _stderr1, code1) = run(&["-f", "sco"], Some(&fasta));
    assert_eq!(code1, 0);

    let (stdout2, _stderr2, code2) = run(&["-i", PHIX174, "-f", "sco"], None);
    assert_eq!(code2, 0);

    assert_eq!(stdout1, stdout2, "stdin and file input should produce identical output");
}

// ---------------------------------------------------------------------------
// Closed ends (-c)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_c_closed_ends() {
    let (stdout_open, _stderr, _code) = run(&["-i", PHIX174, "-f", "sco"], None);
    let (stdout_closed, _stderr, _code) = run(&["-i", PHIX174, "-c", "-f", "sco"], None);

    let open_count = stdout_open.lines().count();
    let closed_count = stdout_closed.lines().count();

    assert!(
        closed_count <= open_count,
        "closed ends should not have more ORFs: open={}, closed={}",
        open_count,
        closed_count
    );
}

// ---------------------------------------------------------------------------
// Mask N runs (-m)
// ---------------------------------------------------------------------------
#[test]
fn test_flag_m_mask_n() {
    let (stdout, _stderr, code) = run(&["-m", "-f", "sco"], Some(MASKED_FASTA));
    assert_eq!(code, 0);

    // The N-run is at positions 61..122 (after normalization preserves Ns).
    // No gene should span this region.
    for line in stdout.lines() {
        if line.starts_with("#id:") {
            continue; // "NO ORFS FOUND" line
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "SCO line should have 4 columns: {}", line);
        let start: usize = cols[0].parse().unwrap();
        let stop: usize = cols[1].parse().unwrap();
        let (lo, hi) = (start.min(stop), start.max(stop));
        assert!(
            hi < 61 || lo > 122,
            "gene {}..{} should not span N-run at 61..122",
            lo,
            hi
        );
    }
}

// ---------------------------------------------------------------------------
// Combo -c -m
// ---------------------------------------------------------------------------
#[test]
fn test_flag_combo_c_m() {
    let (stdout, _stderr, code) = run(&["-c", "-m", "-f", "sco"], Some(MASKED_FASTA));
    assert_eq!(code, 0);
    for line in stdout.lines() {
        if line.starts_with("#id:") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "SCO line should have 4 columns: {}", line);
    }
}

// ---------------------------------------------------------------------------
// End-to-end on phiX174
// ---------------------------------------------------------------------------
#[test]
fn test_phix174_gbk_output() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "gbk"], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("LOCUS       phiX174"));
    assert!(stdout.contains("5386 bp"));
    let cds_count = stdout.lines().filter(|l| l.contains("CDS")).count();
    assert!(cds_count >= 6, "phiX174 should have at least 6 CDS features, got {}", cds_count);
}

#[test]
fn test_phix174_sco_matches_golden() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "sco"], None);
    assert_eq!(code, 0);

    let golden = std::fs::read_to_string("tests/golden/phiX174.tabular").unwrap();
    let golden_lines: Vec<&str> = golden.lines().skip(2).collect();
    let output_lines: Vec<&str> = stdout.lines().collect();

    assert_eq!(
        output_lines.len(),
        golden_lines.len(),
        "SCO output should have same number of genes as golden"
    );

    for (out, gold) in output_lines.iter().zip(golden_lines.iter()) {
        let out_cols: Vec<&str> = out.split('\t').collect();
        let gold_cols: Vec<&str> = gold.split('\t').collect();
        assert_eq!(out_cols[0], gold_cols[0], "start position mismatch");
        assert_eq!(out_cols[1], gold_cols[1], "stop position mismatch");
        assert_eq!(out_cols[2], gold_cols[2], "strand mismatch");
    }
}

// ---------------------------------------------------------------------------
// Regression test for internal stop codons with non-standard genetic codes
// ---------------------------------------------------------------------------
#[test]
fn test_no_internal_stops_table4() {
    let fasta = include_str!("../../test_genomes/MT135298.fasta");
    let (stdout, _stderr, code) = run(
        &["-g", "4", "-a", "/tmp/test_table4_proteins.faa"],
        Some(fasta),
    );
    assert_eq!(code, 0, "non-zero exit: {}", stdout);

    let proteins = std::fs::read_to_string("/tmp/test_table4_proteins.faa").unwrap();
    for line in proteins.lines() {
        if line.starts_with('>') {
            continue;
        }
        // Count stop codons in the protein sequence
        let stops: Vec<_> = line.match_indices('*').collect();
        // Each protein should have at most one stop codon (at the end)
        assert!(
            stops.len() <= 1,
            "Protein has internal stop codons: {} stops in '{}'",
            stops.len(),
            line
        );
    }
}
