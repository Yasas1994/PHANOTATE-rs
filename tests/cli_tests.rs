use std::io::Write;
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
    assert!(stdout.contains("--model"), "help should mention --model");
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
    for line in stdout.lines().skip(2) {
        if line.starts_with('#') {
            continue;
        }
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
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5, "SCO line should have 5 columns: {}", line);
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
    assert!(
        stderr.contains("table"),
        "error should mention table: {}",
        stderr
    );
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
        assert!(
            lines.len() >= 2,
            "each protein record needs sequence: {:?}",
            lines
        );
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

    assert_eq!(
        stdout1, stdout2,
        "stdin and file input should produce identical output"
    );
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
        if line.starts_with("#id:") || line.starts_with("# uses_sd:") {
            continue; // header / "NO ORFS FOUND" line
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5, "SCO line should have 5 columns: {}", line);
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
        if line.starts_with("#id:") || line.starts_with("# uses_sd:") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5, "SCO line should have 5 columns: {}", line);
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
    assert!(
        cds_count >= 6,
        "phiX174 should have at least 6 CDS features, got {}",
        cds_count
    );
}

#[test]
fn test_phix174_sco_matches_golden() {
    let (stdout, _stderr, code) = run(&["-i", PHIX174, "-f", "sco"], None);
    assert_eq!(code, 0);

    let golden = std::fs::read_to_string("tests/golden/phiX174.tabular").unwrap();
    let golden_lines: Vec<&str> = golden.lines().skip(2).collect();
    let output_lines: Vec<&str> = stdout.lines().skip(1).collect();

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
// RBS scoring modes (--rbs-mode)
// ---------------------------------------------------------------------------
#[test]
fn rbs_mode_non_sd_runs_without_error() {
    let (stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/data/small.fasta",
            "--rbs-mode",
            "non-sd",
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(code, 0);
    assert!(stdout.contains("uses_sd: 0"));
}

#[test]
fn rbs_mode_non_sd_sco_includes_motif_column() {
    let (stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/data/small.fasta",
            "--rbs-mode",
            "non-sd",
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(code, 0);
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap();
    let cols: Vec<&str> = data_line.split('\t').collect();
    assert_eq!(cols.len(), 5);
    assert!(!cols[4].is_empty(), "motif column should not be empty");
}

#[test]
fn rbs_mode_sd_runs_without_error() {
    let (stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/data/small.fasta",
            "--rbs-mode",
            "sd",
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(code, 0);
    assert!(stdout.contains("uses_sd: 1"));
}

#[test]
fn default_run_emits_uses_sd_header() {
    let (stdout, _stderr, code) = run(&["-i", "tests/data/small.fasta", "-f", "sco"], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("uses_sd:"));
}

#[test]
fn rbs_mode_rejects_invalid_value() {
    let (_stdout, stderr, code) = run(
        &["-i", "tests/data/small.fasta", "--rbs-mode", "invalid"],
        None,
    );
    assert_ne!(code, 0, "invalid --rbs-mode should fail");
    assert!(
        stderr.contains("invalid value") || stderr.contains("invalid"),
        "error should mention invalid value: {}",
        stderr
    );
}

#[test]
fn old_rbs_flags_are_removed() {
    let (_stdout, _stderr, code1) = run(&["-i", "tests/data/small.fasta", "--non-sd"], None);
    let (_stdout, _stderr, code2) = run(&["-i", "tests/data/small.fasta", "--sd"], None);
    let (_stdout, _stderr, code3) = run(&["-i", "tests/data/small.fasta", "--prodigal-rbs"], None);
    assert_ne!(code1, 0, "--non-sd should no longer exist");
    assert_ne!(code2, 0, "--sd should no longer exist");
    assert_ne!(code3, 0, "--prodigal-rbs should no longer exist");
}

// ---------------------------------------------------------------------------
// Prodigal-style RBS scanner (--rbs-mode prodigal)
// ---------------------------------------------------------------------------
#[test]
fn prodigal_rbs_produces_sco_with_mixed_motifs() {
    let tmpdir = tempfile::tempdir().unwrap();
    let input = tmpdir.path().join("input.fa");
    let output = tmpdir.path().join("out.sco");

    // Synthetic genome with three forward ORFs. The first ORF has a planted
    // AGGAGG Shine-Dalgarno motif; the remaining ORFs fall back to the non-SD
    // motif model under --rbs-mode prodigal. Spacers contain reverse-complement
    // stop codons so the long reverse-strand ORF does not dominate the path.
    let orf = "ATG".to_string() + &"AAACGC".repeat(20) + "TAA";
    let seq =
        "aaaaaaaaaaggaggaaaaa".to_string() + &orf + "TTACTA" + &orf + "TTATGA" + &orf + "TTATAG";
    let fasta = format!(">test\n{}\n", seq);
    std::fs::write(&input, fasta).unwrap();

    let (_stdout, _stderr, code) = run(
        &[
            "-i",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "-f",
            "sco",
            "--rbs-mode",
            "prodigal",
        ],
        None,
    );
    assert_eq!(code, 0, "--rbs-mode prodigal should succeed");

    let text = std::fs::read_to_string(&output).unwrap();
    assert!(text.starts_with("# uses_sd: 1"));
    assert!(text.contains("AGGAGG"), "expected at least one SD motif");
    assert!(
        text.contains("nonSD:"),
        "expected at least one non-SD fallback motif"
    );
}

#[test]
fn prodigal_rbs_preserves_gene_count_on_spv4() {
    // Regression test: the non-SD fallback must not inflate the number of
    // predicted genes.  The Prodigal SD scanner changes motif labels and
    // weights, but the overall gene set should remain comparable.
    let tmpdir = tempfile::tempdir().unwrap();
    let default_out = tmpdir.path().join("default.sco");
    let prodigal_out = tmpdir.path().join("prodigal.sco");

    let (_stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/golden/spv4_NC003438.fa",
            "-o",
            default_out.to_str().unwrap(),
            "-f",
            "sco",
            "-g",
            "4",
        ],
        None,
    );
    assert_eq!(code, 0, "default run should succeed");

    let (_stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/golden/spv4_NC003438.fa",
            "-o",
            prodigal_out.to_str().unwrap(),
            "-f",
            "sco",
            "-g",
            "4",
            "--rbs-mode",
            "prodigal",
        ],
        None,
    );
    assert_eq!(code, 0, "--rbs-mode prodigal run should succeed");

    let default_lines = std::fs::read_to_string(&default_out)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .count();
    let prodigal_lines = std::fs::read_to_string(&prodigal_out)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .count();
    assert_eq!(
        default_lines, prodigal_lines,
        "--rbs-mode prodigal should predict the same number of genes as default mode on SpV4"
    );
}

// ---------------------------------------------------------------------------
// Dicodon scoring
// ---------------------------------------------------------------------------
#[test]
fn dicodon_flag_runs_without_error() {
    let (stdout, _stderr, code) = run(
        &["-i", "tests/data/small.fasta", "--dicodon", "-f", "sco"],
        None,
    );
    assert_eq!(code, 0, "--dicodon run should succeed");
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap_or("");
    assert!(!data_line.is_empty(), "--dicodon should produce data lines");
    let cols: Vec<_> = data_line.split('\t').collect();
    assert_eq!(cols.len(), 5, "SCO line should have 5 columns");
}

#[test]
fn dicodon_changes_output() {
    let (default_out, _, default_code) = run(&["-i", "tests/data/small.fasta", "-f", "sco"], None);
    let (dicodon_out, _, dicodon_code) = run(
        &["-i", "tests/data/small.fasta", "--dicodon", "-f", "sco"],
        None,
    );
    assert_eq!(default_code, 0, "default run should succeed");
    assert_eq!(dicodon_code, 0, "--dicodon run should succeed");
    assert_ne!(
        default_out, dicodon_out,
        "--dicodon should change the annotation output"
    );
}

// ---------------------------------------------------------------------------
// Regression test for internal stop codons with non-standard genetic codes
// ---------------------------------------------------------------------------
#[test]
fn test_no_internal_stops_table4() {
    // This test requires a genome file that lives outside the repo.
    // Skip if the file is not present (e.g. in CI).
    let path = std::path::Path::new("../../test_genomes/MT135298.fasta");
    if !path.exists() {
        eprintln!(
            "Skipping test_no_internal_stops_table4: {} not found",
            path.display()
        );
        return;
    }
    let fasta = std::fs::read_to_string(path).unwrap();
    let (stdout, _stderr, code) = run(
        &["-g", "4", "-a", "/tmp/test_table4_proteins.faa"],
        Some(&fasta),
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

// ---------------------------------------------------------------------------
// Dicodon scoring
// ---------------------------------------------------------------------------
#[test]
fn dicodon_runs_on_genbank() {
    let (stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/golden/NC_001365.gb",
            "-g",
            "4",
            "--dicodon",
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(code, 0, "should exit successfully");
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap();
    assert_eq!(
        data_line.split('\t').count(),
        5,
        "SCO line should have 5 columns"
    );
}

// ---------------------------------------------------------------------------
// ORF scoring model (--model)
// ---------------------------------------------------------------------------

fn write_temp_model(content: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut file, content.as_bytes()).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn model_flag_runs_without_error() {
    let model = r#"{"version":1,"num_features":14,"coeffs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"mean":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"std":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}"#;
    let file = write_temp_model(model);
    let path = file.path().to_str().unwrap();
    let (stdout, _stderr, code) = run(
        &[
            "-i",
            "tests/golden/NC_001365.gb",
            "--model",
            path,
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(code, 0, "--model should exit successfully: {stdout}");
    let data_line = stdout.lines().find(|l| !l.starts_with('#')).unwrap();
    assert_eq!(
        data_line.split('\t').count(),
        5,
        "SCO line should have 5 columns"
    );
}

#[test]
fn model_flag_changes_output() {
    // A model that strongly rewards longer ORFs should change predictions
    // relative to the default PHANOTATE heuristic.
    let model = r#"{"version":1,"num_features":14,"coeffs":[5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"mean":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"std":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}"#;
    let file = write_temp_model(model);
    let path = file.path().to_str().unwrap();
    let (default_out, _, default_code) = run(
        &["-i", "tests/golden/NC_001365.gb", "-g", "4", "-f", "sco"],
        None,
    );
    assert_eq!(default_code, 0);
    let (model_out, _, model_code) = run(
        &[
            "-i",
            "tests/golden/NC_001365.gb",
            "-g",
            "4",
            "--model",
            path,
            "-f",
            "sco",
        ],
        None,
    );
    assert_eq!(model_code, 0);
    assert_ne!(
        default_out.lines().collect::<Vec<_>>(),
        model_out.lines().collect::<Vec<_>>(),
        "--model should change the predicted gene set"
    );
}

#[test]
fn model_flag_rejects_invalid_json() {
    let model = r#"{"version":1,"num_features":14}"#;
    let file = write_temp_model(model);
    let path = file.path().to_str().unwrap();
    let (_stdout, stderr, code) = run(&["-i", "tests/golden/NC_001365.gb", "--model", path], None);
    assert_ne!(code, 0, "invalid model should fail");
    assert!(
        stderr.to_lowercase().contains("model") || stderr.to_lowercase().contains("parse"),
        "error should mention model: {stderr}"
    );
}
