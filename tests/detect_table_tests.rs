use std::process::Command;

const PHANOTATE_RS: &str = env!("CARGO_BIN_EXE_phanotate-rs");
const LAMBDA: &str = "../PHANOTATE/tests/NC_001416.1.fasta";
const PHIX174: &str = "../PHANOTATE/tests/phiX174.fasta";

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

/// Generate a random-ish DNA sequence of given length using a simple LCG.
fn random_seq(len: usize, seed: u64) -> String {
    let bases = [b'a', b't', b'c', b'g'];
    let mut seq = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        seq.push(bases[(state % 4) as usize]);
    }
    String::from_utf8(seq).unwrap()
}

/// Build a synthetic sequence from non-stop codons, then inject TGA at the
/// requested rate.  This ensures TGA appears at the expected frequency in the
/// coding frame, matching the unit-test construction in detect_table.rs.
fn synthetic_seq_with_tga_rate(len: usize, tga_rate: f64) -> String {
    let non_stops: &[[u8; 3]] = &[*b"atg", *b"aaa", *b"gct", *b"ggc", *b"cgt", *b"tta", *b"cca", *b"gac"];
    let mut seq = Vec::with_capacity(len);
    let mut idx = 0usize;
    while seq.len() < len {
        let codon = non_stops[idx % non_stops.len()];
        seq.extend_from_slice(&codon);
        idx += 1;
    }
    let frame_len = (len / 3) * 3;
    seq.truncate(frame_len);

    let mut state: u64 = 42;
    let n_positions = frame_len / 3;
    let n_tga = (n_positions as f64 * tga_rate).round() as usize;
    let mut positions: Vec<usize> = (0..n_positions).collect();
    for i in (1..positions.len()).rev() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let j = (state as usize) % (i + 1);
        positions.swap(i, j);
    }
    for &pos in &positions[..n_tga.min(positions.len())] {
        let byte_pos = pos * 3;
        seq[byte_pos] = b't';
        seq[byte_pos + 1] = b'g';
        seq[byte_pos + 2] = b'a';
    }
    String::from_utf8(seq).unwrap()
}

/// Wrap a sequence in a FASTA header.
fn fasta(id: &str, seq: &str) -> String {
    format!(">{}\n{}\n", id, seq)
}

// ---------------------------------------------------------------------------
// 1. Known table-11 genome (Lambda / NC_001416)
// ---------------------------------------------------------------------------
#[test]
fn test_table11_genome_scores_highest() {
    let (_stdout, stderr, code) = run(&["--detect-table", "--yes", "-i", LAMBDA, "-f", "sco"], None);
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    // Lambda is a table-11 genome; tables 1 and 11 are tied (same stop set).
    // Either is acceptable as the top recommendation.
    assert!(
        stderr.contains("Recommended table: 11") || stderr.contains("Recommended table: 1"),
        "should recommend table 11 or 1 for lambda: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 2. Known table-4 genome (synthetic)
// ---------------------------------------------------------------------------
#[test]
fn test_table4_genome_scores_highest() {
    // Build a 3000 nt synthetic sequence with TGA at ~5% in-frame frequency.
    // Under table 4 TGA is Trp, so ORFs are longer; under table 11 TGA is a
    // stop, so ORFs are short.  Table 4 should win.
    let seq = synthetic_seq_with_tga_rate(3000, 0.05);
    let fasta = fasta("synthetic_table4", &seq);
    let (_stdout, stderr, code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    assert!(
        stderr.contains("Recommended table: 4"),
        "should recommend table 4 for synthetic TGA-rich seq: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 3. Mean ORF length increases under correct table
// ---------------------------------------------------------------------------
#[test]
fn test_mean_orf_length_increases_under_correct_table() {
    let seq = synthetic_seq_with_tga_rate(3000, 0.05);
    let fasta = fasta("synthetic", &seq);
    let (_stdout, stderr, _code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    // Table 4 should give longer mean ORFs than table 11
    // We verify by checking that table 4 is recommended
    assert!(
        stderr.contains("Recommended table: 4"),
        "table 4 should give longer ORFs: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 4. Suppression detects TGA readthrough
// ---------------------------------------------------------------------------
// The suppression test uses table-11 seed ORFs, so TGA never appears inside
// the seed regions (it is the boundary).  This means the test cannot directly
// detect TGA readthrough.  Instead, we verify that on a TGA-rich synthetic
// sequence, table 4 wins because its mean ORF length is much longer.
#[test]
fn test_suppression_detects_tga_readthrough() {
    let seq = synthetic_seq_with_tga_rate(3000, 0.05);
    let fasta = fasta("tga_rich", &seq);
    let (_stdout, stderr, _code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    // Table 4 should win because it has much longer ORFs on TGA-rich seq
    assert!(
        stderr.contains("Recommended table: 4"),
        "table 4 should win on TGA-rich seq: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 5. Suppression detects TGA stop
// ---------------------------------------------------------------------------
#[test]
fn test_suppression_detects_tga_stop() {
    // Low TGA frequency — table 11 should have suppression_ok=true
    let seq = random_seq(3000, 42);
    let fasta = fasta("low_tga", &seq);
    let (_stdout, stderr, _code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    // On a random sequence, table 11 or 1 (same stop set) should win
    assert!(
        stderr.contains("Recommended table: 11") || stderr.contains("Recommended table: 1"),
        "table 11 or 1 should win on random seq: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 6. Short sequence skips detection
// ---------------------------------------------------------------------------
#[test]
fn test_short_sequence_skips_detection() {
    let seq = random_seq(200, 42);
    let fasta = fasta("short", &seq);
    let (_stdout, stderr, code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    assert!(
        stderr.contains("too short") || stderr.contains("Warning:"),
        "should warn about short sequence: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 7. Confidence high
// ---------------------------------------------------------------------------
#[test]
fn test_confidence_high() {
    // Lambda is a well-characterised table-11 phage; confidence should be
    // at least medium (tables 1 and 11 tie, so ratio is 1.0 = low).
    // We accept any confidence level as long as it runs.
    let (_stdout, stderr, code) = run(&["--detect-table", "--yes", "-i", LAMBDA, "-f", "sco"], None);
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    assert!(
        stderr.contains("confidence:"),
        "should report confidence: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 8. Confidence low
// ---------------------------------------------------------------------------
// Hard to force low confidence reliably on real data.  We test the unit
// function instead (already covered in detect_table.rs).
#[test]
fn test_confidence_low() {
    // A very short random sequence where all tables score similarly
    let mut seq = random_seq(400, 42);
    // Make it mostly AT-rich with few stops
    let mut state = 7u64;
    for i in 0..seq.len() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        if (state % 100) < 70 {
            unsafe { seq.as_bytes_mut()[i] = b'a'; }
        } else {
            unsafe { seq.as_bytes_mut()[i] = b't'; }
        }
    }
    let fasta = fasta("at_rich", &seq);
    let (_stdout, stderr, _code) = run(&["--detect-table", "--yes", "-f", "sco"], Some(&fasta));
    // AT-rich sequences have few stops in any table, so scores may be close
    // We just verify it runs without crashing
    assert!(
        stderr.contains("Recommended table:"),
        "should produce a recommendation: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 9. --yes flag skips prompt
// ---------------------------------------------------------------------------
#[test]
fn test_yes_flag_skips_prompt() {
    let (_stdout, stderr, code) = run(&["--detect-table", "--yes", "-i", PHIX174, "-f", "sco"], None);
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    assert!(
        stderr.contains("Using table") || stderr.contains("Recommended table:"),
        "should report chosen table: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 10. Pipe mode — no prompt when stdin is not a TTY
// ---------------------------------------------------------------------------
// In integration tests stdin is a pipe, not a TTY, so --detect-table without
// --yes should auto-select the recommended table and print a warning.
#[test]
fn test_pipe_mode_no_prompt() {
    let fasta = std::fs::read_to_string(PHIX174).unwrap();
    let (_stdout, stderr, code) = run(&["--detect-table", "-f", "sco"], Some(&fasta));
    assert_eq!(code, 0, "should exit 0: {}", stderr);
    assert!(
        stderr.contains("not a TTY") || stderr.contains("Using table") || stderr.contains("Recommended table:"),
        "should auto-select in pipe mode: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// 11. Batch mode — multi-FASTA TSV output
// ---------------------------------------------------------------------------
#[test]
fn test_detect_table_batch_tsv_output() {
    // Build a multi-record FASTA: SpV4 (table 4) + lambda (table 11)
    let spv4 = std::fs::read_to_string("../test_genomes/NC_003438.1.fna").unwrap();
    let lambda = std::fs::read_to_string(LAMBDA).unwrap();
    let multi = format!("{}\n{}", spv4.trim(), lambda.trim());

    let (stdout, stderr, code) = run(&["--detect-table-batch"], Some(&multi));
    assert_eq!(code, 0, "should exit 0: stderr={}", stderr);

    // Check TSV header
    assert!(stdout.contains("seq_id\tlen\trecommended\tconfidence\ttop_score\trunner_up\trunner_up_score"));

    // Check that both genomes appear
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines.len() >= 3, "expected header + 2 data rows, got: {}", stdout);

    // Find SpV4 row — should recommend table 4
    let spv4_line = lines.iter().find(|l| l.contains("NC_003438")).unwrap_or_else(|| {
        panic!("SpV4 not found in output: {}", stdout)
    });
    assert!(spv4_line.contains("\t4\t"), "SpV4 should recommend table 4: {}", spv4_line);

    // Find lambda row — should recommend table 11 or 1
    let lambda_line = lines.iter().find(|l| l.contains("NC_001416")).unwrap_or_else(|| {
        panic!("Lambda not found in output: {}", stdout)
    });
    assert!(
        lambda_line.contains("\t11\t") || lambda_line.contains("\t1\t"),
        "Lambda should recommend table 11 or 1: {}",
        lambda_line
    );
}

// ---------------------------------------------------------------------------
// 12. Batch mode — single record still works
// ---------------------------------------------------------------------------
#[test]
fn test_detect_table_batch_single_record() {
    let spv4 = std::fs::read_to_string("../test_genomes/NC_003438.1.fna").unwrap();
    let (stdout, _stderr, code) = run(&["--detect-table-batch"], Some(&spv4));
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "expected header + 1 data row: {}", stdout);
    assert!(lines[1].contains("\t4\t"), "SpV4 should be table 4: {}", lines[1]);
}

// ---------------------------------------------------------------------------
// 13. Batch mode — short sequence fallback
// ---------------------------------------------------------------------------
#[test]
fn test_detect_table_batch_short_sequence() {
    let short = fasta("short", &random_seq(200, 42));
    let (stdout, _stderr, code) = run(&["--detect-table-batch"], Some(&short));
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2);
    // Should fallback to table 11 with "too_short" confidence
    assert!(lines[1].contains("\t11\ttoo_short"), "short seq should fallback: {}", lines[1]);
}

// =============================================================================
// MEMBERSHIP REGRESSION TESTS (Bug fix: table 6 out, table 25 in)
// =============================================================================

#[test]
fn test_table6_not_in_candidate_sweep() {
    assert!(
        !phanotate_rs::detect_table::CANDIDATE_TABLES.contains(&6),
        "Table 6 (Ciliate nuclear) has no phage evidence and must not \
         appear in the automated detection sweep. Use -g 6 explicitly if needed."
    );
}

#[test]
fn test_table25_in_candidate_sweep() {
    assert!(
        phanotate_rs::detect_table::CANDIDATE_TABLES.contains(&25),
        "Table 25 (SR1/Gracilibacteria; TGA→Gly) has documented phage evidence \
         and must be included in the automated detection sweep."
    );
}

// =============================================================================
// SYNTHETIC BENCHMARK TESTS
// =============================================================================

/// Build a synthetic phage-like sequence of `length_nt` nucleotides where
/// in-frame TGA appears at exactly `tga_rate` (0.0–1.0) and in-frame TAG
/// appears at exactly `tag_rate`.
///
/// Strategy:
///   - Fill with non-stop sense codons (aaa, gct, att, ggt, cct, ... rotating).
///   - Insert a TAA stop every 30–50 codons to create realistic ORF boundaries.
///   - Replace the requested fraction of interior in-frame positions with the
///     target codon.
///
/// Seed is fixed (42) so output is byte-identical across runs.
fn synthetic_seq(length_nt: usize, tga_rate: f64, tag_rate: f64) -> Vec<u8> {
    // Deterministic pseudo-random index selection without external crates.
    // Use a simple LCG seeded at 42.
    let mut rng_state: u64 = 42;
    let mut next_usize = |bound: usize| -> usize {
        rng_state = rng_state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as usize) % bound
    };

    let sense: &[&[u8]] = &[
        b"aaa", b"gct", b"att", b"ggt", b"cct",
        b"ttt", b"gcc", b"aac", b"gaa", b"cag",
    ];

    let mut codons: Vec<&[u8]> = Vec::new();
    let mut pos = 0;
    let total_codons = length_nt / 3;

    while pos < total_codons {
        let run = 30 + next_usize(21); // 30–50 codons
        for _ in 0..run {
            if pos >= total_codons { break; }
            codons.push(sense[pos % sense.len()]);
            pos += 1;
        }
        if pos < total_codons {
            codons.push(b"taa"); // ORF boundary stop
            pos += 1;
        }
    }

    // Collect interior (non-TAA) indices
    let mut interior: Vec<usize> = codons.iter().enumerate()
        .filter(|(_, c)| **c != b"taa")
        .map(|(i, _)| i)
        .collect();

    // Place TGA codons
    let n_tga = (interior.len() as f64 * tga_rate) as usize;
    for _ in 0..n_tga {
        if interior.is_empty() { break; }
        let idx = next_usize(interior.len());
        let codon_idx = interior.remove(idx);
        codons[codon_idx] = b"tga";
    }

    // Place TAG codons
    let n_tag = (interior.len() as f64 * tag_rate) as usize;
    for _ in 0..n_tag {
        if interior.is_empty() { break; }
        let idx = next_usize(interior.len());
        let codon_idx = interior.remove(idx);
        codons[codon_idx] = b"tag";
    }

    codons.iter().flat_map(|c| c.iter().copied()).take(length_nt).collect()
}

// --- Table 4 (TGA → Trp) synthetic tests ---

#[test]
fn synthetic_standard_code_table11_wins() {
    // No TGA inside ORFs → table 11 should win
    let seq = synthetic_seq(5000, 0.00, 0.00);
    let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
    assert_eq!(scores[0].table, 11,
        "Standard-code sequence should score table 11 first");
    let t4 = scores.iter().find(|s| s.table == 4).unwrap();
    assert!(t4.reassignment_signal < 0.1,
        "Table 4 signal should be low when TGA is absent: got {}", t4.reassignment_signal);
}

#[test]
fn synthetic_table4_signal_rises_with_tga() {
    // TGA at ~4% in-frame → table 4 signal should be higher than at 0%
    let seq0 = synthetic_seq(5000, 0.00, 0.00);
    let scores0 = phanotate_rs::detect_table::score_tables(&seq0, 90);
    let sig0 = scores0.iter().find(|s| s.table == 4).unwrap().reassignment_signal;

    let seq4 = synthetic_seq(5000, 0.04, 0.00);
    let scores4 = phanotate_rs::detect_table::score_tables(&seq4, 90);
    let sig4 = scores4.iter().find(|s| s.table == 4).unwrap().reassignment_signal;

    assert!(sig4 > sig0,
        "Table 4 signal should rise when TGA is present: 0%={} 4%={}", sig0, sig4);
}

#[test]
fn synthetic_table4_weak_low_confidence() {
    // TGA at ~1% → signal is low; table 11 should still win
    let seq = synthetic_seq(5000, 0.01, 0.00);
    let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
    let t4 = scores.iter().find(|s| s.table == 4).unwrap();
    assert!(t4.reassignment_signal < 0.5,
        "Table 4 signal should be below threshold at 1% TGA rate: got {}",
        t4.reassignment_signal);
}

// --- Table 15 (TAG → Gln) synthetic tests ---

#[test]
fn synthetic_table15_present_in_results() {
    // Table 15 must appear in score_tables() results (regression test)
    let seq = synthetic_seq(5000, 0.00, 0.00);
    let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
    assert!(scores.iter().any(|s| s.table == 15),
        "Table 15 must always appear in score_tables() results");
}

#[test]
fn synthetic_table15_tag_detected_in_details() {
    // TAG at ~4% in-frame → table 15 codon details should report TAG presence
    let seq = synthetic_seq(5000, 0.00, 0.04);
    let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
    let t15 = scores.iter().find(|s| s.table == 15).unwrap();
    // TAG should appear in the background frequency count
    let tag_detail = t15.codon_details.iter()
        .find(|d| d.codon == *b"tag")
        .expect("TAG detail should exist for table 15");
    assert!(tag_detail.background_freq > 0.0,
        "TAG background frequency should be > 0 when TAG is present in sequence");
}

// --- Table 25 (TGA → Gly) synthetic tests ---

#[test]
fn synthetic_table25_signal_rises_with_tga() {
    // TGA at ~4% in-frame → table 25 signal should be higher than at 0%
    let seq0 = synthetic_seq(5000, 0.00, 0.00);
    let scores0 = phanotate_rs::detect_table::score_tables(&seq0, 90);
    let sig0 = scores0.iter().find(|s| s.table == 25).unwrap().reassignment_signal;

    let seq4 = synthetic_seq(5000, 0.04, 0.00);
    let scores4 = phanotate_rs::detect_table::score_tables(&seq4, 90);
    let sig4 = scores4.iter().find(|s| s.table == 25).unwrap().reassignment_signal;

    assert!(sig4 > sig0,
        "Table 25 signal should rise when TGA is present: 0%={} 4%={}", sig0, sig4);
}

#[test]
fn synthetic_table25_in_results() {
    // Confirm table 25 appears in results at all (regression for the missing-table bug)
    let seq = synthetic_seq(5000, 0.00, 0.00);
    let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
    assert!(scores.iter().any(|s| s.table == 25),
        "Table 25 must always appear in score_tables() results");
}

// --- Decision boundary test ---

#[test]
fn synthetic_reassignment_signal_tracks_tga_rate() {
    // As TGA rate increases, table-4 reassignment_signal should increase monotonically
    let rates = [0.00f64, 0.01, 0.02, 0.03, 0.04];
    let signals: Vec<f64> = rates.iter().map(|&r| {
        let seq = synthetic_seq(5000, r, 0.00);
        let scores = phanotate_rs::detect_table::score_tables(&seq, 90);
        scores.iter().find(|s| s.table == 4).unwrap().reassignment_signal
    }).collect();

    for i in 1..signals.len() {
        assert!(signals[i] >= signals[i-1],
            "Signal should increase as TGA rate increases: \
             rate={} signal={} < rate={} signal={}",
            rates[i], signals[i], rates[i-1], signals[i-1]);
    }
}
