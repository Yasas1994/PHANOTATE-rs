//! Python bindings for PHANOTATE-rs via PyO3.
//!
//! Exposes:
//!   • `phanotate()`      — run the full gene-calling pipeline on a single genome
//!   • `find_orfs()`      — low-level ORF finder
//!   • `detect_table()`   — translation-table detection for a DNA sequence
//!   • `score_rbs()`      — Shine-Dalgarno scoring
//!   • `translate()`      — translate a DNA sequence using an NCBI table
//!
//! Build with maturin:  `maturin develop` or `maturin build --release`

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::bellman_ford;
use crate::codon_table;
use crate::detect_table;
use crate::gcfp::{max_idx, min_idx, GCframe};
use crate::genome;
use crate::graph::{Graph, Node};
use crate::orf::{self, Orf};
use crate::output::{self, Format};

// ---------------------------------------------------------------------------
// Helper: decide whether a genome uses Shine-Dalgarno RBS signalling.
// Mirrors main.rs.
// ---------------------------------------------------------------------------
fn detect_uses_sd(background: &[f64; 28], training: &[f64; 28]) -> bool {
    let top_bins = [27, 26, 25, 24, 22, 20];
    let signal: f64 = top_bins.iter().map(|&i| training[i] / background[i]).sum();
    signal >= 2.0
}

// ---------------------------------------------------------------------------
// Helper: extract the 21-bp upstream window of an ORF used for RBS scoring.
// Mirrors main.rs.
// ---------------------------------------------------------------------------
fn upstream_window(dna: &[u8], rc_dna: &[u8], orf: &Orf) -> Vec<u8> {
    if orf.frame > 0 {
        crate::rbs_scanner::get_rbs(dna, orf.start)
    } else {
        let rbs_start = dna.len().saturating_sub(orf.start + 21);
        let rbs_end = dna.len().saturating_sub(orf.start);
        rc_dna[rbs_start..rbs_end].to_vec()
    }
}

// ---------------------------------------------------------------------------
// Helper: build start-codon weights (mirrors main.rs)
// ---------------------------------------------------------------------------
fn build_start_weights(start_codons: &[Vec<u8>]) -> HashMap<Vec<u8>, f64> {
    let mut map = HashMap::new();
    for codon in start_codons {
        let w = match codon.as_slice() {
            b"atg" => 0.85,
            b"gtg" => 0.10,
            b"ttg" => 0.05,
            _ => 1.0,
        };
        map.insert(codon.clone(), w);
    }
    let max_w = map.values().cloned().fold(0.0, f64::max);
    if max_w > 0.0 {
        for v in map.values_mut() {
            *v /= max_w;
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Helper: parse output format string
// ---------------------------------------------------------------------------
fn parse_format(format: &str) -> PyResult<Format> {
    match format.to_lowercase().as_str() {
        "gbk" | "genbank" => Ok(Format::Gbk),
        "gff" | "gff3" => Ok(Format::Gff),
        "sco" => Ok(Format::Sco),
        _ => Err(PyValueError::new_err(format!(
            "Invalid output format: '{}'. Supported: gbk, gff, sco",
            format
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helper: validate translation table
// ---------------------------------------------------------------------------
fn validate_table(table: u8) -> PyResult<()> {
    if !codon_table::is_supported_table(table) {
        return Err(PyValueError::new_err(format!(
            "Invalid translation table: {}. Supported: 1, 4, 6, 11, 15, 25",
            table
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: build codon sets from a translation table
// ---------------------------------------------------------------------------
fn build_codon_sets(table: u8) -> (Vec<Vec<u8>>, HashMap<Vec<u8>, f64>, Vec<Vec<u8>>) {
    let stop_codons: Vec<Vec<u8>> = codon_table::stop_codons(table)
        .iter()
        .map(|&c| c.to_vec())
        .collect();

    let (start_codons, start_codons_map) = match table {
        1 | 11 => {
            let codons: Vec<Vec<u8>> = vec![b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()];
            let weights = build_start_weights(&codons);
            (codons, weights)
        }
        _ => {
            let codons: Vec<Vec<u8>> = codon_table::start_codons(table)
                .iter()
                .map(|&c| c.to_vec())
                .collect();
            let weights = build_start_weights(&codons);
            (codons, weights)
        }
    };

    (start_codons, start_codons_map, stop_codons)
}

// ---------------------------------------------------------------------------
// PyOrf — Python-facing wrapper for Orf
// ---------------------------------------------------------------------------
#[pyclass(name = "Orf")]
#[derive(Clone)]
pub struct PyOrf {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub stop: usize,
    #[pyo3(get)]
    pub frame: i8,
    #[pyo3(get)]
    pub rbs_score: usize,
    #[pyo3(get)]
    pub pstop: f64,
    #[pyo3(get)]
    pub weight_rbs: f64,
    #[pyo3(get)]
    pub motif_score: f64,
    #[pyo3(get)]
    pub rbs_motif: Option<String>,
    #[pyo3(get)]
    pub hold: f64,
    #[pyo3(get)]
    pub weight: f64,
    #[pyo3(get)]
    pub start_codon: String,
    #[pyo3(get)]
    pub sequence: String,
}

impl From<&Orf> for PyOrf {
    fn from(orf: &Orf) -> Self {
        PyOrf {
            start: orf.start,
            stop: orf.stop,
            frame: orf.frame,
            rbs_score: orf.rbs_score,
            pstop: orf.pstop,
            weight_rbs: orf.weight_rbs,
            motif_score: orf.motif_score,
            rbs_motif: orf.rbs_motif.clone(),
            hold: orf.hold,
            weight: orf.weight,
            start_codon: String::from_utf8_lossy(orf.start_codon()).to_string(),
            sequence: String::from_utf8_lossy(&orf.seq).to_string(),
        }
    }
}

#[pymethods]
impl PyOrf {
    fn __repr__(&self) -> String {
        format!(
            "Orf(start={}, stop={}, frame={}, start_codon='{}', rbs_score={})",
            self.start, self.stop, self.frame, self.start_codon, self.rbs_score
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// PyGene — Python-facing gene result from the full pipeline
// ---------------------------------------------------------------------------
#[pyclass(name = "Gene")]
#[derive(Clone)]
pub struct PyGene {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub stop: usize,
    #[pyo3(get)]
    pub strand: char,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub start_codon: String,
    #[pyo3(get)]
    pub rbs_motif: Option<String>,
}

#[pymethods]
impl PyGene {
    fn __repr__(&self) -> String {
        format!(
            "Gene(start={}, stop={}, strand='{}', score={:.2e}, start_codon='{}')",
            self.start, self.stop, self.strand, self.score, self.start_codon
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Return an attribute by name, mirroring dict.get for the test suite.
    fn get(&self, key: &str) -> Option<String> {
        match key {
            "start" => Some(self.start.to_string()),
            "stop" => Some(self.stop.to_string()),
            "strand" => Some(self.strand.to_string()),
            "score" => Some(self.score.to_string()),
            "start_codon" => Some(self.start_codon.clone()),
            "rbs_motif" => self.rbs_motif.clone(),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// PyTableScore — Python-facing table detection score
// ---------------------------------------------------------------------------
#[pyclass(name = "TableScore")]
#[derive(Clone)]
pub struct PyTableScore {
    #[pyo3(get)]
    pub table: u8,
    #[pyo3(get)]
    pub mean_orf_len: f64,
    #[pyo3(get)]
    pub mol_ratio: f64,
    #[pyo3(get)]
    pub reassignment_signal: f64,
    #[pyo3(get)]
    pub composite: f64,
}

#[pymethods]
impl PyTableScore {
    fn __repr__(&self) -> String {
        format!(
            "TableScore(table={}, composite={:.2}, mol_ratio={:.2}, signal={:.3})",
            self.table, self.composite, self.mol_ratio, self.reassignment_signal
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// 1. phanotate() — full pipeline for a single genome
// ---------------------------------------------------------------------------

/// Run the full PHANOTATE gene-calling pipeline on a single genome.
///
/// Parameters
/// ----------
/// sequence : str
///     DNA sequence (ACGT, case-insensitive). Can also be a FASTA-formatted
///     string (lines starting with '>').
/// seq_id : str, optional
///     Identifier for the sequence. Required when `sequence` is plain DNA.
///     Ignored when `sequence` is FASTA-formatted (the FASTA header is used).
/// format : str, optional
///     Output format for the primary annotation. One of "gbk", "gff", "sco".
///     Default is "gbk".
/// table : int, optional
///     NCBI translation table number. Default is 11.
///     Supported: 1, 4, 6, 11, 15, 25.
/// closed_ends : bool, optional
///     If True, do not allow genes to run off sequence edges.
///     Default is False.
/// mask_n : bool, optional
///     If True, treat runs of N as masked sequence; don't build genes across them.
///     Default is False.
/// detect_table : bool, optional
///     If True, detect the most likely translation table before annotating
///     and use the top-ranked table. Default is False.
/// non_sd : bool, optional
///     If True, force non-Shine-Dalgarno motif discovery for start-codon
///     scoring. Default is False.
/// sd : bool, optional
///     If True, force Shine-Dalgarno scoring (skip non-SD auto-detection).
///     Default is False. `non_sd` and `sd` are mutually exclusive.
/// prodigal_rbs : bool, optional
///     If True, use Prodigal-style SD scanning with non-SD fallback.
///     Default is False. `prodigal_rbs` is mutually exclusive with `sd`
///     and `non_sd`.
/// dicodon : bool, optional
///     If True, use a Prodigal-style 6-mer dicodon coding-potential model.
///     Default is False.
/// start_model : str, optional
///     Path to a JSON start-site scoring model produced by
///     `scripts/train_start_model.py`. Default is None.
/// min_orf_len : int, optional
///     Minimum ORF length in nucleotides. Default is 90.
///
/// Returns
/// -------
/// dict
///     A dictionary with keys:
///     - "primary"    : str  — annotation in the chosen format
///     - "protein"    : str  — protein FASTA of predicted genes
///     - "nucleotide" : str  — nucleotide FASTA of predicted genes
///     - "genes"      : list[Gene] — structured gene objects
///     - "table_used" : int  — the translation table that was actually used
///     - "uses_sd"    : bool — True when Shine-Dalgarno scoring was used,
///                       False when non-SD motif discovery was used
///
/// Examples
/// --------
/// >>> import phanotate
/// >>> result = phanotate.phanotate("ATG...TAA", seq_id="my_phage")
/// >>> print(result["primary"])
/// >>> print(result["genes"][0].start)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    sequence,
    seq_id = None,
    format = "gbk",
    table = 11,
    closed_ends = false,
    mask_n = false,
    detect_table = false,
    non_sd = false,
    sd = false,
    prodigal_rbs = false,
    dicodon = false,
    start_model = None,
    min_orf_len = 90,
))]
fn phanotate(
    sequence: &str,
    seq_id: Option<&str>,
    format: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    detect_table: bool,
    non_sd: bool,
    sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    start_model: Option<&str>,
    min_orf_len: usize,
) -> PyResult<PyObject> {
    if non_sd && sd {
        return Err(PyValueError::new_err(
            "non_sd and sd are mutually exclusive",
        ));
    }
    if prodigal_rbs && (sd || non_sd) {
        return Err(PyValueError::new_err(
            "prodigal_rbs is mutually exclusive with sd and non_sd",
        ));
    }
    // Parse format
    let out_format = parse_format(format)?;

    // Validate table (will be overridden if detect_table is true)
    validate_table(table)?;

    // Parse input: FASTA or plain sequence
    let (genome_id, dna) = if sequence.trim_start().starts_with('>') {
        let genomes = genome::read_fasta_data(sequence)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse FASTA: {}", e)))?;
        if genomes.is_empty() {
            return Err(PyValueError::new_err("No sequences found in FASTA input."));
        }
        let g = &genomes[0];
        (g.id.clone(), g.seq.clone())
    } else {
        let id = seq_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unnamed".to_string());
        let seq = genome::normalize_seq(sequence);
        (id, seq)
    };

    if dna.is_empty() {
        return Err(PyValueError::new_err("Empty DNA sequence."));
    }

    // Table detection
    let mut effective_table = table;
    if detect_table {
        let scores = detect_table::score_tables(&dna, min_orf_len);
        if !scores.is_empty() {
            effective_table = scores[0].table;
        }
    }

    // Build codon sets
    let (start_codons, start_codons_map, stop_codons) = build_codon_sets(effective_table);

    // Compute reverse complement
    let rc_dna = genome::rev_comp(&dna);

    // --- Run the pipeline ---
    let (primary, protein, nucleotide, genes, uses_sd) = process_single_genome(
        &genome_id,
        &dna,
        &rc_dna,
        &start_codons_map,
        &start_codons,
        &stop_codons,
        out_format,
        closed_ends,
        mask_n,
        effective_table,
        min_orf_len,
        non_sd,
        sd,
        prodigal_rbs,
        dicodon,
        start_model,
    )?;

    // Build Python dict return
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("primary", primary)?;
        dict.set_item("protein", protein)?;
        dict.set_item("nucleotide", nucleotide)?;
        let py_genes: Vec<PyObject> = genes.into_iter().map(|g| g.into_py(py)).collect();
        dict.set_item("genes", py_genes)?;
        dict.set_item("table_used", effective_table)?;
        dict.set_item("uses_sd", uses_sd)?;
        Ok(dict.into())
    })
}

/// Internal: process a single genome through the full PHANOTATE pipeline.
/// Mirrors `process_genome` in main.rs but returns structured data too.
#[allow(clippy::too_many_arguments)]
fn process_single_genome(
    id: &str,
    dna: &[u8],
    rc_dna: &[u8],
    start_codons_map: &HashMap<Vec<u8>, f64>,
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    format: Format,
    closed_ends: bool,
    mask_n: bool,
    table: u8,
    min_orf_len: usize,
    force_non_sd: bool,
    force_sd: bool,
    prodigal_rbs: bool,
    dicodon: bool,
    start_model: Option<&str>,
) -> PyResult<(String, String, String, Vec<PyGene>, bool)> {
    let contig_length = dna.len();

    // --- Nucleotide frequencies and background RBS ---
    let mut freq = [0usize; 4];
    let mut background_rbs = [1.0f64; 28];
    let mut frame_plot = GCframe::new();
    let len = dna.len();

    for i in 0..len {
        let base = dna[i];
        match base {
            b'a' => {
                freq[0] += 1;
                freq[1] += 1;
            }
            b't' => {
                freq[1] += 1;
                freq[0] += 1;
            }
            b'c' => {
                freq[2] += 1;
                freq[3] += 1;
            }
            b'g' => {
                freq[3] += 1;
                freq[2] += 1;
            }
            _ => {}
        }

        let window = if i + 21 <= len {
            &dna[i..i + 21]
        } else {
            &dna[i..]
        };
        let score = if prodigal_rbs {
            crate::rbs_scanner::score_rbs_prodigal(window)
        } else {
            orf::score_rbs(window)
        };
        background_rbs[score] += 1.0;

        let rc_start = len.saturating_sub(i + 21);
        let rc_window = &rc_dna[rc_start..len - i];
        let rc_score = if prodigal_rbs {
            crate::rbs_scanner::score_rbs_prodigal(rc_window)
        } else {
            orf::score_rbs(rc_window)
        };
        background_rbs[rc_score] += 1.0;

        frame_plot.add_base(base);
    }

    let gc_pos_freq = frame_plot.get();

    let total_bases = (contig_length * 2) as f64;
    let pt = freq[1] as f64 / total_bases;
    let pa = freq[0] as f64 / total_bases;
    let pg = freq[3] as f64 / total_bases;
    let pstop = pt * pa * pa + pt * pg * pa + pt * pa * pg;

    let bg_sum: f64 = background_rbs.iter().sum();
    for v in &mut background_rbs {
        *v /= bg_sum;
    }

    // --- Find ORFs ---
    let mut orfs = orf::find_orfs_with_rc(
        dna,
        rc_dna,
        start_codons,
        stop_codons,
        min_orf_len,
        closed_ends,
        mask_n,
    );

    if orfs.is_empty() {
        let no_orfs = format!("#id:\t{} NO ORFS FOUND\n", id);
        return Ok((
            no_orfs.clone(),
            no_orfs.clone(),
            no_orfs,
            Vec::new(),
            force_non_sd,
        ));
    }

    // --- Training RBS ---
    let use_non_sd;
    if prodigal_rbs {
        use_non_sd = false;

        // Train non-SD model once for the genome.
        let non_sd_model =
            crate::nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);

        // Training distribution from actual ORF upstream windows (Prodigal bins).
        let mut training_rbs = [1.0f64; crate::rbs_scanner::NUM_RBS_BINS];
        for orf in &orfs {
            let upstream = upstream_window(dna, rc_dna, orf);
            training_rbs[crate::rbs_scanner::score_rbs_prodigal(&upstream)] += 1.0;
        }
        let tr_sum: f64 = training_rbs.iter().sum();
        for v in &mut training_rbs {
            *v /= tr_sum;
        }

        for orf in &mut orfs {
            let upstream = upstream_window(dna, rc_dna, orf);
            let bin = crate::rbs_scanner::score_rbs_prodigal(&upstream);
            if bin > 0 {
                orf.rbs_score = bin;
                orf.weight_rbs = training_rbs[bin] / background_rbs[bin];
                orf.motif_score = 1.0;
                orf.rbs_motif = crate::rbs_scanner::format_prodigal_rbs_motif(bin);
            } else {
                orf.rbs_score = 0;
                orf.weight_rbs = 1.0;
                orf.motif_score = 1.0;
                // Record the best non-SD motif label for display, but do not
                // let the non-SD score influence the graph path.  The model's
                // absolute scale is not comparable to the per-bin SD weight.
                orf.rbs_motif = non_sd_model
                    .best_motif_label(orf, dna, rc_dna)
                    .map(|m| format!("nonSD:{m}"));
            }
        }
    } else {
        // --- Existing legacy RBS training block ---
        let mut training_rbs = [1.0f64; 28];
        for orf in &orfs {
            training_rbs[orf.rbs_score] += 1.0;
        }
        let tr_sum: f64 = training_rbs.iter().sum();
        for v in &mut training_rbs {
            *v /= tr_sum;
        }
        for orf in &mut orfs {
            orf.weight_rbs = training_rbs[orf.rbs_score] / background_rbs[orf.rbs_score];
        }

        // --- Existing non-SD auto-detect block ---
        use_non_sd = if force_non_sd {
            true
        } else if force_sd {
            false
        } else {
            !detect_uses_sd(&background_rbs, &training_rbs)
        };

        if use_non_sd {
            // Neutralize the SD-based RBS weight so the path is scored purely by
            // the non-SD motif model and start-codon type weights.
            for orf in &mut orfs {
                orf.weight_rbs = 1.0;
            }
            let model = crate::nonsd_motif::NonSdModel::train(&orfs, dna, rc_dna, start_codons_map);
            for orf in &mut orfs {
                orf.motif_score = model.score_orf(orf, dna, rc_dna);
                let (wseq, start) = crate::nonsd_motif::upstream_context(dna, rc_dna, orf);
                let hit = if start >= 18 + crate::nonsd_motif::MIN_MOTIF_LEN {
                    crate::nonsd_motif::find_best_motif(&model.mot_wt, wseq, start, model.no_mot)
                } else {
                    crate::nonsd_motif::MotifHit::default()
                };
                orf.rbs_motif = crate::nonsd_motif::format_motif_hit(&hit);
            }
        }
    }

    // --- GC frame plot scoring ---
    let mut pos_max = [1.0f64; 4];
    let mut pos_min = [1.0f64; 4];

    let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
        std::collections::BTreeMap::new();
    for orf in &orfs {
        by_stop.entry(orf.stop).or_default().push(orf);
    }

    for (_, orfs_at_stop) in by_stop.iter_mut() {
        if orfs_at_stop[0].frame > 0 {
            orfs_at_stop.sort_by_key(|o| o.start);
        } else {
            orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
        }
    }

    for (_, orfs_at_stop) in by_stop {
        let mut selected = None;
        for orf in orfs_at_stop {
            if orf.start_codon() == b"atg" {
                selected = Some(orf);
                break;
            }
        }
        let orf = match selected {
            Some(o) => o,
            None => continue,
        };

        let (start, stop) = (orf.start, orf.stop);
        if start < stop {
            let n = ((stop - start) / 8) * 3;
            let mut base = start + n;
            while base + 36 < stop && base < gc_pos_freq.len() {
                let idx = max_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                pos_max[idx] += 1.0;
                let idx = min_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                pos_min[idx] += 1.0;
                base += 3;
            }
        } else {
            let n = ((start - stop) / 8) * 3;
            let mut base = start.saturating_sub(n);
            while base > stop + 36 && base < gc_pos_freq.len() {
                let idx = max_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                pos_max[idx] += 1.0;
                let idx = min_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                pos_min[idx] += 1.0;
                if base >= 3 {
                    base -= 3;
                } else {
                    break;
                }
            }
        }
    }

    let max_max = pos_max.iter().cloned().fold(0.0, f64::max);
    if max_max > 0.0 {
        for v in &mut pos_max {
            *v /= max_max;
        }
    }
    let max_min = pos_min.iter().cloned().fold(0.0, f64::max);
    if max_min > 0.0 {
        for v in &mut pos_min {
            *v /= max_min;
        }
    }

    for orf in &mut orfs {
        let (start, stop) = (orf.start, orf.stop);
        let ln_pns = (1.0 - orf.pstop).ln();
        let mut log_hold = 0.0f64;
        if orf.frame > 0 {
            let mut base = start;
            while base < stop && base < gc_pos_freq.len() {
                let ind_max = max_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                let ind_min = min_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                log_hold += ln_pns * pos_max[ind_max] * pos_min[ind_min];
                base += 3;
            }
        } else {
            let mut base = start;
            while base > stop && base < gc_pos_freq.len() {
                let ind_max = max_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                let ind_min = min_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                log_hold += ln_pns * pos_max[ind_max] * pos_min[ind_min];
                if base >= 3 {
                    base -= 3;
                } else {
                    break;
                }
            }
        }
        orf.hold = log_hold.exp();
        if !dicodon {
            orf.dicodon_score = 1.0 / orf.hold;
        }
    }

    if dicodon {
        let model = crate::dicodon::DicodonModel::train(&orfs, dna, rc_dna);
        for orf in &mut orfs {
            orf.dicodon_score = model.score_orf(orf);
        }
    }

    if let Some(path) = start_model {
        let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
        for orf in &mut orfs {
            let features = crate::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }

    for orf in &mut orfs {
        orf.score(start_codons_map);
    }

    // --- Build graph ---
    let (graph, endpoints) = Graph::from_orfs(&orfs, contig_length, pstop);
    let source_idx = endpoints[0];
    let target_idx = endpoints[1];

    // --- Shortest path ---
    let path = bellman_ford::shortest_path(&graph, source_idx, target_idx);

    let mut path_edges: Vec<(Node, Node, f64)> = Vec::new();
    if let Some(path_indices) = path {
        for i in 0..path_indices.len() - 1 {
            let u = path_indices[i];
            let v = path_indices[i + 1];
            let left = &graph.nodes[u];
            let right = &graph.nodes[v];

            let weight = if left.gene == "CDS" && right.gene == "CDS" {
                if left.node_type == "start" && right.node_type == "stop" && left.frame > 0 {
                    orfs.iter()
                        .find(|o| {
                            o.start == left.position
                                && o.stop == right.position
                                && o.frame == left.frame
                        })
                        .map(|o| o.weight)
                        .unwrap_or(0.0)
                } else if left.node_type == "stop" && right.node_type == "start" && left.frame < 0 {
                    orfs.iter()
                        .find(|o| {
                            o.stop == left.position
                                && o.start == right.position
                                && o.frame == left.frame
                        })
                        .map(|o| o.weight)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            path_edges.push((*left, *right, weight));
        }
    }

    // --- Collect structured gene results ---
    let mut genes = Vec::new();
    for (left, right, weight) in &path_edges {
        if left.gene != "CDS" || right.gene != "CDS" {
            continue;
        }
        let (start, stop, strand) = if left.node_type == "start" && right.node_type == "stop" {
            (left.position, right.position + 2, '+')
        } else if left.node_type == "stop" && right.node_type == "start" {
            (right.position + 2, left.position, '-')
        } else {
            continue;
        };

        let (start_codon, rbs_motif) = if strand == '+' {
            orfs.iter()
                .find(|o| o.start == left.position && o.stop == right.position && o.frame > 0)
                .map(|o| {
                    (
                        String::from_utf8_lossy(o.start_codon()).to_string(),
                        o.rbs_motif.clone(),
                    )
                })
                .unwrap_or_default()
        } else {
            orfs.iter()
                .find(|o| o.stop == left.position && o.start == right.position && o.frame < 0)
                .map(|o| {
                    (
                        String::from_utf8_lossy(o.start_codon()).to_string(),
                        o.rbs_motif.clone(),
                    )
                })
                .unwrap_or_default()
        };

        genes.push(PyGene {
            start,
            stop,
            strand,
            score: *weight,
            start_codon,
            rbs_motif,
        });
    }

    // --- Primary output ---
    let primary = output::write_primary(
        id,
        dna,
        &path_edges,
        &orfs,
        contig_length,
        format,
        !use_non_sd,
    );

    // --- Protein output ---
    let protein = output::write_protein_fasta(id, &path_edges, &orfs, table);

    // --- Nucleotide output ---
    let nucleotide = output::write_nucleotide_fasta(id, &path_edges, &orfs);

    Ok((primary, protein, nucleotide, genes, !use_non_sd))
}

// ---------------------------------------------------------------------------
// 2. find_orfs() — low-level ORF finder
// ---------------------------------------------------------------------------

/// Find all ORFs in a DNA sequence.
///
/// Parameters
/// ----------
/// sequence : str
///     DNA sequence (ACGT, case-insensitive).
/// table : int, optional
///     NCBI translation table number. Default is 11.
/// closed_ends : bool, optional
///     If True, do not allow ORFs to run off sequence edges.
///     Default is False.
/// mask_n : bool, optional
///     If True, do not build any ORF that spans a run of N nucleotides.
///     Default is False.
/// min_orf_len : int, optional
///     Minimum ORF length in nucleotides. Default is 90.
/// prodigal_rbs : bool, optional
///     If True, score upstream RBS motifs with Prodigal-style SD scanning
///     and fall back to non-SD motifs for ORFs without a strong SD signal.
///     Default is False.
/// start_model : str, optional
///     Path to a JSON start-site scoring model produced by
///     `scripts/train_start_model.py`. Default is None.
///
/// Returns
/// -------
/// list[Orf]
///     A list of Orf objects, each with attributes:
///     start, stop, frame, rbs_score, pstop, weight_rbs, motif_score,
///     rbs_motif, hold, weight, start_codon, sequence.
///
/// Examples
/// --------
/// >>> import phanotate
/// >>> orfs = phanotate.find_orfs("ATG...TAA")
/// >>> print(orfs[0].start, orfs[0].stop, orfs[0].frame)
#[pyfunction]
#[pyo3(signature = (
    sequence,
    table = 11,
    closed_ends = false,
    mask_n = false,
    min_orf_len = 90,
    prodigal_rbs = false,
    start_model = None,
))]
fn find_orfs(
    sequence: &str,
    table: u8,
    closed_ends: bool,
    mask_n: bool,
    min_orf_len: usize,
    prodigal_rbs: bool,
    start_model: Option<&str>,
) -> PyResult<Vec<PyOrf>> {
    validate_table(table)?;

    let dna = genome::normalize_seq(sequence);
    if dna.is_empty() {
        return Err(PyValueError::new_err("Empty DNA sequence."));
    }

    let stop_codons: Vec<Vec<u8>> = codon_table::stop_codons(table)
        .iter()
        .map(|&c| c.to_vec())
        .collect();
    let start_codons: Vec<Vec<u8>> = codon_table::start_codons(table)
        .iter()
        .map(|&c| c.to_vec())
        .collect();
    let start_codons_map = build_start_weights(&start_codons);

    let rc_dna = genome::rev_comp(&dna);
    let mut orfs = orf::find_orfs_with_rc(
        &dna,
        &rc_dna,
        &start_codons,
        &stop_codons,
        min_orf_len,
        closed_ends,
        mask_n,
    );

    if prodigal_rbs {
        // Background distribution over all 21-nt windows.
        let mut background_rbs = [1.0f64; crate::rbs_scanner::NUM_RBS_BINS];
        let len = dna.len();
        for i in 0..len {
            let window = if i + 21 <= len {
                &dna[i..i + 21]
            } else {
                &dna[i..]
            };
            background_rbs[crate::rbs_scanner::score_rbs_prodigal(window)] += 1.0;

            let rc_start = len.saturating_sub(i + 21);
            let rc_window = &rc_dna[rc_start..len - i];
            background_rbs[crate::rbs_scanner::score_rbs_prodigal(rc_window)] += 1.0;
        }
        let bg_sum: f64 = background_rbs.iter().sum();
        for v in &mut background_rbs {
            *v /= bg_sum;
        }

        // Train non-SD model for the fallback branch.
        let non_sd_model =
            crate::nonsd_motif::NonSdModel::train(&orfs, &dna, &rc_dna, &start_codons_map);

        // Training distribution from actual ORF upstream windows.
        let mut training_rbs = [1.0f64; crate::rbs_scanner::NUM_RBS_BINS];
        for orf in &orfs {
            let upstream = upstream_window(&dna, &rc_dna, orf);
            training_rbs[crate::rbs_scanner::score_rbs_prodigal(&upstream)] += 1.0;
        }
        let tr_sum: f64 = training_rbs.iter().sum();
        for v in &mut training_rbs {
            *v /= tr_sum;
        }

        for orf in &mut orfs {
            let upstream = upstream_window(&dna, &rc_dna, orf);
            let bin = crate::rbs_scanner::score_rbs_prodigal(&upstream);
            if bin > 0 {
                orf.rbs_score = bin;
                orf.weight_rbs = training_rbs[bin] / background_rbs[bin];
                orf.motif_score = 1.0;
                orf.rbs_motif = crate::rbs_scanner::format_prodigal_rbs_motif(bin);
            } else {
                orf.rbs_score = 0;
                orf.weight_rbs = 1.0;
                orf.motif_score = 1.0;
                // Record the best non-SD motif label for display only.
                orf.rbs_motif = non_sd_model
                    .best_motif_label(orf, &dna, &rc_dna)
                    .map(|m| format!("nonSD:{m}"));
            }
        }
    }

    if let Some(path) = start_model {
        let model = crate::start_refiner::StartModel::from_json(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load start model: {}", e)))?;
        for orf in &mut orfs {
            let features = crate::start_refiner::StartSiteFeatures::new(orf);
            orf.start_score = model.score(&features);
        }
    }

    Ok(orfs.iter().map(|o| PyOrf::from(o)).collect())
}

// ---------------------------------------------------------------------------
// 3. detect_table() — translation table detection
// ---------------------------------------------------------------------------

/// Detect the most likely NCBI translation table for a DNA sequence.
///
/// Parameters
/// ----------
/// sequence : str
///     DNA sequence (ACGT, case-insensitive).
/// min_orf_len : int, optional
///     Minimum ORF length for the detection heuristic. Default is 90.
///
/// Returns
/// -------
/// list[TableScore]
///     A list of TableScore objects, sorted by composite score (best first).
///     Each object has attributes: table, mean_orf_len, mol_ratio,
///     reassignment_signal, composite.
///
/// Examples
/// --------
/// >>> import phanotate
/// >>> scores = phanotate.detect_table("ATG...TAA")
/// >>> print(scores[0].table, scores[0].composite)
#[pyfunction]
#[pyo3(signature = (sequence, min_orf_len = 90))]
#[pyo3(name = "detect_table")]
fn detect_table_py(sequence: &str, min_orf_len: usize) -> PyResult<Vec<PyTableScore>> {
    let dna = genome::normalize_seq(sequence);
    if dna.is_empty() {
        return Err(PyValueError::new_err("Empty DNA sequence."));
    }

    let scores = detect_table::score_tables(&dna, min_orf_len);

    Ok(scores
        .into_iter()
        .map(|s| PyTableScore {
            table: s.table,
            mean_orf_len: s.mean_orf_len,
            mol_ratio: s.mol_ratio,
            reassignment_signal: s.reassignment_signal,
            composite: s.composite,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// 4. score_rbs() — Shine-Dalgarno scoring
// ---------------------------------------------------------------------------

/// Score a Shine-Dalgarno sequence.
///
/// Parameters
/// ----------
/// sequence : str
///     A DNA sequence (typically the 21 nt upstream of a start codon).
///     The function reverses the sequence internally before scoring,
///     matching the PHANOTATE reference implementation.
///
/// Returns
/// -------
/// int
///     RBS score (0–27). Higher is stronger.
///
/// Examples
/// --------
/// >>> import phanotate
/// >>> phanotate.score_rbs("aaggaggtgagtaacaaaacc")
#[pyfunction]
fn score_rbs(sequence: &str) -> PyResult<usize> {
    let seq = genome::normalize_seq(sequence);
    Ok(orf::score_rbs(&seq))
}

// ---------------------------------------------------------------------------
// 5. translate() — DNA → protein
// ---------------------------------------------------------------------------

/// Translate a DNA sequence into protein using an NCBI translation table.
///
/// Parameters
/// ----------
/// sequence : str
///     DNA sequence (ACGT, case-insensitive). Length should be a multiple of 3.
/// table : int, optional
///     NCBI translation table number. Default is 11.
///     Supported: 1, 4, 6, 11, 15, 25.
///
/// Returns
/// -------
/// str
///     Protein sequence. '*' denotes a stop codon, 'X' denotes an
///     unknown/incomplete codon.
///
/// Examples
/// --------
/// >>> import phanotate
/// >>> phanotate.translate("atgtggtaa")
/// 'MW*'
#[pyfunction]
#[pyo3(signature = (sequence, table = 11))]
fn translate(sequence: &str, table: u8) -> PyResult<String> {
    validate_table(table)?;
    let seq = genome::normalize_seq(sequence);
    codon_table::translate(&seq, table).map_err(|e| PyValueError::new_err(e))
}

// ---------------------------------------------------------------------------
// 6. Utility: get supported tables, stop codons, start codons
// ---------------------------------------------------------------------------

/// Return the list of supported NCBI translation table numbers.
///
/// Returns
/// -------
/// list[int]
///     Supported table numbers: [1, 4, 6, 11, 15, 25].
#[pyfunction]
fn supported_tables() -> Vec<u8> {
    vec![1, 4, 6, 11, 15, 25]
}

/// Return the stop codons for a given translation table.
///
/// Parameters
/// ----------
/// table : int
///     NCBI translation table number.
///
/// Returns
/// -------
/// list[str]
///     List of stop codons (lowercase).
#[pyfunction]
fn stop_codons(table: u8) -> PyResult<Vec<String>> {
    validate_table(table)?;
    Ok(codon_table::stop_codons(table)
        .iter()
        .map(|&c| String::from_utf8_lossy(c).to_string())
        .collect())
}

/// Return the start codons for a given translation table.
///
/// Parameters
/// ----------
/// table : int
///     NCBI translation table number.
///
/// Returns
/// -------
/// list[str]
///     List of start codons (lowercase).
#[pyfunction]
fn start_codons(table: u8) -> PyResult<Vec<String>> {
    validate_table(table)?;
    Ok(codon_table::start_codons(table)
        .iter()
        .map(|&c| String::from_utf8_lossy(c).to_string())
        .collect())
}

/// Return the canonical NCBI name for a translation table.
///
/// Parameters
/// ----------
/// table : int
///     NCBI translation table number.
///
/// Returns
/// -------
/// str
///     Human-readable name of the table.
#[pyfunction]
fn table_name(table: u8) -> PyResult<String> {
    validate_table(table)?;
    Ok(codon_table::table_name(table).to_string())
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// PHANOTATE-rs — fast gene caller for phage genomes.
///
/// This module exposes the core PHANOTATE algorithms to Python:
///
/// • `phanotate()`      — full gene-calling pipeline
/// • `find_orfs()`      — low-level ORF enumeration
/// • `detect_table()`   — automatic translation-table detection
/// • `score_rbs()`      — Shine-Dalgarno scoring
/// • `translate()`      — DNA → protein translation
/// • `supported_tables()` — list supported NCBI tables
/// • `stop_codons()`    — stop codons for a table
/// • `start_codons()`   — start codons for a table
/// • `table_name()`     — human-readable table name
///
/// Data classes:
/// • `Orf`        — open reading frame
/// • `Gene`       — predicted gene from the full pipeline
/// • `TableScore` — table detection result
#[pymodule]
fn phanotate_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phanotate, m)?)?;
    m.add_function(wrap_pyfunction!(find_orfs, m)?)?;
    m.add_function(wrap_pyfunction!(detect_table_py, m)?)?;
    // Note: detect_table_py is registered as "detect_table" via #[pyo3(name = "detect_table")]
    m.add_function(wrap_pyfunction!(score_rbs, m)?)?;
    m.add_function(wrap_pyfunction!(translate, m)?)?;
    m.add_function(wrap_pyfunction!(supported_tables, m)?)?;
    m.add_function(wrap_pyfunction!(stop_codons, m)?)?;
    m.add_function(wrap_pyfunction!(start_codons, m)?)?;
    m.add_function(wrap_pyfunction!(table_name, m)?)?;

    m.add_class::<PyOrf>()?;
    m.add_class::<PyGene>()?;
    m.add_class::<PyTableScore>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper tests (pure Rust, no PyO3)
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_start_weights() {
        let codons = vec![b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()];
        let weights = build_start_weights(&codons);
        assert_eq!(weights.len(), 3);
        // ATG should have highest weight (0.85 normalized)
        assert!(weights[&b"atg".to_vec()] >= weights[&b"gtg".to_vec()]);
        assert!(weights[&b"gtg".to_vec()] >= weights[&b"ttg".to_vec()]);
    }

    #[test]
    fn test_parse_format_gbk() {
        assert!(matches!(parse_format("gbk").unwrap(), Format::Gbk));
        assert!(matches!(parse_format("genbank").unwrap(), Format::Gbk));
    }

    #[test]
    fn test_parse_format_gff() {
        assert!(matches!(parse_format("gff").unwrap(), Format::Gff));
        assert!(matches!(parse_format("gff3").unwrap(), Format::Gff));
    }

    #[test]
    fn test_parse_format_sco() {
        assert!(matches!(parse_format("sco").unwrap(), Format::Sco));
    }

    #[test]
    fn test_parse_format_invalid() {
        assert!(parse_format("xyz").is_err());
    }

    #[test]
    fn test_validate_table_valid() {
        for table in [1, 4, 6, 11, 15, 25] {
            assert!(validate_table(table).is_ok());
        }
    }

    #[test]
    fn test_validate_table_invalid() {
        assert!(validate_table(99).is_err());
        assert!(validate_table(0).is_err());
    }

    #[test]
    fn test_build_codon_sets_table11() {
        let (starts, weights, stops) = build_codon_sets(11);
        assert!(!starts.is_empty());
        assert!(!stops.is_empty());
        assert_eq!(stops.len(), 3); // taa, tag, tga
        assert!(weights.contains_key(&b"atg".to_vec()));
    }

    #[test]
    fn test_build_codon_sets_table4() {
        let (_starts, weights, stops) = build_codon_sets(4);
        assert_eq!(stops.len(), 2); // taa, tag (no tga)
        assert!(!weights.is_empty());
    }

    // -----------------------------------------------------------------------
    // PyOrf conversion tests (pure Rust, no PyO3)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pyorf_from_orf() {
        let hold = 1.0;
        let orf = Orf {
            start: 1,
            stop: 30,
            frame: 1,
            seq: b"atg".to_vec(),
            rbs_score: 10,
            pstop: 0.05,
            weight_rbs: 1.0,
            motif_score: 1.0,
            rbs_motif: None,
            hold,
            dicodon_score: 1.0 / hold,
            start_score: 1.0,
            weight: -1.0,
        };
        let py_orf = PyOrf::from(&orf);
        assert_eq!(py_orf.start, 1);
        assert_eq!(py_orf.stop, 30);
        assert_eq!(py_orf.frame, 1);
        assert_eq!(py_orf.start_codon, "atg");
        assert_eq!(py_orf.sequence, "atg");
    }
}
