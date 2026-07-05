use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use phanotate_rs::bellman_ford;
use phanotate_rs::codon_table;
use phanotate_rs::detect_table;
use phanotate_rs::gcfp;
use phanotate_rs::genome;
use phanotate_rs::graph;
use phanotate_rs::orf;
use phanotate_rs::output;
use phanotate_rs::overlap_rescue;
use phanotate_rs::rbs_mode::RbsMode;
use phanotate_rs::threshold_calibration::{compute_effective_model_threshold, AutoThresholdMode};

use codon_table::is_supported_table;
use gcfp::GCframe;
use genome::{read_fasta_data, read_genbank, Genome};
use graph::{Graph, Node};
use orf::find_orfs_with_rc;
use output::Format;

#[derive(Parser, Debug)]
#[command(name = "phanotate-rs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "A Gene caller for phage genomes based on PHANOTATE https://github.com/deprekate/PHANOTATE"
)]
struct Cli {
    /// Write protein translations to FILE
    #[arg(short = 'a', value_name = "FILE")]
    protein_out: Option<PathBuf>,

    /// Closed ends: do not allow genes to run off sequence edges
    #[arg(short = 'c')]
    closed_ends: bool,

    /// Treat the input as a circular genome; allow genes to wrap around the origin.
    #[arg(long, conflicts_with_all = ["closed_ends", "export_features"])]
    circular: bool,

    /// Write nucleotide sequences of genes to FILE
    #[arg(short = 'd', value_name = "FILE")]
    nuc_out: Option<PathBuf>,

    /// Output format: gbk, gff, or sco [default: gbk]
    #[arg(short = 'f', value_name = "FORMAT", default_value = "gbk")]
    format: String,

    /// Translation table number [default: 11]
    #[arg(short = 'g', value_name = "TABLE", default_value_t = 11)]
    table: u8,

    /// Input FASTA or GenBank file (default: stdin)
    #[arg(short = 'i', value_name = "FILE")]
    input: Option<PathBuf>,

    /// Treat runs of N as masked sequence; don't build genes across them
    #[arg(short = 'm')]
    mask_n: bool,

    /// Number of threads to use (default: all available)
    #[arg(short = 't', value_name = "N")]
    threads: Option<usize>,

    /// Show a progress bar while processing
    #[arg(long = "progress")]
    progress: bool,

    /// Write primary output to FILE instead of stdout
    #[arg(short = 'o', value_name = "FILE")]
    output: Option<PathBuf>,

    /// Detect the most likely translation table before annotating.
    /// Prints a ranked report and prompts for confirmation unless --yes is also set.
    #[arg(long, default_value_t = false, conflicts_with = "detect_table_batch")]
    detect_table: bool,

    /// Detect the translation table for every record in a multi-FASTA file
    /// and print a TSV summary table.  Does not run annotation.
    #[arg(long, default_value_t = false)]
    detect_table_batch: bool,

    /// Skip the confirmation prompt when used with --detect-table.
    /// Uses the top-ranked table automatically.
    #[arg(long, default_value_t = false)]
    yes: bool,

    /// Export ORF features to FILE and exit without running annotation.
    /// Useful for generating training data.
    #[arg(
        long = "export-features",
        value_name = "FILE",
        conflicts_with_all = ["detect_table", "detect_table_batch"]
    )]
    export_features: Option<PathBuf>,

    /// How to score upstream start-codon motifs.
    #[arg(long, value_enum, default_value = "auto")]
    rbs_mode: RbsMode,

    /// Path to an ONNX ORF scoring model.
    ///
    /// When given, the model replaces the default PHANOTATE heuristic scoring
    /// function and uses a Prodigal-style hexamer cscore log-odds as a
    /// learned feature.
    #[arg(long = "model", value_name = "FILE")]
    model: Option<PathBuf>,

    /// Scale factor applied to the ONNX model's log-odds score before it is
    /// used as a graph edge weight. Values > 1 make the model more decisive;
    /// values < 1 make it more conservative.
    #[arg(long = "model-scale", value_name = "FLOAT", default_value_t = 1.0)]
    model_scale: f64,

    /// Probability threshold that separates ORF "reward" from "penalty".
    /// An ORF whose predicted probability is below this value gets a positive
    /// weight (discouraged); above it gets a negative weight (encouraged).
    /// The default 0.5 is the natural logit decision boundary.
    #[arg(long = "model-threshold", value_name = "FLOAT", default_value_t = 0.5)]
    model_threshold: f64,

    /// Automatically calibrate the model decision threshold per genome.
    #[arg(long = "auto-threshold", value_enum, default_value = "none")]
    auto_threshold: AutoThresholdMode,

    /// Find high-confidence overlapping genes excluded by the primary shortest
    /// path. Requires `--model`.
    #[cfg(feature = "dev")]
    #[arg(long = "find-overlaps", default_value_t = false)]
    find_overlaps: bool,

    /// Per-rescued-gene DP penalty used by overlap rescue.
    #[cfg(feature = "dev")]
    #[arg(long = "overlap-lambda", value_name = "FLOAT", default_value_t = 0.0)]
    overlap_lambda: f64,

    /// Minimum rescue score to include an overlapping ORF.
    #[cfg(feature = "dev")]
    #[arg(
        long = "overlap-threshold",
        value_name = "FLOAT",
        default_value_t = 0.7
    )]
    overlap_threshold: f64,

    /// Weight applied to the overlap-ratio penalty when computing rescue scores.
    #[cfg(feature = "dev")]
    #[arg(
        long = "overlap-penalty-weight",
        value_name = "FLOAT",
        default_value_t = 0.3
    )]
    overlap_penalty_weight: f64,

    /// Minimum ORF length (in bp) to be considered for overlap rescue.
    #[cfg(feature = "dev")]
    #[arg(long = "min-rescue-orf-len", value_name = "N", default_value_t = 90)]
    min_rescue_orf_len: usize,

    /// Write the ORF graph to FILE in Graphviz DOT format, highlighting the
    /// nodes and edges on the shortest-path chosen by the algorithm.
    /// The primary output is still produced normally.
    #[arg(long = "visualize-dag", value_name = "FILE")]
    visualize_dag: Option<PathBuf>,
}

/// Load genomes from file or stdin, auto-detecting format.
fn load_genomes(input: &Option<PathBuf>) -> Result<Vec<Genome>> {
    let data = if let Some(path) = input {
        fs::read_to_string(path)
            .with_context(|| format!("Failed to read input file: {:?}", path))?
    } else {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read from stdin")?;
        buf
    };

    // Auto-detect format
    let trimmed = data.trim_start();
    let is_genbank = trimmed.starts_with("LOCUS");
    let is_fasta = trimmed.starts_with('>');

    if is_genbank {
        read_genbank(&data).context("Failed to parse GenBank input")
    } else if is_fasta {
        read_fasta_data(&data).context("Failed to parse FASTA input")
    } else {
        // Try file extension if available
        if let Some(path) = input {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "gb" | "gbk" => read_genbank(&data).context("Failed to parse GenBank input"),
                _ => read_fasta_data(&data).context("Failed to parse FASTA input"),
            }
        } else {
            anyhow::bail!("Could not auto-detect input format. Expected FASTA (starts with '>') or GenBank (starts with 'LOCUS').")
        }
    }
}

/// Process a single genome through the full PHANOTATE pipeline.
#[allow(clippy::too_many_arguments, unused_variables)]
fn process_genome(
    genome: Genome,
    start_codons_map: &HashMap<Vec<u8>, f64>,
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    format: Format,
    closed_ends: bool,
    mask_n: bool,
    circular: bool,
    table: u8,
    rbs_mode: RbsMode,
    orf_model: Option<&phanotate_rs::onnx_scorer::OnnxScorer>,
    model_scale: f64,
    model_threshold: f64,
    auto_threshold: AutoThresholdMode,
    visualize_dag: Option<&std::path::Path>,
    is_single_input: bool,
    find_overlaps: bool,
    overlap_lambda: f64,
    overlap_threshold: f64,
    overlap_penalty_weight: f64,
    min_rescue_orf_len: usize,
) -> Result<(String, String, String)> {
    let contig_length = genome.seq.len();
    let original_seq = genome.seq.clone();
    let original_rc = genome.rc_seq.clone();

    // For circular genomes, operate on the doubled sequence so ORFs that cross
    // the origin appear as contiguous intervals.
    let (doubled_seq, doubled_rc, _original_len) = if circular {
        genome::circular_sequences(&original_seq, &original_rc)
    } else {
        (
            original_seq.clone(),
            original_rc.clone(),
            original_seq.len(),
        )
    };
    let dna = if circular {
        &doubled_seq
    } else {
        &original_seq
    };
    let rc_dna = if circular { &doubled_rc } else { &original_rc };

    // --- Nucleotide frequencies and GC frame plot ---
    let mut freq = [0usize; 4];
    let mut frame_plot = GCframe::new();

    for &base in dna {
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

        frame_plot.add_base(base);
    }

    let gc_pos_freq = frame_plot.get();

    let total_bases = (contig_length * 2) as f64;
    // freq[0] counts A+T on both strands, freq[2] counts C+G on both strands.
    let pa = freq[0] as f64 / total_bases; // AT fraction
    let pg = freq[2] as f64 / total_bases; // GC fraction
    let pstop = crate::codon_table::genome_wide_pstop(pa, pg, table);

    // --- Find ORFs ---
    let mut orfs = find_orfs_with_rc(
        dna,
        rc_dna,
        start_codons,
        stop_codons,
        90,
        closed_ends,
        mask_n,
        rbs_mode,
    );

    if orfs.is_empty() {
        let no_orfs = format!("#id:\t{} NO ORFS FOUND\n", genome.id);
        return Ok((no_orfs.clone(), no_orfs.clone(), no_orfs));
    }

    // --- Training RBS ---
    let use_non_sd = phanotate_rs::rbs_training::train_rbs_scores(
        &mut orfs,
        dna,
        rc_dna,
        rbs_mode,
        start_codons_map,
    );

    // --- GC frame plot scoring + optional hexamer cscore signal ---
    let annotated: Option<Vec<usize>> = if orf_model.is_some() && !genome.cds.is_empty() {
        Some(
            orfs.iter()
                .enumerate()
                .filter(|(_, o)| {
                    genome.cds.iter().any(|(s, e, strand)| {
                        if *strand > 0 {
                            // Forward: ORF start == CDS start, stop codon ends at CDS end.
                            o.frame > 0 && o.start == *s && o.stop + 2 == *e
                        } else {
                            // Reverse: ORF stop == CDS start, start codon ends at CDS end.
                            o.frame < 0 && o.stop == *s && o.start + 2 == *e
                        }
                    })
                })
                .map(|(i, _)| i)
                .collect(),
        )
    } else {
        None
    };

    phanotate_rs::orf_signals::compute_orf_signals_with_plot(
        &mut orfs,
        dna,
        rc_dna,
        &gc_pos_freq,
        orf_model.is_some(),
        annotated.as_deref(),
        table,
    );

    if orf_model.is_some() {
        phanotate_rs::ml_features::compute_extra_ml_features(&mut orfs, dna, rc_dna, stop_codons);
    }

    // --- Score ORFs ---
    let effective_model_threshold = orf_model.map_or(model_threshold, |m| {
        compute_effective_model_threshold(&orfs, m, contig_length, auto_threshold, model_threshold)
    });
    for orf in &mut orfs {
        orf.score(
            start_codons_map,
            orf_model,
            model_scale,
            effective_model_threshold,
        );
    }

    // --- Build graph ---
    let (gap_scale, overlap_scale) = if orf_model.is_some() {
        phanotate_rs::penalty_calibration::compute_model_penalty_scales(&orfs, pstop)
    } else {
        (1.0, 1.0)
    };
    let (graph, endpoints) = if circular {
        Graph::from_orfs_circular(&orfs, contig_length, pstop, gap_scale, overlap_scale)
    } else {
        Graph::from_orfs(&orfs, contig_length, pstop, gap_scale, overlap_scale)
    };
    let source_idx = endpoints[0];
    let target_idx = endpoints[1];

    // --- Shortest path ---
    let path = bellman_ford::shortest_path(&graph, source_idx, target_idx);
    let _source_idx = endpoints[0];
    let _target_idx = endpoints[1];

    let mut path_edges: Vec<(Node, Node, f64)> = Vec::new();
    if let Some(ref path_indices) = path {
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

    // --- Map circular coordinates back to the original genome ---
    if circular {
        for (left, right, _) in &mut path_edges {
            if left.position > contig_length && left.position != contig_length + 1 {
                left.position -= contig_length;
            }
            if right.position > contig_length && right.position != contig_length + 1 {
                right.position -= contig_length;
            }
        }
        orf::map_orfs_to_circular(&mut orfs, contig_length);
        orf::remove_contained_in_wrapped(&mut orfs, contig_length);
    }

    // --- Visualize DAG ---
    if let Some(dag_base) = visualize_dag {
        let dot_path = if is_single_input {
            dag_base.to_path_buf()
        } else {
            let mut p = dag_base.to_path_buf();
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "dag".to_string());
            let ext = p
                .extension()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "dot".to_string());
            p.set_file_name(format!("{}_{}.{}", stem, genome.id, ext));
            p
        };
        let dot = graph.to_dot(path.as_deref(), &endpoints);
        fs::write(&dot_path, dot)
            .with_context(|| format!("Failed to write DAG visualization to {:?}", dot_path))?;
    }

    // --- Overlap rescue (dev only) ---
    if find_overlaps {
        if let Some(scorer) = orf_model {
            let rescues = overlap_rescue::find_overlapping_genes(
                &path_edges,
                &orfs,
                scorer,
                overlap_threshold,
                overlap_penalty_weight,
                min_rescue_orf_len,
                overlap_lambda,
            );
            for (idx, score) in rescues {
                path_edges.push(overlap_rescue::orf_to_path_edge(&orfs[idx], -score));
            }
        }
    }

    // --- Primary output ---
    let primary = output::write_primary(
        &genome.id,
        &original_seq,
        &path_edges,
        &orfs,
        contig_length,
        format,
        !use_non_sd,
    );

    // --- Protein output ---
    let protein = output::write_protein_fasta(&genome.id, &path_edges, &orfs, table);

    // --- Nucleotide output ---
    let nucleotide = output::write_nucleotide_fasta(&genome.id, &path_edges, &orfs);

    Ok((primary, protein, nucleotide))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate format
    let format = match cli.format.to_lowercase().as_str() {
        "gbk" | "genbank" => Format::Gbk,
        "gff" | "gff3" => Format::Gff,
        "sco" => Format::Sco,
        _ => {
            anyhow::bail!(
                "Invalid output format: '{}'. Supported: gbk, gff, sco",
                cli.format
            );
        }
    };

    // Validate translation table
    if !is_supported_table(cli.table) {
        anyhow::bail!(
            "Invalid translation table: {}. Supported: 1-6, 9-16, 21-31",
            cli.table
        );
    }

    // Set thread pool size if requested
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build thread pool")?;
    }

    // Load genomes
    let genomes = load_genomes(&cli.input)?;
    if genomes.is_empty() {
        anyhow::bail!("No sequences found in input.");
    }

    // --- Export features mode (no annotation) ---
    if let Some(ref features_path) = cli.export_features {
        let stop_codons: Vec<Vec<u8>> = codon_table::stop_codons(cli.table)
            .iter()
            .map(|&c| c.to_vec())
            .collect();
        let start_codons: Vec<Vec<u8>> = match cli.table {
            1 | 11 => vec![b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()],
            _ => codon_table::start_codons(cli.table)
                .iter()
                .map(|&c| c.to_vec())
                .collect(),
        };
        let start_codons_map =
            phanotate_rs::rbs_training::build_start_weights(&start_codons, cli.table);

        let mut file = fs::File::create(features_path)
            .with_context(|| format!("Failed to create features file: {:?}", features_path))?;
        let mut header_written = false;

        for genome in genomes {
            let mut orfs = find_orfs_with_rc(
                &genome.seq,
                &genome.rc_seq,
                &start_codons,
                &stop_codons,
                90,
                cli.closed_ends,
                cli.mask_n,
                cli.rbs_mode,
            );

            // Train RBS scores so that sd_rbs_score / non_sd_rbs_score features
            // are meaningful in exported feature vectors.
            phanotate_rs::rbs_training::train_rbs_scores(
                &mut orfs,
                &genome.seq,
                &genome.rc_seq,
                cli.rbs_mode,
                &start_codons_map,
            );

            // Compute GC-frame hold and a Prodigal-style hexamer cscore
            // so that exported feature vectors are useful for model training.
            let annotated: Option<Vec<usize>> = if genome.cds.is_empty() {
                Some(Vec::new()) // triggers unsupervised hexamer training
            } else {
                Some(
                    orfs.iter()
                        .enumerate()
                        .filter(|(_, o)| {
                            genome.cds.iter().any(|(s, e, strand)| {
                                if *strand > 0 {
                                    o.frame > 0 && o.start == *s && o.stop + 2 == *e
                                } else {
                                    o.frame < 0 && o.stop == *s && o.start + 2 == *e
                                }
                            })
                        })
                        .map(|(i, _)| i)
                        .collect(),
                )
            };

            phanotate_rs::orf_signals::compute_orf_signals(
                &mut orfs,
                &genome.seq,
                &genome.rc_seq,
                true,
                annotated.as_deref(),
                cli.table,
            );

            phanotate_rs::ml_features::compute_extra_ml_features(
                &mut orfs,
                &genome.seq,
                &genome.rc_seq,
                &stop_codons,
            );

            phanotate_rs::ml_features::write_features_tsv(&mut file, &orfs, !header_written, None)
                .context("Failed to write features")?;
            header_written = true;
        }
        eprintln!("Exported features to {:?}", features_path);
        return Ok(());
    }

    // --- Batch table detection (all records, no annotation) ---
    if cli.detect_table_batch {
        let genome_refs: Vec<(String, Vec<u8>)> = genomes
            .iter()
            .map(|g| (g.id.clone(), g.seq.clone()))
            .collect();
        let results = detect_table::detect_tables_batch(&genome_refs, 90);
        let tsv = detect_table::format_batch_tsv(&results);
        print!("{}", tsv);
        return Ok(());
    }

    // --- Table detection (first record only) ---
    let mut effective_table = cli.table;
    if cli.detect_table {
        let first = &genomes[0];
        let scores = detect_table::score_tables(&first.seq, 90);
        if scores.is_empty() {
            eprintln!(
                "Warning: sequence too short ({} nt) for table detection. Using table {}.",
                first.seq.len(),
                cli.table
            );
        } else {
            let report = detect_table::format_report(&scores, &first.id, first.seq.len());
            eprintln!("{}", report);

            let recommended = scores[0].table;

            // Check if we are in a non-interactive environment
            let is_tty = atty::is(atty::Stream::Stdin);
            if cli.yes || !is_tty {
                if !is_tty && !cli.yes {
                    eprintln!(
                        "Warning: stdin is not a TTY. Using recommended table {} automatically.",
                        recommended
                    );
                } else {
                    eprintln!("Using table {} (--yes)", recommended);
                }
                effective_table = recommended;
            } else {
                // interactive mode: prompt the user
                eprint!("Proceed with table {}? [Y/n/number]: ", recommended);
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let trimmed = input.trim();
                effective_table = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("y") {
                    recommended
                } else if let Ok(n) = trimmed.parse::<u8>() {
                    if codon_table::is_supported_table(n) {
                        n
                    } else {
                        eprintln!("Table {} is not supported. Using {}.", n, recommended);
                        recommended
                    }
                } else {
                    eprintln!("Unrecognised input. Using {}.", recommended);
                    recommended
                };
            }
        }
    }

    // Build codon sets from the effective table
    let stop_codons: Vec<Vec<u8>> = codon_table::stop_codons(effective_table)
        .iter()
        .map(|&c| c.to_vec())
        .collect();
    let (start_codons, start_codons_map) = match effective_table {
        1 | 11 => {
            let codons: Vec<Vec<u8>> = vec![b"atg".to_vec(), b"gtg".to_vec(), b"ttg".to_vec()];
            let weights = phanotate_rs::rbs_training::build_start_weights(&codons, effective_table);
            (codons, weights)
        }
        _ => {
            let codons: Vec<Vec<u8>> = codon_table::start_codons(effective_table)
                .iter()
                .map(|&c| c.to_vec())
                .collect();
            let weights = phanotate_rs::rbs_training::build_start_weights(&codons, effective_table);
            (codons, weights)
        }
    };

    // Load learned ORF scoring model if provided.
    let orf_model: Option<phanotate_rs::onnx_scorer::OnnxScorer> = cli
        .model
        .as_ref()
        .map(|p| {
            phanotate_rs::onnx_scorer::OnnxScorer::from_file(p)
                .with_context(|| format!("Failed to load ORF scoring model from {:?}", p))
        })
        .transpose()?;

    // Overlap-rescue tuning parameters (dev feature only).
    #[cfg(feature = "dev")]
    let (
        find_overlaps,
        overlap_lambda,
        overlap_threshold,
        overlap_penalty_weight,
        min_rescue_orf_len,
    ) = (
        cli.find_overlaps,
        cli.overlap_lambda,
        cli.overlap_threshold,
        cli.overlap_penalty_weight,
        cli.min_rescue_orf_len,
    );
    #[cfg(not(feature = "dev"))]
    let (
        find_overlaps,
        overlap_lambda,
        overlap_threshold,
        overlap_penalty_weight,
        min_rescue_orf_len,
    ) = (false, 0.0, 0.7, 0.3, 90_usize);
    if find_overlaps && cli.model.is_none() {
        anyhow::bail!("--find-overlaps requires --model");
    }

    // Process each contig in parallel
    let results: Vec<(String, String, String)> = if cli.progress {
        let pb = ProgressBar::new(genomes.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        let single = genomes.len() == 1;
        genomes
            .into_par_iter()
            .progress_with(pb)
            .map(|genome| {
                process_genome(
                    genome,
                    &start_codons_map,
                    &start_codons,
                    &stop_codons,
                    format,
                    cli.closed_ends,
                    cli.mask_n,
                    cli.circular,
                    effective_table,
                    cli.rbs_mode,
                    orf_model.as_ref(),
                    cli.model_scale,
                    cli.model_threshold,
                    cli.auto_threshold,
                    cli.visualize_dag.as_deref(),
                    single,
                    find_overlaps,
                    overlap_lambda,
                    overlap_threshold,
                    overlap_penalty_weight,
                    min_rescue_orf_len,
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let single = genomes.len() == 1;
        genomes
            .into_par_iter()
            .map(|genome| {
                process_genome(
                    genome,
                    &start_codons_map,
                    &start_codons,
                    &stop_codons,
                    format,
                    cli.closed_ends,
                    cli.mask_n,
                    cli.circular,
                    effective_table,
                    cli.rbs_mode,
                    orf_model.as_ref(),
                    cli.model_scale,
                    cli.model_threshold,
                    cli.auto_threshold,
                    cli.visualize_dag.as_deref(),
                    single,
                    find_overlaps,
                    overlap_lambda,
                    overlap_threshold,
                    overlap_penalty_weight,
                    min_rescue_orf_len,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };

    let primary_output = results
        .iter()
        .map(|(p, _, _)| p.as_str())
        .collect::<String>();
    let protein_output = results
        .iter()
        .map(|(_, pr, _)| pr.as_str())
        .collect::<String>();
    let nuc_output = results
        .iter()
        .map(|(_, _, n)| n.as_str())
        .collect::<String>();

    // Write primary output
    if let Some(path) = cli.output {
        fs::write(&path, &primary_output)
            .with_context(|| format!("Failed to write primary output to {:?}", path))?;
    } else {
        print!("{}", primary_output);
    }

    // Write side outputs if requested
    if let Some(path) = cli.protein_out {
        fs::write(&path, protein_output)
            .with_context(|| format!("Failed to write protein output to {:?}", path))?;
    }
    if let Some(path) = cli.nuc_out {
        fs::write(&path, nuc_output)
            .with_context(|| format!("Failed to write nucleotide output to {:?}", path))?;
    }

    Ok(())
}
