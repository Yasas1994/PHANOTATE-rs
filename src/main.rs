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

use codon_table::is_supported_table;
use gcfp::{max_idx, min_idx, GCframe};
use genome::{read_fasta_data, read_genbank, Genome};
use graph::{Graph, Node};
use orf::{find_orfs_with_rc, score_rbs, Orf};
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
    #[arg(long, default_value_t = false)]
    detect_table: bool,

    /// Detect the translation table for every record in a multi-FASTA file
    /// and print a TSV summary table.  Does not run annotation.
    #[arg(long, default_value_t = false)]
    detect_table_batch: bool,

    /// Skip the confirmation prompt when used with --detect-table.
    /// Uses the top-ranked table automatically.
    #[arg(long, default_value_t = false)]
    yes: bool,
}

/// Build start-codon weights from a list of codons.
/// ATG gets weight 0.85, GTG gets 0.10, TTG gets 0.05, all others get 1.0.
/// Weights are normalised to the maximum.
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
#[allow(clippy::too_many_arguments)]
fn process_genome(
    genome: Genome,
    start_codons_map: &HashMap<Vec<u8>, f64>,
    start_codons: &[Vec<u8>],
    stop_codons: &[Vec<u8>],
    format: Format,
    closed_ends: bool,
    mask_n: bool,
    table: u8,
) -> (String, String, String) {
    let contig_length = genome.seq.len();
    let dna = &genome.seq;

    // --- Nucleotide frequencies and background RBS ---
    let mut freq = [0usize; 4];
    let mut background_rbs = vec![1.0f64; 28];
    let mut frame_plot = GCframe::new();
    let rc_dna = &genome.rc_seq;
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
        let score = score_rbs(window);
        background_rbs[score] += 1.0;

        // Reverse-strand window from pre-computed RC genome
        let rc_start = len.saturating_sub(i + 21);
        let rc_window = &rc_dna[rc_start..len - i];
        let rc_score = score_rbs(rc_window);
        background_rbs[rc_score] += 1.0;

        frame_plot.add_base(base);
    }

    let gc_pos_freq = frame_plot.get();

    let total_bases = (contig_length * 2) as f64;
    let pa = freq[0] as f64 / total_bases;
    let pt = freq[1] as f64 / total_bases;
    let pg = freq[2] as f64 / total_bases;
    let pstop = pt * pa * pa + pt * pg * pa + pt * pa * pg;

    let bg_sum: f64 = background_rbs.iter().sum();
    for v in &mut background_rbs {
        *v /= bg_sum;
    }

    // --- Find ORFs ---
    let mut orfs = find_orfs_with_rc(
        dna,
        &genome.rc_seq,
        start_codons,
        stop_codons,
        90,
        closed_ends,
        mask_n,
    );

    if orfs.is_empty() {
        let no_orfs = format!("#id:\t{} NO ORFS FOUND\n", genome.id);
        return (no_orfs.clone(), no_orfs.clone(), no_orfs);
    }

    // --- Training RBS ---
    let mut training_rbs = vec![1.0f64; 28];
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

    // Compute hold in log-space to avoid underflow on long ORFs.
    // hold = product of pns^(pos_max * pos_min) per codon
    // ln(hold) = sum of ln(pns) * pos_max * pos_min per codon
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

    // --- Primary output ---
    let primary = output::write_primary(&genome.id, dna, &path_edges, &orfs, contig_length, format);

    // --- Protein output ---
    let protein = output::write_protein_fasta(&genome.id, &path_edges, &orfs, table);

    // --- Nucleotide output ---
    let nucleotide = output::write_nucleotide_fasta(&genome.id, &path_edges, &orfs);

    (primary, protein, nucleotide)
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
            "Invalid translation table: {}. Supported: 1, 4, 6, 11, 15, 25",
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
            let weights = build_start_weights(&codons);
            (codons, weights)
        }
        _ => {
            let codons: Vec<Vec<u8>> = codon_table::start_codons(effective_table)
                .iter()
                .map(|&c| c.to_vec())
                .collect();
            let weights = build_start_weights(&codons);
            (codons, weights)
        }
    };

    // Process each contig in parallel
    let results: Vec<(String, String, String)> = if cli.progress {
        let pb = ProgressBar::new(genomes.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
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
                    effective_table,
                )
            })
            .collect()
    } else {
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
                    effective_table,
                )
            })
            .collect()
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
