use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

mod bellman_ford;
mod codon_table;
mod gcfp;
mod genome;
mod graph;
mod orf;
mod output;
mod weights;

use genome::{read_fasta, read_genbank, rev_comp, Genome};
use gcfp::{max_idx, min_idx, GCframe};
use graph::{Graph, Node};
use orf::{find_orfs, score_rbs, Orf};
use output::Format;

#[derive(Parser, Debug)]
#[command(name = "phanotate-rs")]
#[command(about = "Gene caller for phage genomes")]
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
}

fn parse_start_codons(s: &str) -> HashMap<Vec<u8>, f64> {
    let mut map = HashMap::new();
    for part in s.split(',') {
        let mut kv = part.split(':');
        let codon = kv
            .next()
            .unwrap()
            .trim()
            .to_ascii_lowercase()
            .as_bytes()
            .to_vec();
        let weight: f64 = kv.next().unwrap_or("1.0").trim().parse().unwrap_or(1.0);
        map.insert(codon, weight);
    }
    let max_w = map.values().cloned().fold(0.0, f64::max);
    if max_w > 0.0 {
        for v in map.values_mut() {
            *v /= max_w;
        }
    }
    map
}

fn parse_stop_codons(s: &str) -> Vec<Vec<u8>> {
    s.split(',')
        .map(|p| p.trim().to_ascii_lowercase().as_bytes().to_vec())
        .collect()
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

/// Parse FASTA data from a string.
fn read_fasta_data(data: &str) -> Result<Vec<Genome>> {
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
                genomes.push(Genome {
                    id: current_id.clone(),
                    seq: normalize_seq(&current_seq),
                });
            }
            current_id = line.split_whitespace().next().unwrap_or("").to_string();
            current_seq.clear();
        } else {
            current_seq.push_str(line);
        }
    }

    if !current_id.is_empty() {
        genomes.push(Genome {
            id: current_id,
            seq: normalize_seq(&current_seq),
        });
    }

    Ok(genomes)
}

fn normalize_seq(seq: &str) -> Vec<u8> {
    seq.bytes()
        .map(|b| {
            let b = b.to_ascii_lowercase();
            match b {
                b's' | b'b' | b'v' => b'g',
                b'a' | b'c' | b't' | b'g' => b,
                _ => b'a',
            }
        })
        .collect()
}

/// Process a single genome through the full PHANOTATE pipeline.
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

    for i in 0..dna.len() {
        let base = dna[i];
        match base {
            b'a' => freq[0] += 1,
            b't' => freq[1] += 1,
            b'c' => freq[2] += 1,
            b'g' => freq[3] += 1,
            _ => {}
        }
        match rev_comp(&[base])[0] {
            b'a' => freq[0] += 1,
            b't' => freq[1] += 1,
            b'c' => freq[2] += 1,
            b'g' => freq[3] += 1,
            _ => {}
        }

        let window = if i + 21 <= dna.len() {
            &dna[i..i + 21]
        } else {
            &dna[i..]
        };
        let score = score_rbs(window);
        background_rbs[score] += 1.0;
        let rc_window = rev_comp(window);
        let rc_score = score_rbs(&rc_window);
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
    let mut orfs = find_orfs(dna, start_codons, stop_codons, 90, closed_ends, mask_n);

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

    let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> = std::collections::BTreeMap::new();
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
                let pns = 1.0 - orf.pstop;
                orf.hold *= pns.powf(pos_max[ind_max]).powf(pos_min[ind_min]);
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
                let pns = 1.0 - orf.pstop;
                orf.hold *= pns.powf(pos_max[ind_max]).powf(pos_min[ind_min]);
                if base >= 3 {
                    base -= 3;
                } else {
                    break;
                }
            }
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
                        .find(|o| o.start == left.position && o.stop == right.position && o.frame == left.frame)
                        .map(|o| o.weight)
                        .unwrap_or(0.0)
                } else if left.node_type == "stop" && right.node_type == "start" && left.frame < 0 {
                    orfs.iter()
                        .find(|o| o.stop == left.position && o.start == right.position && o.frame == left.frame)
                        .map(|o| o.weight)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            path_edges.push((left.clone(), right.clone(), weight));
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
    if cli.table != 1 && cli.table != 11 {
        anyhow::bail!(
            "Invalid translation table: {}. Supported: 1, 11",
            cli.table
        );
    }

    // Load genomes
    let genomes = load_genomes(&cli.input)?;
    if genomes.is_empty() {
        anyhow::bail!("No sequences found in input.");
    }

    let start_codons_map = parse_start_codons("atg:0.85,gtg:0.10,ttg:0.05");
    let start_codons: Vec<Vec<u8>> = start_codons_map.keys().cloned().collect();
    let stop_codons = parse_stop_codons("tag,tga,taa");

    // Process each contig in parallel
    let results: Vec<(String, String, String)> = genomes
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
                cli.table,
            )
        })
        .collect();

    let primary_output = results.iter().map(|(p, _, _)| p.as_str()).collect::<String>();
    let protein_output = results.iter().map(|(_, pr, _)| pr.as_str()).collect::<String>();
    let nuc_output = results.iter().map(|(_, _, n)| n.as_str()).collect::<String>();

    // Write primary output to stdout
    print!("{}", primary_output);

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
