# PHANOTATE-rs

A fast Rust reimplementation of **PHANOTATE** — a gene caller optimized for bacteriophage genomes.

[![Rust](https://shields.io/badge/-Rust-3776AB?style=flat&logo=rust)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

> **Note:** This is a Rust port of the original Python implementation. If you use this tool in your research, please cite the original PHANOTATE publication (see [Citation](#citation) below).

---

## Features

- **Gene calling for phage genomes** — Optimised for the compact, overlapping gene organisation typical of bacteriophages
- **Multiple output formats** — GenBank (`gbk`), GFF3 (`gff`), and simple coordinate (`sco`) output
- **Five NCBI translation tables** — Supports tables 1, 4, 11, 15, and 25 for phage-relevant genetic codes
- **Automatic translation table detection** (`--detect-table`) — Detects the most likely genetic code from the sequence itself before annotation
- **Multi-threaded** — Process multiple contigs in parallel via Rayon
- **Fast** — Significantly faster than the Python reference implementation (see [Benchmarks](#benchmarks))

---

## Installation

### One-liner (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/Yasas1994/PHANOTATE-rs/main/install.sh | bash
```

The script auto-detects your platform, downloads the latest release, and installs
to `~/.local/bin` (or `/usr/local/bin` if writable).

### Homebrew (macOS / Linux)

```bash
brew tap deprekate/bioinformatics
brew install phanotate-rs
```

### Bioconda

```bash
conda install -c bioconda phanotate-rs
# or
mamba install -c bioconda phanotate-rs
# or
pixi add phanotate-rs
```

### Cargo (Rust toolchain required)

```bash
cargo install phanotate-rs
```

### From source

Requires [Rust](https://www.rust-lang.org/tools/install) (1.70+):

```bash
git clone https://github.com/Yasas1994/PHANOTATE-rs.git
cd phanotate-rs
cargo build --release
```

The binary will be at `./target/release/phanotate-rs`.

### Pre-built binaries

Pre-built binaries for Linux (x86_64, aarch64, musl), macOS (Intel, Apple Silicon),
and Windows (x86_64) are available from the
[Releases](https://github.com/Yasas1994/PHANOTATE-rs/releases) page.

---

## Quick Start

```bash
# Annotate a phage genome (default: GenBank output, table 11)
phanotate-rs -i genome.fasta > genome.gbk

# GFF3 output
phanotate-rs -i genome.fasta -f gff > genome.gff

# Simple coordinate output
phanotate-rs -i genome.fasta -f sco > genome.sco

# Detect the genetic code automatically
phanotate-rs --detect-table --yes -i genome.fasta -f sco

# Specify a translation table (e.g. Mycoplasma phage — table 4)
phanotate-rs -i genome.fasta -g 4 -f sco

# Write protein and nucleotide sequences to separate files
phanotate-rs -i genome.fasta -a proteins.faa -d nucleotides.fna
```

### Stdin input

```bash
cat genome.fasta | phanotate-rs -f sco
```

---

## Usage

```
Usage: phanotate-rs [OPTIONS]

Options:
  -a <FILE>           Write protein translations to FILE
  -c                  Closed ends: do not allow genes to run off sequence edges
  -d <FILE>           Write nucleotide sequences of genes to FILE
  -f <FORMAT>         Output format: gbk, gff, or sco [default: gbk]
  -g <TABLE>          Translation table number [default: 11]
  -i <FILE>           Input FASTA or GenBank file (default: stdin)
  -m                  Treat runs of N as masked sequence
  -t <N>              Number of threads to use
      --progress      Show a progress bar
  -o <FILE>           Write primary output to FILE instead of stdout
      --detect-table       Detect the most likely translation table
      --detect-table-batch Detect tables for all records, print TSV summary
      --yes                Skip the confirmation prompt with --detect-table
  -h, --help          Print help
```

### Translation tables

| Table | Name | Use case |
|-------|------|----------|
| 1 | Standard | Baseline; some eukaryotic phages |
| 4 | Mold/Mycoplasma/Spiroplasma | Mycoplasma & Spiroplasma phages (TGA → Trp) |
| 11 | Bacterial/Archaeal | **Default** for all bacteriophages |
| 15 | Blepharisma nuclear | Some Crassvirales (TAG → Gln) |
| 25 | SR1/Gracilibacteria | Gracilibacteria phages (TGA → Gly) |

---

## Benchmarks

Performance on common phage genomes (single thread, release build):

| Genome | Size | Time | Speedup vs Python |
|--------|------|------|-------------------|
| phiX174 | 5.4 kb | ~6 ms | ~10× |
| Lambda (NC_001416) | 48.5 kb | ~46 ms | ~12× |
| T4 (NC_000866) | 169 kb | ~100 ms | ~15× |

Benchmarks were run on an AMD Ryzen 9 5900X. Your results may vary.

---

## How it works

phanotate-rs implements the full PHANOTATE algorithm:

1. **ORF enumeration** — Finds all open reading frames in all six frames using the selected stop-codon set
2. **RBS scoring** — Scores Shine-Dalgarno motifs upstream of start codons using a byte-based pattern matcher
3. **GC frame plot** — Computes nucleotide frequencies at codon positions to distinguish coding from non-coding regions
4. **Graph construction** — Builds a weighted directed acyclic graph where nodes are ORF starts/stops and edges represent genes or inter-genic gaps
5. **Shortest path** — Solves for the maximum-weight path through the graph using Bellman-Ford with topological relaxation
6. **Output formatting** — Writes the predicted genes in GenBank, GFF3, or coordinate format

### Automatic table detection (`--detect-table`)

When `--detect-table` is enabled, the tool analyses the first sequence record and recommends the most likely translation table:

1. **Mean ORF length ratio** — The correct table produces longer open regions because it has fewer premature stops.  We normalise by the table-11 baseline so the score is comparable across genomes.

2. **Reassigned-codon signal** — Codons that are stops in table 11 but sense codons in an alternative table (e.g. TGA → Trp in table 4) are the key discriminators.  Their frequency inside candidate-table ORFs tells us whether the alternative code is actually in use:
   - **Tables 4 / 25** (TGA readthrough) — Compares TGA frequency inside long table-4 ORFs against TGG (its sibling Trp/Gly codon) or background.  A ratio near 1.0 means TGA is being read as an amino acid.
   - **Table 15** (TAG/TAA → Gln) — Uses sliding windows (3–9 kb) with per-window background to handle mosaic genomes like crAssphage, where different lineages use different genetic codes.  Strong local TAG enrichment in a subset of windows is the tell-tale signal.

3. **Composite scoring** — `composite = mol_ratio × signal × boost`, where the boost rewards strong signals combined with dramatically longer max ORFs.

4. **Tie-breaking** — Tables 1 and 11 have identical stop sets; table 11 wins if the genome contains table-11-exclusive start codons (ATT, ATC, ATA, GTG).

5. **Confidence level** — Based on the ratio between the top and second-best composite scores: high (>2×), medium (1.2–2×), or low (<1.2×).

The tool prints a ranked report to stderr and prompts for confirmation (or auto-selects with `--yes`).

**Example output:**

```
──────────────────────────────────────────────────────────────────────────────
Codon table detection: seq=NC_003438.1 len=4421 nt (first record)
──────────────────────────────────────────────────────────────────────────────
Rank  Table  Name                                       ORF ratio  Reass. signal  Composite
   1      4  Mold/Protozoan/Coelenterate Mitochon..          1.27          2.375       3.01
              └─ tga: bg=3.3%  orf=1.8%  ratio=0.55
   2     11  Bacterial, Archaeal and Plant Plastid           1.00          1.000       1.00
   3      1  Standard                                        1.00          1.000       1.00
   4     25  Candidate Division SR1 and Graciliba..          1.27          0.384       0.49
              └─ tga: bg=3.3%  orf=1.8%  ratio=0.55
   5     15  Blepharisma Nuclear                             1.08          0.182       0.08
              └─ tag: bg=2.0%  orf=1.4%  ratio=0.68

Recommended table: 4  (Mold/Protozoan/Coelenterate Mitochondrial + Mycoplasma/Spiroplasma)  [confidence: high]
──────────────────────────────────────────────────────────────────────────────
```

### Batch table detection (`--detect-table-batch`)

For multi-FASTA files, `--detect-table-batch` runs detection on **every** record in parallel and prints a TSV summary table to stdout.  This is useful for screening large phage collections without running the full annotation pipeline.

```bash
phanotate-rs --detect-table-batch -i phage_collection.fasta > tables.tsv
```

**Output format:**

```
seq_id	len	recommended	confidence	top_score	runner_up	runner_up_score
NC_003438.1	4421	4	high	3.0061	11	1.0000
NC_001416	48502	11	low	1.0000	1	1.0000
BK025033.1	138252	15	high	1.0945	11	1.0000
```

Columns:
- `seq_id` — sequence identifier from the FASTA header
- `len` — sequence length in nucleotides
- `recommended` — recommended translation table
- `confidence` — `high` (>2× gap), `medium` (1.2–2×), `low` (<1.2×), or `too_short` (<300 nt)
- `top_score` — composite score of the recommended table
- `runner_up` — second-best table
- `runner_up_score` — composite score of the runner-up

---

## Testing

The test suite includes 117 tests across three categories:

- **84 unit tests** — ORF enumeration, weight calculations, table detection algorithms, confidence scoring
- **20 CLI integration tests** — Flag parsing, output format validation, golden-file comparisons against the Python reference
- **10 detect-table integration tests** — Real-genome regression tests (SpV4 → table 4, Lambda → table 11, crAssphage → table 15) plus synthetic sequence tests

```bash
# Run all tests
cargo test

# Run with Clippy (zero warnings policy)
cargo clippy -- -D warnings

# Run benchmarks
cargo bench
```

---

## Citation

If you use **phanotate-rs** or the original **PHANOTATE** in your research, please cite:

> Katelyn McNair, Carol Zhou, Elizabeth A Dinsdale, Brian Souza, Robert A Edwards, **PHANOTATE: a novel approach to gene identification in phage genomes**, *Bioinformatics*, Volume 35, Issue 22, November 2019, Pages 4537–4542, [https://doi.org/10.1093/bioinformatics/btz265](https://doi.org/10.1093/bioinformatics/btz265)

---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This Rust port is based on the original [PHANOTATE](https://github.com/deprekate/PHANOTATE) Python implementation by Robert A. Edwards, Katelyn McNair, and colleagues. The algorithm, training data, and biological insights are entirely theirs.
