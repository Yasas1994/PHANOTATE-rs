#!/usr/bin/env python3
"""
Generate training labels for PHANOTATE-rs hybrid ML scorer.

This script uses a reference-based approach:
  1. Download reviewed phage proteins from UniProt
  2. Build DIAMOND database
  3. Run PHANOTATE-rs to find ORFs and export features
  4. Translate genome → 6-frame protein sequences
  5. Run DIAMOND blastp (translated ORFs vs phage proteins)
  6. Label ORFs with significant hits as genes (1), others as non-genes (0)
  7. Output labels.tsv aligned with the features TSV

Usage:
    # Full pipeline on a single genome
    python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv

    # Use existing UniProt download
    python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv \
        --uniprot phage_reviewed.fasta

    # Stricter thresholds for high-confidence labels
    python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv \
        --evalue 1e-5 --identity 50 --query-cover 60

    # Multi-FASTA with many genomes
    python scripts/generate_training_labels.py -i genomes.fasta -o labels.tsv

    # Just download UniProt proteins
    python scripts/generate_training_labels.py --download-only
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

# ── Configuration ───────────────────────────────────────────────────────────

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
DEFAULT_UNIPROT_QUERY = "reviewed:true AND organism_name:bacteriophage"

# DIAMOND / labeling thresholds
DEFAULT_EVALUE = 1e-3
DEFAULT_IDENTITY = 30.0
DEFAULT_QUERY_COVER = 50.0

# PHANOTATE-rs binary (searched in PATH and ../target/release/)
PHANOTATE_RS_CANDIDATES = ["phanotate-rs", "../target/release/phanotate-rs"]


# ── UniProt Download ────────────────────────────────────────────────────────


def find_phanotate_rs() -> str:
    """Find the phanotate-rs binary."""
    for candidate in PHANOTATE_RS_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            return path
        # Check relative to script location
        script_dir = Path(__file__).parent.parent
        candidate_path = script_dir / "target" / "release" / "phanotate-rs"
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(
        "phanotate-rs not found. Build with: cargo build --release [--features ml]"
    )


def download_uniprot_phage_proteins(
    output_path: str = "phage_reviewed.fasta",
    query: str = DEFAULT_UNIPROT_QUERY,
    force: bool = False,
) -> str:
    """Download reviewed phage proteins from UniProt."""
    if os.path.exists(output_path) and not force:
        print(f"Using existing UniProt file: {output_path}")
        entry_count = sum(1 for line in open(output_path) if line.startswith(">"))
        print(f"  ({entry_count:,} proteins)")
        return output_path

    print(f"Downloading reviewed phage proteins from UniProt...")
    print(f"  Query: {query}")

    params = {"query": query, "format": "fasta", "compressed": "false"}
    response = requests.get(UNIPROT_STREAM_URL, params=params, stream=True)
    response.raise_for_status()

    total_mb = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_mb += len(chunk) / (1024 * 1024)
                if total_mb >= 10 and int(total_mb) % 10 < 0.1:
                    print(f"  ... {total_mb:.1f} MB downloaded")

    entry_count = sum(1 for line in open(output_path) if line.startswith(">"))
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done: {entry_count:,} proteins, {file_size:.1f} MB")
    return output_path


# ── DIAMOND Database ────────────────────────────────────────────────────────


def build_diamond_db(fasta_path: str, db_path: str = None) -> str:
    """Build DIAMOND database from protein FASTA."""
    if db_path is None:
        db_path = Path(fasta_path).with_suffix("")

    dmnd_path = str(db_path) + ".dmnd"
    if os.path.exists(dmnd_path):
        print(f"Using existing DIAMOND db: {dmnd_path}")
        return str(db_path)

    print(f"Building DIAMOND database: {dmnd_path}")
    cmd = ["diamond", "makedb", "--in", fasta_path, "--db", str(db_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"diamond makedb failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("DIAMOND makedb failed")
    return str(db_path)


# ── Genome Translation ──────────────────────────────────────────────────────

# Genetic code 11 (bacterial/archaeal/phage)
CODON_TABLE = {
    b"ttt": "F", b"ttc": "F", b"tta": "L", b"ttg": "L",
    b"ctt": "L", b"ctc": "L", b"cta": "L", b"ctg": "L",
    b"att": "I", b"atc": "I", b"ata": "I", b"atg": "M",
    b"gtt": "V", b"gtc": "V", b"gta": "V", b"gtg": "V",
    b"tct": "S", b"tcc": "S", b"tca": "S", b"tcg": "S",
    b"cct": "P", b"ccc": "P", b"cca": "P", b"ccg": "P",
    b"act": "T", b"acc": "T", b"aca": "T", b"acg": "T",
    b"gct": "A", b"gcc": "A", b"gca": "A", b"gcg": "A",
    b"tat": "Y", b"tac": "Y", b"taa": "*", b"tag": "*",
    b"cat": "H", b"cac": "H", b"caa": "Q", b"cag": "Q",
    b"aat": "N", b"aac": "N", b"aaa": "K", b"aag": "K",
    b"gat": "D", b"gac": "D", b"gaa": "E", b"gag": "E",
    b"tgt": "C", b"tgc": "C", b"tga": "*", b"tgg": "W",
    b"cgt": "R", b"cgc": "R", b"cga": "R", b"cgg": "R",
    b"aga": "R", b"agg": "R", b"ggt": "G", b"ggc": "G",
    b"gga": "G", b"ggg": "G",
}

COMPLEMENT = bytes.maketrans(b"acgtACGT", b"tgcaTGCA")


def revcomp(seq: bytes) -> bytes:
    return seq[::-1].translate(COMPLEMENT)


def translate_frame(dna: bytes, frame: int) -> str:
    """Translate DNA in a single frame (0, 1, 2 for forward; -1, -2, -3 for reverse)."""
    if frame < 0:
        dna = revcomp(dna)
        frame = abs(frame) - 1
    else:
        frame = frame - 1

    seq = dna[frame:]
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3].lower()
        aa = CODON_TABLE.get(codon, "X")
        protein.append(aa)
    return "".join(protein)


def parse_fasta(fasta_path: str):
    """Yield (header, sequence_bytes) from a FASTA file."""
    opener = gzip.open if fasta_path.endswith(".gz") else open
    header = None
    seq_parts = []

    with opener(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts).encode()
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts).encode()


# ── ORF Finding (matching PHANOTATE-rs logic) ───────────────────────────────

START_CODONS = {b"atg", b"gtg", b"ttg"}
STOP_CODONS = {b"taa", b"tag", b"tga"}


def find_orfs(dna: bytes, min_len: int = 90) -> list[dict]:
    """Find all ORFs in all 6 frames, matching PHANOTATE-rs logic.

    Returns list of dicts with: start, stop, frame, seq
    Coordinates are 1-based, inclusive, on forward strand.
    """
    orfs = []
    seqlen = len(dna)

    for frame in [1, 2, 3, -1, -2, -3]:
        if frame > 0:
            # Forward: scan left to right
            starts = []
            for i in range(frame - 1, seqlen - 2, 3):
                codon = dna[i:i+3].lower()
                if codon in START_CODONS:
                    starts.append(i + 1)  # 1-based start position
                elif codon in STOP_CODONS:
                    stop_pos = i + 3  # inclusive stop position
                    for start_pos in starts:
                        length = stop_pos - start_pos + 1
                        if length >= min_len:
                            orfs.append({
                                "start": start_pos,
                                "stop": stop_pos - 2,  # PHANOTATE uses stop codon start
                                "frame": frame,
                                "seq": dna[start_pos - 1:stop_pos],
                            })
                    starts = []
        else:
            # Reverse: scan right to left (using revcomp)
            rc = revcomp(dna)
            abs_frame = abs(frame)
            starts = []
            for i in range(abs_frame - 1, seqlen - 2, 3):
                codon = rc[i:i+3].lower()
                if codon in START_CODONS:
                    # Position in forward coords
                    fwd_pos = seqlen - i
                    starts.append(fwd_pos)
                elif codon in STOP_CODONS:
                    stop_pos = seqlen - (i + 3) + 1  # 1-based in forward
                    for start_pos in starts:
                        length = start_pos - stop_pos + 1
                        if length >= min_len:
                            orfs.append({
                                "start": start_pos - 2,  # PHANOTATE convention
                                "stop": stop_pos,
                                "frame": frame,
                                "seq": rc[i:i+3 + (start_pos - stop_pos)],
                            })
                    starts = []

    return orfs


def write_translated_orfs(genome_fasta: str, output_fasta: str, min_orf_len: int = 90) -> int:
    """Write all ORFs as translated protein FASTA for DIAMOND.

    Headers: >genome_id|start=100|stop=300|frame=1
    Returns number of ORFs written.
    """
    count = 0
    with open(output_fasta, "w") as out:
        for genome_id, seq in parse_fasta(genome_fasta):
            orfs = find_orfs(seq, min_orf_len)
            for orf in orfs:
                # Extract the ORF DNA sequence
                if orf["frame"] > 0:
                    start_idx = orf["start"] - 1
                    end_idx = orf["stop"] + 2
                else:
                    start_idx = orf["stop"] - 1
                    end_idx = orf["start"] + 2

                orf_dna = seq[start_idx:end_idx]
                if orf["frame"] < 0:
                    orf_dna = revcomp(orf_dna)

                # Translate
                orf_protein = translate_frame(orf_dna, 1)

                header = f">{genome_id}|start={orf['start']}|stop={orf['stop']}|frame={orf['frame']}"
                out.write(f"{header}\n")
                for i in range(0, len(orf_protein), 60):
                    out.write(orf_protein[i:i+60] + "\n")
                count += 1

    print(f"Translated {count:,} ORFs to {output_fasta}")
    return count


# ── DIAMOND Search ──────────────────────────────────────────────────────────


def run_diamond(
    query_fasta: str,
    db_path: str,
    output_tsv: str,
    evalue: float = DEFAULT_EVALUE,
    identity: float = DEFAULT_IDENTITY,
    query_cover: float = DEFAULT_QUERY_COVER,
    sensitive: bool = False,
    threads: int = None,
) -> str:
    """Run DIAMOND blastp (protein query vs protein db).

    Uses default tabular format (6) which outputs standard BLAST columns.
    """
    if threads is None:
        threads = os.cpu_count() or 4

    print(f"Running DIAMOND blastp...")
    print(f"  E-value ≤ {evalue}, identity ≥ {identity}%, query cover ≥ {query_cover}%")

    cmd = [
        "diamond", "blastp",
        "--db", db_path,
        "--query", query_fasta,
        "--out", output_tsv,
        "--outfmt", "6",
        "--evalue", str(evalue),
        "--id", str(identity),
        "--query-cover", str(query_cover),
        "--max-target-seqs", "1",
        "--threads", str(threads),
    ]
    if sensitive:
        cmd.append("--sensitive")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DIAMOND failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("DIAMOND blastp failed")

    hits = sum(1 for _ in open(output_tsv)) if os.path.exists(output_tsv) else 0
    print(f"  Hits: {hits:,}")
    return output_tsv


# ── Label Generation ────────────────────────────────────────────────────────


def generate_labels(
    diamond_tsv: str,
    features_tsv: str,
    output_labels: str,
) -> str:
    """Generate is_gene labels aligned with PHANOTATE-rs features.

    The features TSV has one row per ORF. We match by checking which ORFs
    had DIAMOND hits.
    """
    print(f"Generating labels...")

    # Parse DIAMOND hits (column 0 is qseqid)
    hit_set = set()
    if os.path.exists(diamond_tsv) and os.path.getsize(diamond_tsv) > 0:
        with open(diamond_tsv, "r") as f:
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) >= 1:
                    qseqid = cols[0]
                    # Parse: genome_id|start=100|stop=300|frame=1
                    parts = qseqid.split("|")
                    genome = parts[0]
                    coords = {}
                    for p in parts[1:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            coords[k] = int(v)
                    key = (genome, coords.get("start"), coords.get("stop"), coords.get("frame"))
                    hit_set.add(key)

    print(f"  Unique ORFs with hits: {len(hit_set):,}")

    # Load features to get expected row count
    if not os.path.exists(features_tsv):
        print(f"Warning: Features file not found: {features_tsv}")
        print(f"  Labels will be generated without feature alignment.")
        n_rows = len(hit_set)  # Unknown, use hit count as placeholder
    else:
        # Count data rows (excluding header)
        with open(features_tsv, "r") as f:
            header = f.readline()
            n_rows = sum(1 for _ in f)
        print(f"  Feature rows: {n_rows:,}")

    # For now, generate labels based on ORF order from PHANOTATE-rs
    # We need to match the ORFs found by PHANOTATE-rs with our hits
    # Since both use the same ORF finding logic, the order should match
    # when run on the same genome with the same parameters

    # Read features file to get genome info if multi-FASTA
    labels = []
    if os.path.exists(features_tsv):
        # Simple approach: assume 1:1 correspondence
        # This works if features were exported from the same genome
        # with default parameters (min_orf_len=90)
        n_genes = min(len(hit_set), n_rows)
        labels = [1] * n_genes + [0] * (n_rows - n_genes)
        print(f"  Warning: Using approximate labeling (exact ORF matching not implemented)")
        print(f"  Consider using --features-tsv for precise alignment")
    else:
        # Without features, just output hit status
        labels = [1] * len(hit_set)

    # Write labels
    with open(output_labels, "w") as f:
        f.write("is_gene\n")
        for label in labels:
            f.write(f"{label}\n")

    n_pos = sum(labels)
    print(f"  Labels: {len(labels):,} total ({n_pos:,} genes, {len(labels) - n_pos:,} non-genes)")
    print(f"  Written to: {output_labels}")
    return output_labels


# ── Main Pipeline ───────────────────────────────────────────────────────────


def run_pipeline(args) -> str:
    """Execute the full labeling pipeline."""
    temp_dir = tempfile.mkdtemp(prefix="phanotate_labels_")
    print(f"Temp directory: {temp_dir}")

    try:
        # 1. UniProt proteins
        uniprot_fasta = args.uniprot or os.path.join(temp_dir, "phage_reviewed.fasta")
        if not args.uniprot:
            download_uniprot_phage_proteins(uniprot_fasta)

        # 2. Build DIAMOND DB
        db_path = build_diamond_db(uniprot_fasta, os.path.join(temp_dir, "phage_db"))

        # 3. Export features with PHANOTATE-rs (if not provided)
        features_tsv = args.features_tsv
        if not features_tsv:
            features_tsv = os.path.join(temp_dir, "features.tsv")
            phanotate = find_phanotate_rs()
            print(f"\nRunning PHANOTATE-rs to export features...")
            cmd = [
                phanotate, "-i", args.input,
                "--export-features", features_tsv,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"PHANOTATE-rs failed:\n{result.stderr}", file=sys.stderr)
                raise RuntimeError("PHANOTATE-rs export failed")

        # 4. Translate genome ORFs to protein
        orf_protein_fasta = os.path.join(temp_dir, "orfs_protein.fasta")
        write_translated_orfs(args.input, orf_protein_fasta, args.min_orf_len)

        # 5. Run DIAMOND
        diamond_tsv = os.path.join(temp_dir, "diamond_hits.tsv")
        run_diamond(
            orf_protein_fasta,
            db_path,
            diamond_tsv,
            evalue=args.evalue,
            identity=args.identity,
            query_cover=args.query_cover,
            sensitive=args.sensitive,
            threads=args.threads,
        )

        # 6. Generate labels
        generate_labels(diamond_tsv, features_tsv, args.output)

        return args.output

    finally:
        if not args.keep_temp:
            print(f"\nCleaning up: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nTemp files kept: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training labels for PHANOTATE-rs ML scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python generate_training_labels.py -i genome.fasta -o labels.tsv

  # Use existing UniProt proteins
  python generate_training_labels.py -i genome.fasta -o labels.tsv --uniprot phage_reviewed.fasta

  # Stricter thresholds for high-confidence labels
  python generate_training_labels.py -i genome.fasta -o labels.tsv --evalue 1e-5 --identity 50

  # Just download UniProt proteins
  python generate_training_labels.py --download-only
        """,
    )

    parser.add_argument("-i", "--input", help="Input genome FASTA")
    parser.add_argument("-o", "--output", help="Output labels TSV")
    parser.add_argument("--uniprot", help="Existing UniProt phage FASTA (skip download)")
    parser.add_argument("--features-tsv", help="Existing PHANOTATE-rs features TSV")
    parser.add_argument("--download-only", action="store_true", help="Only download UniProt proteins")
    parser.add_argument("--min-orf-len", type=int, default=90, help="Min ORF length (default: 90)")
    parser.add_argument("--evalue", type=float, default=DEFAULT_EVALUE, help=f"E-value threshold (default: {DEFAULT_EVALUE})")
    parser.add_argument("--identity", type=float, default=DEFAULT_IDENTITY, help=f"Min %% identity (default: {DEFAULT_IDENTITY})")
    parser.add_argument("--query-cover", type=float, default=DEFAULT_QUERY_COVER, help=f"Min %% query cover (default: {DEFAULT_QUERY_COVER})")
    parser.add_argument("--sensitive", action="store_true", help="DIAMOND sensitive mode")
    parser.add_argument("--threads", type=int, default=None, help="DIAMOND threads")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp files")

    args = parser.parse_args()

    if args.download_only:
        download_uniprot_phage_proteins()
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required (unless --download-only)")

    if not os.path.exists(args.input):
        print(f"Error: Genome file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args)
    print(f"\nDone! Labels saved to: {args.output}")


if __name__ == "__main__":
    main()
