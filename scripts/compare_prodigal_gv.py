#!/usr/bin/env python3
"""Compare Prodigal-gv predictions to reference GenBank CDS annotations.

This script can either run Prodigal-gv for you, or evaluate an existing
Prodigal-gv GFF file.

Usage
-----
    # Run prodigal-gv and compare to GenBank
    python scripts/compare_prodigal_gv.py -i genome.fasta -r ref.gb -g 11

    # Compare an existing prodigal-gv GFF
    python scripts/compare_prodigal_gv.py -p prodigal.gff -r ref.gb

    # Allow 3 bp start/stop tolerance
    python scripts/compare_prodigal_gv.py -i genome.fasta -r ref.gb -t 3
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Set, Tuple


def _parse_genbank_cds(path: str) -> Set[Tuple[int, int]]:
    """Return the CDS coordinate set from a GenBank file."""
    text = Path(path).read_text()

    feat_match = re.search(
        r"FEATURES\s+Location/Qualifiers\s+(.*?)(?=^ORIGIN)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not feat_match:
        return set()

    block = feat_match.group(1)
    cds: Set[Tuple[int, int]] = set()
    current: dict | None = None

    def parse_loc(loc: str) -> Tuple[int, int] | None:
        loc = loc.strip()
        strand = "+"
        if loc.startswith("complement("):
            strand = "-"
            loc = loc[len("complement(") :].rstrip(")")
        if loc.startswith("join("):
            loc = loc[len("join(") :].rstrip(")")
        coords = []
        for part in loc.split(","):
            part = part.strip()
            m = re.match(r"(\d+)\.\.(\d+)", part)
            if m:
                coords.append((int(m.group(1)), int(m.group(2))))
        if not coords:
            return None
        low = min(s for s, _ in coords)
        high = max(e for _, e in coords)
        return (low, high) if strand == "+" else (high, low)

    for raw_line in block.splitlines():
        if not raw_line.strip():
            continue

        qual_match = re.match(r"^\s{21}/([^=]+)(?:=(.*))?$", raw_line)
        if qual_match and current is not None:
            continue

        cont_match = re.match(r"^\s{21}(\S.*)$", raw_line)
        if cont_match and current is not None:
            current["location"] += cont_match.group(1)
            continue

        feat_key_match = re.match(r"^\s{5}(\S+)\s+(\S.*)$", raw_line)
        if feat_key_match:
            if current is not None and current["key"] == "CDS":
                parsed = parse_loc(current["location"])
                if parsed is not None:
                    cds.add(parsed)
            current = {
                "key": feat_key_match.group(1),
                "location": feat_key_match.group(2),
            }

    if current is not None and current["key"] == "CDS":
        parsed = parse_loc(current["location"])
        if parsed is not None:
            cds.add(parsed)

    return cds


def _parse_prodigal_gff(path: str) -> Set[Tuple[int, int]]:
    """Return gene coordinates from a Prodigal/Prodigal-gv GFF file."""
    coords: Set[Tuple[int, int]] = set()
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        if parts[2] != "CDS":
            continue
        try:
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
        except ValueError:
            continue
        if strand == "-":
            coords.add((end, start))
        else:
            coords.add((start, end))
    return coords


def _match(pred: Tuple[int, int], true_set: Set[Tuple[int, int]], tolerance: int) -> bool:
    ps, pe = pred
    for ts, te in true_set:
        if abs(ps - ts) <= tolerance and abs(pe - te) <= tolerance:
            return True
    return False


def compare(pred_set: Set[Tuple[int, int]], true_set: Set[Tuple[int, int]], tolerance: int) -> dict:
    tp = sum(1 for p in pred_set if _match(p, true_set, tolerance))
    fp = len(pred_set) - tp
    fn = sum(1 for t in true_set if not _match(t, pred_set, tolerance))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": len(pred_set),
        "reference": len(true_set),
    }


def _extract_genbank_origin(path: str) -> str:
    """Return the lower-case nucleotide sequence from a GenBank ORIGIN block."""
    text = Path(path).read_text()
    match = re.search(r"ORIGIN\s+(.*?)\n//", text, re.DOTALL)
    if not match:
        raise ValueError(f"No ORIGIN block found in {path}")
    return "".join(re.findall(r"[a-z]+", match.group(1)))


def _run_prodigal_gv(fasta: str, table: int, procedure: str) -> str:
    """Run prodigal-gv and return the path to the generated GFF."""
    binary = shutil.which("prodigal-gv")
    if not binary:
        raise RuntimeError(
            "prodigal-gv not found in PATH. Install it with:\n"
            "  pip install prodigal-gv\n"
            "or pass an existing GFF with -p/--predictions."
        )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".gff", delete=False)
    tmp.close()
    cmd = [
        binary,
        "-i",
        fasta,
        "-o",
        tmp.name,
        "-f",
        "gff",
        "-q",
        "-g",
        str(table),
        "-p",
        procedure,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"prodigal-gv failed: {result.stderr}")
    return tmp.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Prodigal-gv predictions to GenBank CDS annotations."
    )
    parser.add_argument(
        "-i", "--input", help="Input FASTA file for prodigal-gv (required unless -p is given)."
    )
    parser.add_argument(
        "-p", "--predictions", help="Existing prodigal-gv GFF file to evaluate."
    )
    parser.add_argument(
        "-r", "--reference", required=True, help="Reference GenBank file with CDS features."
    )
    parser.add_argument(
        "-g", "--table", type=int, default=11, help="Translation table for prodigal-gv (default: 11)."
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=0,
        help="Allowed start/stop coordinate difference (default: 0).",
    )
    parser.add_argument(
        "--procedure",
        choices=["single", "meta"],
        default="meta",
        help="Prodigal-gv procedure (default: meta, required for short sequences).",
    )
    args = parser.parse_args()

    if not args.predictions and not args.input:
        print("Error: provide either -i/--input (FASTA) or -p/--predictions (GFF).", file=sys.stderr)
        return 1

    gff_path = args.predictions
    tmp_gff = None
    tmp_fasta = None
    try:
        if gff_path is None:
            input_path = Path(args.input)
            if input_path.suffix.lower() in (".gb", ".gbk", ".genbank"):
                seq = _extract_genbank_origin(str(input_path))
                fa_tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".fasta", delete=False
                )
                fa_tmp.write(f">{input_path.stem}\n{seq}\n")
                fa_tmp.close()
                tmp_fasta = fa_tmp.name
                fasta_for_prodigal = tmp_fasta
            else:
                fasta_for_prodigal = str(input_path)

            gff_path = _run_prodigal_gv(fasta_for_prodigal, args.table, args.procedure)
            tmp_gff = gff_path
            print(f"Ran prodigal-gv: {gff_path}")

        pred = _parse_prodigal_gff(gff_path)
        true = _parse_genbank_cds(args.reference)

        if not true:
            print("Error: no CDS features found in reference.", file=sys.stderr)
            return 1

        m = compare(pred, true, args.tolerance)
        print(f"Predicted genes: {m['predicted']}")
        print(f"Reference genes: {m['reference']}")
        print(f"TP: {m['tp']}  FP: {m['fp']}  FN: {m['fn']}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1:        {m['f1']:.4f}")
    finally:
        if tmp_gff:
            Path(tmp_gff).unlink(missing_ok=True)
        if tmp_fasta:
            Path(tmp_fasta).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
