#!/usr/bin/env python3
"""Benchmark PHANOTATE-rs predictions against GenBank annotations."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd


def _parse_location_segments(loc: str) -> tuple[list[tuple[int, int]], int]:
    """Parse a GenBank location string into linear segments and strand.

    Supports simple ranges (``100..200``), ``join(...)``, and
    ``complement(join(...))``. Segments are returned with ascending
    coordinates regardless of strand.
    """
    loc = loc.strip()
    strand = -1 if loc.startswith("complement(") else 1
    if strand == -1:
        inner = loc[len("complement("):-1]
    else:
        inner = loc

    if inner.startswith("join(") and inner.endswith(")"):
        content = inner[5:-1]
        segment_strs = [s.strip() for s in content.split(",")]
    else:
        segment_strs = [inner]

    segments: list[tuple[int, int]] = []
    for seg in segment_strs:
        if ".." not in seg:
            continue
        a_str, b_str = seg.split("..", 1)
        a = int(a_str.strip().strip("<>"))
        b = int(b_str.strip().strip("<>"))
        segments.append((a, b))
    return segments, strand


def parse_genbank_cds(path: Path) -> set[tuple[int, int]]:
    """Return the set of CDS coordinates from a GenBank file.

    Each coordinate is a ``(start, end)`` tuple with ``start <= end``.
    Only ``CDS`` features are considered; their location strings are parsed
    with the same logic used by the training pipeline.
    """
    features_blocks: list[list[str]] = []
    current_block: list[str] = []
    in_features = False

    with path.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("FEATURES"):
                in_features = True
                continue
            if stripped.startswith("ORIGIN") or stripped.startswith("//"):
                if current_block:
                    features_blocks.append(current_block)
                    current_block = []
                in_features = False
                continue
            if not in_features:
                continue
            # Feature key lines have a non-space character at column 5.
            if len(line) > 5 and line[:5] == "     " and line[5] != " ":
                if current_block:
                    features_blocks.append(current_block)
                    current_block = []
                current_block.append(line)
            elif current_block:
                current_block.append(line)
        if current_block:
            features_blocks.append(current_block)

    coords: set[tuple[int, int]] = set()
    for block in features_blocks:
        key = block[0][5:21].strip()
        if key != "CDS":
            continue
        loc_line = block[0][21:].strip()
        for extra in block[1:]:
            if extra.strip().startswith("/"):
                break
            loc_line += extra.strip()

        segments, _ = _parse_location_segments(loc_line)
        for a, b in segments:
            coords.add((min(a, b), max(a, b)))
    return coords


def run_phanotate(genome: Path, binary: Path, table: int, model: Path | None = None) -> set[tuple[int, int]]:
    """Run phanotate-rs on ``genome`` and return normalized SCO coordinates."""
    cmd = [str(binary), "-i", str(genome), "-g", str(table), "-f", "sco"]
    if model:
        cmd.extend(["--model", str(model)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    preds: set[tuple[int, int]] = set()
    for line in result.stdout.splitlines():
        if line.startswith("#"):
            continue
        cols = line.strip().split("\t")
        if len(cols) >= 2:
            s, e = int(cols[0]), int(cols[1])
            preds.add((min(s, e), max(s, e)))
    return preds


def gene_metrics(pred: set[tuple[int, int]], true: set[tuple[int, int]], tol: int = 3) -> dict:
    """Compute gene-level precision, recall, and F1 with a coordinate tolerance.

    A predicted gene matches a true gene when the intervals are within
    ``tol`` bp of overlapping. Matching is performed greedily by choosing
    the closest predicted interval for each true interval.
    """
    pred_list = sorted(pred)
    true_list = sorted(true)
    matched_pred: set[int] = set()
    matched_true: set[tuple[int, int]] = set()

    for t in true_list:
        best_idx = -1
        best_dist = None
        for i, p in enumerate(pred_list):
            if i in matched_pred:
                continue
            # Intervals overlap or are separated by at most tol bp.
            if max(t[0], p[0]) <= min(t[1], p[1]) + tol:
                dist = abs(t[0] - p[0]) + abs(t[1] - p[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
        if best_idx >= 0:
            matched_true.add(t)
            matched_pred.add(best_idx)

    tp = len(matched_true)
    fp = len(pred) - tp
    fn = len(true) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _average_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(r["precision"] for r in rows) / len(rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
        "f1": sum(r["f1"] for r in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PHANOTATE-rs predictions against GenBank CDS annotations."
    )
    parser.add_argument(
        "--genome-dir",
        type=Path,
        default=Path("tests/golden/annotgenomes_gb"),
        help="Directory containing annotated GenBank files (*.gb)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("./target/release/phanotate-rs"),
        help="Path to the phanotate-rs binary",
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=[],
        help="One or more ONNX model files to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/benchmark_results.json"),
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--max-genomes",
        type=int,
        default=50,
        help="Maximum number of genomes to benchmark",
    )
    parser.add_argument(
        "--table",
        type=int,
        default=11,
        help="NCBI translation table to use for all genomes",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1

    if not args.genome_dir.exists():
        print(f"Error: genome directory not found: {args.genome_dir}", file=sys.stderr)
        return 1

    gb_files = sorted(args.genome_dir.glob("*.gb"))
    if not gb_files:
        print(f"Error: no *.gb files found in {args.genome_dir}", file=sys.stderr)
        return 1

    gb_files = gb_files[: args.max_genomes]

    for model in args.models:
        if not model.exists():
            print(f"Error: model not found: {model}", file=sys.stderr)
            return 1

    per_genome: list[dict] = []
    heuristic_rows: list[dict] = []
    model_rows: dict[str, list[dict]] = {m.name: [] for m in args.models}

    for genome_path in gb_files:
        true_coords = parse_genbank_cds(genome_path)
        if not true_coords:
            warnings.warn(f"Skipping {genome_path.name}: no annotated CDS")
            continue

        record = {
            "genome": genome_path.stem,
            "table": args.table,
            "n_true": len(true_coords),
            "heuristic": None,
            "models": {},
        }

        try:
            pred = run_phanotate(genome_path, args.binary, args.table)
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: heuristic run failed for {genome_path.name}: {e}",
                file=sys.stderr,
            )
            continue
        record["heuristic"] = gene_metrics(pred, true_coords)
        heuristic_rows.append(record["heuristic"])

        for model in args.models:
            try:
                pred = run_phanotate(genome_path, args.binary, args.table, model)
            except subprocess.CalledProcessError as e:
                print(
                    f"Warning: model {model.name} failed for {genome_path.name}: {e}",
                    file=sys.stderr,
                )
                continue
            metrics = gene_metrics(pred, true_coords)
            record["models"][model.name] = metrics
            model_rows[model.name].append(metrics)

        per_genome.append(record)

    if not per_genome:
        print("Error: no genomes produced predictions", file=sys.stderr)
        return 1

    summary = {
        "heuristic": _average_metrics(heuristic_rows),
        "models": {
            name: _average_metrics(rows) for name, rows in model_rows.items()
        },
    }

    results = {
        "config": {
            "binary": str(args.binary),
            "table": args.table,
            "max_genomes": args.max_genomes,
            "models": [str(m) for m in args.models],
        },
        "summary": summary,
        "per_genome": per_genome,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote results to {args.output}")

    # Summary table
    table_rows = []
    table_rows.append({
        "model": "heuristic",
        "precision": summary["heuristic"]["precision"],
        "recall": summary["heuristic"]["recall"],
        "f1": summary["heuristic"]["f1"],
    })
    for name, metrics in summary["models"].items():
        table_rows.append({
            "model": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
    df = pd.DataFrame(table_rows)
    print(f"\nBenchmarked {len(per_genome)} genome(s) (table={args.table}):")
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
