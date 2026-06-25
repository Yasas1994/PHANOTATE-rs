#!/usr/bin/env python3
"""Train an ORF scoring model for PHANOTATE-rs --model.

The model is a JSON logistic regression over the `OrfFeatures` vector used by
`--export-features`. It predicts the probability that an ORF is a real gene;
the Rust runtime converts the probability to a negative log-odds edge weight.

Examples
--------
Train on a single annotated GenBank file:

    python scripts/train_orf_score_model.py \
        -i tests/golden/NC_001365.gb -t 4 \
        -o /tmp/orf_score_model.json

Cross-validate on a directory of GenBank files:

    python scripts/train_orf_score_model.py \
        -i tests/golden/annotgenomes --cv-folds 3 --seed 42 \
        -o /tmp/orf_score_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

NUM_FEATURES = 14
SUPPORTED_TABLES = {1, 4, 6, 11, 15, 25}

FEATURE_NAMES = [
    "log_length",
    "rbs_score_norm",
    "log_hold",
    "pstop",
    "log_sd_rbs_score",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
    "log_non_sd_rbs_score",
]


def _find_phanotate_binary() -> str:
    """Locate the phanotate-rs binary, preferring release builds."""
    candidates = [
        "target/release/phanotate-rs",
        "target/debug/phanotate-rs",
        "phanotate-rs",
    ]
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
    raise RuntimeError(
        "Could not find phanotate-rs binary. Build it with `cargo build --release`."
    )


def _extract_origin(text: str) -> str:
    """Return the lower-case nucleotide sequence from a GenBank ORIGIN block."""
    match = re.search(r"ORIGIN\s+(.*?)\n//", text, re.DOTALL)
    if not match:
        raise ValueError("No ORIGIN block found")
    return "".join(re.findall(r"[a-z]+", match.group(1)))


def _parse_location(location: str) -> Tuple[int, int, str] | None:
    """Parse a GenBank location string into (start, stop, strand)."""
    loc = location.strip()
    strand = "+"
    if loc.startswith("complement("):
        strand = "-"
        loc = loc[len("complement(") :].rstrip(")")
    if loc.startswith("join("):
        loc = loc[len("join(") :].rstrip(")")

    coords: List[Tuple[int, int]] = []
    for part in loc.split(","):
        part = part.strip()
        m = re.match(r"(\d+)\.\.(\d+)", part)
        if m:
            coords.append((int(m.group(1)), int(m.group(2))))

    if not coords:
        return None
    low = min(s for s, _ in coords)
    high = max(e for _, e in coords)
    return (low, high, "+") if strand == "+" else (high, low, "-")


def parse_genbank(path: str) -> Tuple[str, List[Tuple[int, int, str, int]]]:
    """Return (sequence, CDS entries) from a GenBank file."""
    text = Path(path).read_text()
    seq = _extract_origin(text)

    feat_match = re.search(
        r"FEATURES\s+Location/Qualifiers\s+(.*?)(?=^ORIGIN)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not feat_match:
        return seq, []

    block = feat_match.group(1)
    entries: List[Tuple[int, int, str, int]] = []
    current: dict | None = None

    for raw_line in block.splitlines():
        if not raw_line.strip():
            continue

        qual_match = re.match(r"^\s{21}/([^=]+)(?:=(.*))?$", raw_line)
        if qual_match and current is not None:
            key = qual_match.group(1).strip()
            value = qual_match.group(2)
            if value is not None:
                value = value.strip().strip('"').strip("'")
            if key == "transl_table":
                try:
                    current["transl_table"] = int(value)
                except (TypeError, ValueError):
                    pass
            continue

        cont_match = re.match(r"^\s{21}(\S.*)$", raw_line)
        if cont_match and current is not None:
            current["location"] += cont_match.group(1)
            continue

        feat_key_match = re.match(r"^\s{5}(\S+)\s+(\S.*)$", raw_line)
        if feat_key_match:
            if current is not None and current["key"] == "CDS":
                parsed = _parse_location(current["location"])
                if parsed is not None:
                    s, e, strand = parsed
                    entries.append((s, e, strand, current["transl_table"]))
            key = feat_key_match.group(1)
            location = feat_key_match.group(2)
            current = {"key": key, "location": location, "transl_table": 11}

    if current is not None and current["key"] == "CDS":
        parsed = _parse_location(current["location"])
        if parsed is not None:
            s, e, strand = parsed
            entries.append((s, e, strand, current["transl_table"]))

    return seq, entries


def _majority_table(entries: List[Tuple[int, int, str, int]]) -> int | None:
    if not entries:
        return None
    counts: dict[int, int] = {}
    for *_, table in entries:
        counts[table] = counts.get(table, 0) + 1
    return max(counts, key=counts.get)


def _collect_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for arg in inputs:
        p = Path(arg)
        if not p.exists():
            print(f"Warning: path does not exist, skipping: {p}", file=sys.stderr)
            continue
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.gb")))
        else:
            paths.append(p)
    return paths


def _load_genome(path: Path, args: argparse.Namespace, binary: str):
    try:
        seq, entries = parse_genbank(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)
        return None

    if not entries:
        print(f"Warning: no CDS features in {path}, skipping", file=sys.stderr)
        return None

    if args.table is not None:
        genome_table = args.table
    else:
        genome_table = _majority_table(entries)
        if genome_table is None:
            print(
                f"Warning: could not determine table for {path}, skipping",
                file=sys.stderr,
            )
            return None
        if genome_table not in SUPPORTED_TABLES:
            print(
                f"Warning: unsupported table {genome_table} in {path}, skipping",
                file=sys.stderr,
            )
            return None

    ann_set: set = set()
    for s, e, strand, _ in entries:
        if strand == "+":
            if e - s + 1 >= 3:
                ann_set.add((s, e - 2))
        else:
            ann_set.add((s, e))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False
    ) as tmp:
        features_path = tmp.name

    try:
        cmd = [
            binary,
            "-i",
            str(path),
            "-g",
            str(genome_table),
            "--export-features",
            features_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            print(
                f"Warning: feature export failed for {path}: {result.stderr}",
                file=sys.stderr,
            )
            return None

        rows: List[List[float]] = []
        labels: List[int] = []
        with open(features_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                start = int(row["start"])
                stop = int(row["stop"])
                feat = [float(row[name]) for name in FEATURE_NAMES]
                rows.append(feat)
                labels.append(1 if (start, stop) in ann_set else 0)
    finally:
        Path(features_path).unlink(missing_ok=True)

    return {
        "path": path,
        "table": genome_table,
        "X": np.asarray(rows, dtype=float) if rows else np.empty((0, NUM_FEATURES)),
        "y": np.asarray(labels, dtype=int) if labels else np.empty(0, dtype=int),
    }


def _run_cv(genomes: List[dict], n_folds: int, seed: int | None) -> dict:
    n_genomes = len(genomes)
    n_folds = min(n_folds, n_genomes)
    ids = list(range(n_genomes))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(ids)

    fold_size, extra = divmod(n_genomes, n_folds)
    folds: List[List[int]] = []
    pos = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < extra else 1)
        folds.append(ids[pos : pos + size])
        pos += size

    per_fold = []
    for fold_idx, test_ids in enumerate(folds, start=1):
        train_ids = [i for i in range(n_genomes) if i not in test_ids]
        X_train = np.vstack([genomes[i]["X"] for i in train_ids])
        y_train = np.concatenate([genomes[i]["y"] for i in train_ids])
        X_test = np.vstack([genomes[i]["X"] for i in test_ids])
        y_test = np.concatenate([genomes[i]["y"] for i in test_ids])

        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold_idx}: only one class, skipping", file=sys.stderr)
            continue

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0.0] = 1.0
        X_train_s = (X_train - mean) / std
        X_test_s = (X_test - mean) / std

        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs"
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        per_fold.append(
            {
                "fold": fold_idx,
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc) if not np.isnan(auc) else None,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_pos_test": int(np.sum(y_test)),
            }
        )

    return {"folds": per_fold, "n_folds": n_folds}


def _print_cv_summary(cv_result: dict) -> None:
    print("\nCross-validation summary (genome-stratified)")
    print("-" * 70)
    print(
        f"{'Fold':>5} {'N_train':>8} {'N_test':>8} {'Pos':>6} "
        f"{'Prec':>7} {'Rec':>7} {'AUC':>7}"
    )
    print("-" * 70)
    metrics = []
    for row in cv_result["folds"]:
        metrics.append([row[m] for m in ("precision", "recall", "auc")])
        print(
            f"{row['fold']:>5} "
            f"{row['n_train']:>8} {row['n_test']:>8} {row['n_pos_test']:>6} "
            f"{row['precision']:>7.3f} {row['recall']:>7.3f} {_fmt(row['auc']):>7}"
        )
    print("-" * 70)
    if metrics:
        arr = np.array(metrics, dtype=float)
        avgs = np.nanmean(arr, axis=0)
        print(
            f"{'avg':>5} {'':>8} {'':>8} {'':>6} "
            f"{avgs[0]:>7.3f} {avgs[1]:>7.3f} {_fmt(avgs[2]):>7}"
        )
    print()


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "   n/a"
    return f"{value:>7.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an ORF scoring model for PHANOTATE-rs --model."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        help="One or more GenBank files or directories (recursively searched for *.gb).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON path for the trained model.",
    )
    parser.add_argument(
        "-t",
        "--table",
        type=int,
        default=None,
        help="Override the translation table (default: most common /transl_table in each genome).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Optional n-fold genome-stratified cross-validation.",
    )
    parser.add_argument(
        "--min-orf-len",
        type=int,
        default=90,
        help="Minimum ORF length passed to phanotate-rs (default: 90).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for fold shuffling during cross-validation.",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to the phanotate-rs binary (default: target/release/phanotate-rs).",
    )
    args = parser.parse_args()

    binary = args.binary or _find_phanotate_binary()
    if not shutil.which(binary):
        print(f"Error: binary not found: {binary}", file=sys.stderr)
        return 1
    print(f"Using phanotate-rs binary: {binary}")

    paths = _collect_paths(args.input)
    if not paths:
        print("Error: no GenBank files found.", file=sys.stderr)
        return 1

    genomes: List[dict] = []
    for path in paths:
        loaded = _load_genome(path, args, binary)
        if loaded is not None:
            genomes.append(loaded)

    if not genomes:
        print("Error: no usable genomes after parsing.", file=sys.stderr)
        return 1

    print(f"Loaded {len(genomes)} genome(s) from {len(paths)} path(s).")
    total_pos = sum(int(np.sum(g["y"])) for g in genomes)
    total_rows = sum(len(g["y"]) for g in genomes)
    print(f"Total ORFs: {total_rows}; positives: {total_pos}.")

    if args.cv_folds:
        cv_result = _run_cv(genomes, args.cv_folds, args.seed)
        _print_cv_summary(cv_result)

    X = np.vstack([g["X"] for g in genomes])
    y = np.concatenate([g["y"] for g in genomes])
    if len(np.unique(y)) < 2:
        print(
            "Error: need at least one positive and one negative example.",
            file=sys.stderr,
        )
        return 1

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    Xs = (X - mean) / std

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs"
    )
    model.fit(Xs, y)

    out = {
        "version": 1,
        "num_features": NUM_FEATURES,
        "coeffs": model.coef_[0].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(
        f"Wrote {args.output}: {total_pos} positive, {total_rows - total_pos} negative examples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
