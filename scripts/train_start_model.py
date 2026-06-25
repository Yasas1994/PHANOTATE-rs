#!/usr/bin/env python3
"""Train a start-site scoring model from annotated GenBank files.

The model is consumed by the PHANOTATE-rs ``--start-model`` flag.  It is a
simple logistic regression over an 11-feature start-site vector that must stay
in sync with ``StartSiteFeatures::new`` in ``src/start_refiner.rs``.

Examples
--------
Train a model on a single genome with an explicit translation table:

    python scripts/train_start_model.py \\
        -i tests/golden/NC_001365.gb -t 4 \\
        -o /tmp/nc001365_start_model.json

Cross-validate on the first 10 genomes in a directory:

    python scripts/train_start_model.py \\
        -i tests/golden/annotgenomes --cv-folds 3 --seed 42 \\
        -o /tmp/cv_model.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Make the local Python bindings importable when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
import phanotate_rs

NUM_FEATURES = 11
NUM_RBS_BINS = 28
SUPPORTED_TABLES = {1, 4, 6, 11, 15, 25}


def _extract_origin(text: str) -> str:
    """Return the lower-case nucleotide sequence from a GenBank ORIGIN block."""
    match = re.search(r"ORIGIN\s+(.*?)\n//", text, re.DOTALL)
    if not match:
        raise ValueError("No ORIGIN block found")
    return "".join(re.findall(r"[a-z]+", match.group(1)))


def _parse_location(location: str) -> Tuple[int, int, str] | None:
    """Parse a GenBank location string into (start, stop, strand).

    The returned coordinates use PHANOTATE-rs's convention: reverse-strand
    ORFs have ``start > stop``.  ``join()`` locations are flattened to their
    outermost coordinates.
    """
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
        # Ordinary ``a..b`` range.  Ignore single-base or uncertain positions.
        m = re.match(r"(\d+)\.\.(\d+)", part)
        if m:
            coords.append((int(m.group(1)), int(m.group(2))))

    if not coords:
        return None

    low = min(s for s, _ in coords)
    high = max(e for _, e in coords)
    if strand == "+":
        return low, high, "+"
    return high, low, "-"


def parse_genbank(path: str) -> Tuple[str, List[Tuple[int, int, str, int]]]:
    """Return (sequence, CDS entries).

    Each CDS entry is ``(start, stop, strand, transl_table)``.  The genome-wide
    translation table can be derived from the most common ``transl_table`` value.
    """
    text = Path(path).read_text()
    seq = _extract_origin(text)

    # FEATURES ... ORIGIN block.
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

        # Qualifier line: '/key=value' or '/flag' starting at column 21.
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

        # Continuation of a location line (leading spaces, no '/').
        cont_match = re.match(r"^\s{21}(\S.*)$", raw_line)
        if cont_match and current is not None:
            current["location"] += cont_match.group(1)
            continue

        # Feature key line: key in columns 6-20, location from column 21.
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

    # Flush the last feature.
    if current is not None and current["key"] == "CDS":
        parsed = _parse_location(current["location"])
        if parsed is not None:
            s, e, strand = parsed
            entries.append((s, e, strand, current["transl_table"]))

    return seq, entries


def _majority_table(entries: List[Tuple[int, int, str, int]]) -> int | None:
    """Return the most common transl_table, defaulting to 11 for missing values."""
    if not entries:
        return None
    counts: dict[int, int] = {}
    for *_, table in entries:
        counts[table] = counts.get(table, 0) + 1
    return max(counts, key=counts.get)


def build_features(orf) -> np.ndarray:
    """Build the same 11-feature vector as ``StartSiteFeatures::new``."""
    f = np.zeros(NUM_FEATURES, dtype=float)
    codon = orf.start_codon.lower()
    if codon == "atg":
        f[0] = 1.0
    elif codon == "gtg":
        f[1] = 1.0
    elif codon == "ttg":
        f[2] = 1.0
    else:
        f[3] = 1.0
    f[4] = min(orf.rbs_score / NUM_RBS_BINS, 1.0)
    f[5] = min(max(orf.non_sd_motif_score, 0.0), 10.0)
    f[6] = np.log(max(len(orf.sequence), 1))
    f[7] = min(max(1.0 / orf.hold, 0.001), 1000.0)
    frame_abs = abs(orf.frame)
    if frame_abs == 1:
        f[8] = 1.0
    elif frame_abs == 2:
        f[9] = 1.0
    elif frame_abs == 3:
        f[10] = 1.0
    return f


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


def _load_genome(path: Path, args: argparse.Namespace):
    """Load one genome, enumerate ORFs, and return feature rows + metadata."""
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
            print(f"Warning: could not determine table for {path}, skipping", file=sys.stderr)
            return None
        if genome_table not in SUPPORTED_TABLES:
            print(
                f"Warning: unsupported table {genome_table} in {path}, skipping",
                file=sys.stderr,
            )
            return None

    # GenBank CDS coordinates are 1-based inclusive and give the *last* base of
    # the stop codon.  PHANOTATE-rs Orf.stop stores the *first* base of the stop
    # codon, so we map forward annotations from (start, stop) to (start, stop-2).
    # Reverse annotations already use start > stop and match Orf coordinates.
    ann_set: set = set()
    for s, e, strand, _ in entries:
        if strand == "+":
            if e - s + 1 >= 3:
                ann_set.add((s, e - 2))
        else:
            ann_set.add((s, e))
    n_annotations = len(ann_set)

    try:
        orfs = phanotate_rs.find_orfs(
            seq, table=genome_table, min_orf_len=args.min_orf_len
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: find_orfs failed for {path} (table={genome_table}): {exc}",
            file=sys.stderr,
        )
        return None

    rows = []
    labels = []
    coords = []
    for orf in orfs:
        rows.append(build_features(orf))
        labels.append(1 if (orf.start, orf.stop) in ann_set else 0)
        coords.append((orf.start, orf.stop))

    return {
        "path": path,
        "table": genome_table,
        "n_annotations": n_annotations,
        "ann_set": ann_set,
        "X": np.asarray(rows, dtype=float) if rows else np.empty((0, NUM_FEATURES)),
        "y": np.asarray(labels, dtype=int) if labels else np.empty(0, dtype=int),
        "coords": np.asarray(coords, dtype=int) if coords else np.empty((0, 2), dtype=int),
    }


def _run_cv(genomes: List[dict], n_folds: int, seed: int | None) -> dict:
    """Genome-stratified cross-validation.

    Folds are formed from genomes, not individual ORFs, to avoid information
    leakage.  Features are standardized using training-fold statistics.
    """
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
        size = fold_size + (1 if i < extra else 0)
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
            print(
                f"Fold {fold_idx}: training fold has only one class, skipping",
                file=sys.stderr,
            )
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

        # Gene-level recall: fraction of annotated genes with a matching positive ORF.
        offset = 0
        pred_coords: set = set()
        for i in test_ids:
            n = len(genomes[i]["y"])
            local_pred = y_pred[offset : offset + n]
            local_y = genomes[i]["y"]
            coords = genomes[i]["coords"]
            for j in np.where((local_y == 1) & (local_pred == 1))[0]:
                pred_coords.add((int(coords[j, 0]), int(coords[j, 1])))
            offset += n

        test_ann_set: set = set()
        for i in test_ids:
            test_ann_set |= genomes[i]["ann_set"]
        n_genes = len(test_ann_set)
        matched = len(pred_coords & test_ann_set)
        gene_recall = matched / n_genes if n_genes else float("nan")

        per_fold.append(
            {
                "fold": fold_idx,
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc) if not np.isnan(auc) else None,
                "gene_recall": float(gene_recall) if not np.isnan(gene_recall) else None,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_pos_test": int(np.sum(y_test)),
            }
        )

    return {"folds": per_fold, "n_folds": n_folds}


def _print_cv_summary(cv_result: dict) -> None:
    print("\nCross-validation summary (genome-stratified)")
    print("-" * 70)
    header = f"{'Fold':>5} {'N_train':>8} {'N_test':>8} {'Pos':>6} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'GeneRec':>8}"
    print(header)
    print("-" * 70)
    metrics = []
    for row in cv_result["folds"]:
        vals = [row[m] for m in ("precision", "recall", "auc", "gene_recall")]
        metrics.append(vals)
        print(
            f"{row['fold']:>5} "
            f"{row['n_train']:>8} {row['n_test']:>8} {row['n_pos_test']:>6} "
            f"{row['precision']:>7.3f} {row['recall']:>7.3f} "
            f"{_fmt(row['auc']):>7} {_fmt(row['gene_recall']):>8}"
        )
    print("-" * 70)
    if metrics:
        arr = np.array(metrics, dtype=float)
        avgs = np.nanmean(arr, axis=0)
        print(
            f"{'avg':>5} {'':>8} {'':>8} {'':>6} "
            f"{avgs[0]:>7.3f} {avgs[1]:>7.3f} {_fmt(avgs[2]):>7} {_fmt(avgs[3]):>8}"
        )
    print()


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "   n/a"
    return f"{value:>7.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a start-site scoring model from annotated GenBank files."
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
        help="Minimum ORF length passed to phanotate_rs.find_orfs (default: 90).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for fold shuffling during cross-validation.",
    )
    args = parser.parse_args()

    paths = _collect_paths(args.input)
    if not paths:
        print("Error: no GenBank files found.", file=sys.stderr)
        return 1

    genomes: List[dict] = []
    for path in paths:
        loaded = _load_genome(path, args)
        if loaded is not None:
            genomes.append(loaded)

    if not genomes:
        print("Error: no usable genomes after parsing.", file=sys.stderr)
        return 1

    print(f"Loaded {len(genomes)} genome(s) from {len(paths)} path(s).")
    total_pos = sum(int(np.sum(g["y"])) for g in genomes)
    total_rows = sum(len(g["y"]) for g in genomes)
    total_genes = sum(g["n_annotations"] for g in genomes)
    print(
        f"Total ORFs: {total_rows}; positives: {total_pos}; annotated genes: {total_genes}."
    )

    if args.cv_folds:
        cv_result = _run_cv(genomes, args.cv_folds, args.seed)
        _print_cv_summary(cv_result)

    # Final model: fit on all examples.
    X = np.vstack([g["X"] for g in genomes])
    y = np.concatenate([g["y"] for g in genomes])
    if len(np.unique(y)) < 2:
        print("Error: need at least one positive and one negative example.", file=sys.stderr)
        return 1

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    Xs = (X - mean) / std

    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(Xs, y)

    out = {
        "version": 1,
        "num_features": NUM_FEATURES,
        "coeffs": model.coef_[0].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}: {total_pos} positive, {total_rows - total_pos} negative examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
