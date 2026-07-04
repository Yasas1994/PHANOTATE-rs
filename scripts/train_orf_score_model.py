#!/usr/bin/env python3
"""Train an ORF scoring model for PHANOTATE-rs --model.

The model is an ONNX classifier that consumes the `OrfFeatures` vector used by
`--export-features`. It predicts the probability that an ORF is a real gene;
the Rust runtime converts the probability to a negative log-odds edge weight.

Examples
--------
Train a logistic-regression model on a single annotated GenBank file:

    python scripts/train_orf_score_model.py \
        -i tests/golden/NC_001365.gb -t 4 \
        -o /tmp/orf_score_model.onnx

Train an XGBoost model:

    python scripts/train_orf_score_model.py \
        -i tests/golden/annotgenomes --model-type xgboost \
        -o /tmp/orf_score_model.onnx

Cross-validate on a directory of GenBank files:

    python scripts/train_orf_score_model.py \
        -i tests/golden/annotgenomes --cv-folds 3 --seed 42 \
        -o /tmp/orf_score_model.onnx
"""

from __future__ import annotations

import argparse
import csv
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

NUM_FEATURES = 34
SUPPORTED_TABLES = {1, 4, 6, 11, 15, 25}

FEATURE_NAMES = [
    "log_length",
    "rbs_bin",
    "log_hold",
    "pstop",
    "sd_rbs_score",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
    "non_sd_rbs_score",
    "cscore",
    "cai",
    "gc1",
    "gc2",
    "gc3",
    "overlap_upstream_length",
    "overlap_upstream_same_strand",
    "overlap_downstream_length",
    "overlap_downstream_same_strand",
    "stop_sharing_count",
    "gc_skew",
    "truncation_penalty",
    "upstream_pwm_score",
    "rbs_spacer",
    "heuristic_score",
    "best_alt_pwm_score",
    "pwm_ratio",
    "start_rank",
    "num_alt_starts",
    "start_codon_log_freq",
]


def export_linear_to_onnx(
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
    path: Path,
) -> None:
    """Export a standardised logistic-regression model to ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("skl2onnx is required for ONNX export") from exc

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])
    pipe.named_steps["scaler"].mean_ = mean
    pipe.named_steps["scaler"].scale_ = std
    pipe.named_steps["scaler"].var_ = std**2
    pipe.named_steps["scaler"].n_features_in_ = mean.shape[0]
    pipe.named_steps["clf"].coef_ = model.coef_
    pipe.named_steps["clf"].intercept_ = model.intercept_
    pipe.named_steps["clf"].classes_ = model.classes_

    initial_type = [("input", FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset=15,
        options={LogisticRegression: {"zipmap": False}},
    )
    path.write_bytes(onnx_model.SerializeToString())


def export_xgboost_to_onnx(model, path: Path) -> None:
    """Export an XGBoost classifier to ONNX."""
    try:
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnxmltools
    except ImportError as exc:
        raise RuntimeError("onnxmltools is required for XGBoost ONNX export") from exc
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, NUM_FEATURES]))]
    )
    path.write_bytes(onnx_model.SerializeToString())


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
        if abs(e - s) + 1 < 3:
            continue
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
        hard_negs: List[int] = []
        pred_fp_set: set[tuple[int, int]] = set()

        if args.hard_negatives:
            pred_cmd = [
                binary,
                "-i",
                str(path),
                "-g",
                str(genome_table),
                "-f",
                "sco",
            ]
            pred_result = subprocess.run(
                pred_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if pred_result.returncode == 0:
                pred_set = _parse_sco_text(pred_result.stdout)
                pred_fp_set = {
                    c for c in pred_set if not _match_coord(c, ann_set, 3)
                }
            else:
                print(
                    f"Warning: SCO prediction failed for {path}; "
                    "hard-negative mining skipped.",
                    file=sys.stderr,
                )

        with open(features_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                start = int(row["start"])
                stop = int(row["stop"])
                feat = [float(row[name]) for name in FEATURE_NAMES]
                label = 1 if (start, stop) in ann_set else 0
                rows.append(feat)
                labels.append(label)
                hard_negs.append(
                    1
                    if label == 0
                    and pred_fp_set
                    and _match_coord((start, stop), pred_fp_set, 3)
                    else 0
                )
    finally:
        Path(features_path).unlink(missing_ok=True)

    return {
        "path": path,
        "table": genome_table,
        "X": np.asarray(rows, dtype=float) if rows else np.empty((0, NUM_FEATURES)),
        "y": np.asarray(labels, dtype=int) if labels else np.empty(0, dtype=int),
        "hard_neg": np.asarray(hard_negs, dtype=bool)
        if hard_negs
        else np.empty(0, dtype=bool),
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


def _parse_sco_text(text: str) -> set[tuple[int, int]]:
    """Parse a PHANOTATE SCO string into a set of (start, stop) coordinates."""
    coords: set[tuple[int, int]] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            coords.add((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return coords


def _match_coord(coord: tuple[int, int], coord_set: set[tuple[int, int]], tolerance: int = 3) -> bool:
    """Return True if `coord` matches any coordinate in `coord_set` within tolerance."""
    s, e = coord
    for ts, te in coord_set:
        if abs(s - ts) <= tolerance and abs(e - te) <= tolerance:
            return True
    return False


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
        help="Output ONNX path for the trained model.",
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
    parser.add_argument(
        "--model-type",
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Model type to train (default: logistic).",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help=(
            "Run the default heuristic on each genome and keep its false-positive "
            "predictions in the down-sampled negative set."
        ),
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

    if args.model_type == "logistic":
        Xs = (X - mean) / std
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs"
        )
        model.fit(Xs, y)
    elif args.model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            print(
                "Error: xgboost is required for --model-type xgboost",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        n_neg_target = min(len(neg_idx), n_pos * 10)
        rng = np.random.default_rng(args.seed)

        hard_neg_mask = np.concatenate([g["hard_neg"] for g in genomes])
        if hard_neg_mask.any():
            hard_neg_idx = np.where((y == 0) & hard_neg_mask)[0]
            other_neg_idx = np.where((y == 0) & ~hard_neg_mask)[0]
            n_other = max(0, n_neg_target - len(hard_neg_idx))
            sel_other = rng.choice(
                other_neg_idx, size=min(len(other_neg_idx), n_other), replace=False
            )
            sel_neg = np.concatenate([hard_neg_idx, sel_other])
        else:
            sel_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
        sel = np.concatenate([pos_idx, sel_neg])
        X_sel = X[sel]
        y_sel = y[sel]

        scale_pos_weight = (len(y_sel) - n_pos) / max(n_pos, 1)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        model.fit(X_sel, y_sel)
    else:
        print(f"Error: unsupported model type: {args.model_type}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    if out_path.suffix != ".onnx":
        out_path = out_path.with_suffix(".onnx")

    if args.model_type == "logistic":
        export_linear_to_onnx(model, mean, std, out_path)
    else:
        export_xgboost_to_onnx(model, out_path)

    print(
        f"Wrote {out_path}: {total_pos} positive, {total_rows - total_pos} negative examples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
