#!/usr/bin/env python3
"""Train and tune an ORF scoring model for PHANOTATE-rs.

Follows the pipeline in notebooks/01_orf_score_model.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# Must match src/ml_features.rs
FEATURE_NAMES = [
    "log_length", "rbs_bin", "log_hold", "pstop", "sd_rbs_score",
    "start_codon_atg", "start_codon_gtg", "start_codon_ttg",
    "gc_content", "frame_fwd", "frame_1", "frame_2", "frame_3",
    "non_sd_rbs_score", "cscore", "cai", "gc1", "gc2", "gc3",
    "overlap_upstream_length", "overlap_upstream_same_strand",
    "overlap_downstream_length", "overlap_downstream_same_strand",
    "stop_sharing_count", "gc_skew", "truncation_penalty",
    "upstream_pwm_score", "rbs_spacer", "heuristic_score",
    "best_alt_pwm_score", "pwm_ratio", "start_rank",
    "num_alt_starts", "start_codon_log_freq",
]
NUM_FEATURES = len(FEATURE_NAMES)

XGB_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def parse_genbank(path: str) -> tuple[str, list[tuple[int, int, int, str]]]:
    """Return (sequence, [(start, end, strand, transl_table), ...])."""
    seq_parts: list[str] = []
    entries: list[tuple[int, int, int, str]] = []
    in_features = False
    in_origin = False
    current_table = "11"

    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("FEATURES"):
                in_features = True
                in_origin = False
            elif stripped.startswith("ORIGIN"):
                in_features = False
                in_origin = True
            elif stripped.startswith("//"):
                in_origin = False
            elif in_origin:
                parts = stripped.split()
                seq_parts.extend(parts[1:])
            elif in_features and (stripped.startswith("CDS") or stripped.startswith("/transl_table")):
                if stripped.startswith("/transl_table"):
                    current_table = stripped.split("=")[-1].strip('"')
                else:
                    loc = stripped[3:].strip()
                    strand = -1 if loc.startswith("complement(") else 1
                    inner = loc.removeprefix("complement(").removesuffix(")")
                    if ".." in inner and "join" not in inner:
                        a, b = inner.split("..")
                        a = int(a.strip("<").strip(">"))
                        b = int(b.strip("<").strip(">"))
                        entries.append((a, b, strand, current_table))
    return "".join(seq_parts).lower(), entries


def _any_segment_matches(start: int, stop: int, ann_set: set[tuple[int, int]], genome_len: int, tol: int = 3) -> bool:
    """Check whether any annotated CDS segment overlaps (start, stop) within tolerance."""
    lo, hi = (start, stop) if start <= stop else (stop, start)
    for a, b in ann_set:
        if max(lo, a) - tol <= min(hi, b) + tol:
            return True
    return False


def _majority_table(entries: list[tuple[int, int, int, str]]) -> str:
    tables = [t for _, _, _, t in entries if t]
    return max(set(tables), key=tables.count) if tables else "11"


def extract_genome_features(
    path: Path,
    binary: Path,
    table_override: str | None = None,
    hard_negatives: bool = False,
) -> dict:
    """Run phanotate-rs --export-features and label ORFs from GenBank CDS."""
    seq, entries = parse_genbank(str(path))
    genome_len = len(seq)
    genome_table = table_override or _majority_table(entries)

    ann_set = {(s, e) for s, e, _, _ in entries}

    pred_fp_set: set[tuple[int, int]] = set()
    if hard_negatives:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sco", delete=False) as sco_tmp:
            subprocess.run(
                [str(binary), "-i", str(path), "-g", genome_table, "-f", "sco"],
                stdout=sco_tmp.file,
                check=True,
            )
            with open(sco_tmp.name) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    cols = line.strip().split("\t")
                    if len(cols) >= 2:
                        s, e = int(cols[0]), int(cols[1])
                        if not _any_segment_matches(s, e, ann_set, genome_len, 3):
                            pred_fp_set.add((s, e))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as feat_tmp:
        subprocess.run(
            [str(binary), "-i", str(path), "-g", genome_table, "--export-features", feat_tmp.name],
            check=True,
        )
        rows, labels, coords, hard_negs = [], [], [], []
        with open(feat_tmp.name) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                start, stop = int(row["start"]), int(row["stop"])
                feat = [float(row[name]) for name in FEATURE_NAMES]
                label = 1 if _any_segment_matches(start, stop, ann_set, genome_len, 3) else 0
                rows.append(feat)
                labels.append(label)
                coords.append((start, stop))
                hard_negs.append(
                    int(
                        label == 0
                        and pred_fp_set
                        and _any_segment_matches(start, stop, pred_fp_set, genome_len, 3)
                    )
                )

    return {
        "path": path,
        "table": genome_table,
        "X": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int32),
        "coords": np.asarray(coords, dtype=np.int32),
        "hard_neg": np.asarray(hard_negs, dtype=np.int32),
        "genome": path.stem,
    }


def genome_stats(path: Path, binary: Path, table_override: str | None = None) -> dict:
    seq, entries = parse_genbank(str(path))
    gc = (seq.count("g") + seq.count("c")) / len(seq) if seq else 0.0
    table = table_override or _majority_table(entries)
    return {"path": path, "length": len(seq), "gc": gc, "table": table}


def sample_genomes(
    genome_dir: Path,
    binary: Path,
    table11_length_bin: int = 15000,
    table11_gc_bin: float = 0.1,
    n_table4: int = 5,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Stratified sample of table-11 genomes + random table-4 genomes."""
    rng = np.random.default_rng(seed)
    stats = [genome_stats(p, binary) for p in genome_dir.glob("*.gb")]
    df = pd.DataFrame(stats)
    df = df[df["table"].isin({"1", "4", "6", "11", "15", "25"})]

    table11 = df[df["table"] == "11"].copy()
    table11["length_bin"] = (table11["length"] // table11_length_bin) * table11_length_bin
    table11["gc_bin"] = (table11["gc"] // table11_gc_bin) * table11_gc_bin
    sampled11 = (
        table11.groupby(["length_bin", "gc_bin"], group_keys=False)
        .apply(lambda g: g.sample(1, random_state=seed))
        .reset_index(drop=True)
    )

    table4 = df[df["table"] == "4"]
    sampled4 = table4.sample(min(n_table4, len(table4)), random_state=seed)

    train_paths = pd.concat([sampled11, sampled4])["path"].tolist()
    test_paths = [p for p in df["path"] if p not in train_paths]
    return train_paths, test_paths


def sample_for_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    hard_neg: np.ndarray | None = None,
    neg_per_pos: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg_target = min(len(neg_idx), n_pos * neg_per_pos)

    if hard_neg is not None and hard_neg.any():
        hard_neg_idx = np.where((y == 0) & hard_neg)[0]
        other_neg_idx = np.where((y == 0) & ~hard_neg)[0]
        n_other = max(0, n_neg_target - len(hard_neg_idx))
        sel_other = rng.choice(
            other_neg_idx,
            size=min(len(other_neg_idx), n_other),
            replace=False,
        )
        sel_neg = np.concatenate([hard_neg_idx, sel_other])
    else:
        sel_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)

    sel = np.concatenate([pos_idx, sel_neg])
    return X[sel], y[sel]


def cv_train(records: list[dict], n_splits: int = 3, seed: int = 42) -> pd.DataFrame:
    df = pd.concat([
        pd.DataFrame({
            "genome": r["genome"],
            "table": r["table"],
            **{name: r["X"][:, i] for i, name in enumerate(FEATURE_NAMES)},
            "is_gene": r["y"],
            "hard_neg": r["hard_neg"],
        })
        for r in records
    ]).reset_index(drop=True)

    genome_ids = df["genome"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(genome_ids)):
        train_genomes = genome_ids[train_idx]
        val_genomes = genome_ids[val_idx]
        train_df = df[df["genome"].isin(train_genomes)]
        val_df = df[df["genome"].isin(val_genomes)]

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["is_gene"].values
        X_val = val_df[FEATURE_NAMES].values
        y_val = val_df["is_gene"].values

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0.0] = 1.0

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        lr.fit((X_train - mean) / std, y_train)
        val_prob_lr = lr.predict_proba((X_val - mean) / std)[:, 1]

        X_train_xgb, y_train_xgb = sample_for_xgboost(
            X_train, y_train, train_df["hard_neg"].values, neg_per_pos=10, seed=seed
        )
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train_xgb) - y_train_xgb.sum()) / y_train_xgb.sum(),
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
        )
        xgb.fit(X_train_xgb, y_train_xgb)
        val_prob_xgb = xgb.predict_proba(X_val)[:, 1]

        results.append({
            "fold": fold,
            "lr_auc": roc_auc_score(y_val, val_prob_lr),
            "xgb_auc": roc_auc_score(y_val, val_prob_xgb),
        })
    return pd.DataFrame(results)


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1_idx = np.argmax(f1s)
    rec95_idx = np.where(rec >= 0.95)[0]
    best_prec_at_rec95_idx = rec95_idx[np.argmax(prec[rec95_idx])] if len(rec95_idx) else best_f1_idx
    return {
        "best_f1_threshold": float(thr[best_f1_idx]) if best_f1_idx < len(thr) else 0.5,
        "best_f1": float(f1s[best_f1_idx]),
        "rec95_threshold": float(thr[best_prec_at_rec95_idx]) if best_prec_at_rec95_idx < len(thr) else 0.5,
        "rec95_precision": float(prec[best_prec_at_rec95_idx]),
    }


def grid_search_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hard_neg: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict,
    seed: int = 42,
) -> tuple[XGBClassifier, dict]:
    best_auc = -1.0
    best_model = None
    best_params = {}
    for max_depth in grid["max_depth"]:
        for lr in grid["learning_rate"]:
            for subsample in grid["subsample"]:
                for colsample in grid["colsample_bytree"]:
                    X_tr, y_tr = sample_for_xgboost(X_train, y_train, hard_neg, neg_per_pos=10, seed=seed)
                    model = XGBClassifier(
                        n_estimators=200,
                        max_depth=max_depth,
                        learning_rate=lr,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
                        n_jobs=-1,
                        random_state=seed,
                        eval_metric="logloss",
                    )
                    model.fit(X_tr, y_tr)
                    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                    if auc > best_auc:
                        best_auc = auc
                        best_model = model
                        best_params = {
                            "max_depth": max_depth,
                            "learning_rate": lr,
                            "subsample": subsample,
                            "colsample_bytree": colsample,
                            "val_auc": auc,
                        }
    return best_model, best_params


def export_linear_to_onnx(
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
    path: Path,
) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
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


def export_xgboost_to_onnx(model, path: Path, n_features: int = NUM_FEATURES) -> None:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, n_features]))]
    )
    path.write_bytes(onnx_model.SerializeToString())


def train_final_and_export(
    records: list[dict], out_dir: Path, tune: bool, seed: int = 42
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([
        pd.DataFrame({
            **{name: r["X"][:, i] for i, name in enumerate(FEATURE_NAMES)},
            "is_gene": r["y"],
            "hard_neg": r["hard_neg"],
        })
        for r in records
    ])
    X_all = df[FEATURE_NAMES].values
    y_all = df["is_gene"].values

    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std[std == 0.0] = 1.0

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    lr.fit((X_all - mean) / std, y_all)
    export_linear_to_onnx(lr, mean, std, out_dir / "model_lr.onnx")

    X_xgb, y_xgb = sample_for_xgboost(X_all, y_all, df["hard_neg"].values, neg_per_pos=10, seed=seed)
    if tune:
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val, h_tr, _ = train_test_split(
            X_all, y_all, df["hard_neg"].values, test_size=0.2, random_state=seed, stratify=y_all
        )
        xgb, params = grid_search_xgb(X_tr, y_tr, h_tr, X_val, y_val, XGB_GRID, seed=seed)
        json.dump(params, (out_dir / "xgb_best_params.json").open("w"), indent=2)
    else:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(X_xgb) - y_xgb.sum()) / y_xgb.sum(),
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
        )
        xgb.fit(X_xgb, y_xgb)
    export_xgboost_to_onnx(xgb, out_dir / "model_xgb.onnx")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and tune an ORF scoring model for PHANOTATE-rs.")
    parser.add_argument("--genome-dir", type=Path, default=Path("tests/golden/annotgenomes_gb"))
    parser.add_argument("--binary", type=Path, default=Path("./target/release/phanotate-rs"))
    parser.add_argument("--cache", type=Path, default=Path("models/feature_cache.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Run XGBoost hyperparameter grid search")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1

    if args.cache.exists() and not args.smoke:
        print(f"Loading cached features from {args.cache}")
        records = joblib.load(args.cache)
    else:
        train_paths, _ = sample_genomes(args.genome_dir, args.binary, seed=args.seed)
        if args.smoke:
            train_paths = train_paths[:3]
        print(f"Extracting features for {len(train_paths)} genome(s)...")
        records = [extract_genome_features(p, args.binary, hard_negatives=True) for p in train_paths]
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(records, args.cache)
        print(f"Cached features to {args.cache}")

    total_orfs = sum(len(r["y"]) for r in records)
    total_pos = sum(int(r["y"].sum()) for r in records)
    print(f"Genomes: {len(records)} | ORFs: {total_orfs} | Positives: {total_pos}")

    if args.smoke:
        for r in records:
            print(f"  {r['genome']}: table={r['table']} X={r['X'].shape} y_pos={int(r['y'].sum())}")
        if not (args.cv or args.tune or args.export):
            return 0

    if args.cv:
        print("\nRunning genome-stratified CV...")
        cv_results = cv_train(records, n_splits=3, seed=args.seed)
        print(cv_results.to_string(index=False))
        print(f"Mean LR AUC: {cv_results['lr_auc'].mean():.4f}")
        print(f"Mean XGB AUC: {cv_results['xgb_auc'].mean():.4f}")

    if args.export or args.tune:
        print("\nTraining final models and exporting to ONNX...")
        train_final_and_export(records, args.output_dir, tune=args.tune, seed=args.seed)
        print(f"Wrote {args.output_dir / 'model_lr.onnx'}")
        print(f"Wrote {args.output_dir / 'model_xgb.onnx'}")
        if args.tune:
            print(f"Wrote {args.output_dir / 'xgb_best_params.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
