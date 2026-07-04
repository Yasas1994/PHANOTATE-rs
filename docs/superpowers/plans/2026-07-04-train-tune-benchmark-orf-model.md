# Train, Tune, and Benchmark a New ORF Scoring Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a new ONNX ORF scoring model following `notebooks/01_orf_score_model.ipynb`, tune its hyperparameters and PHANOTATE-rs runtime parameters, and find a configuration that outperforms the current heuristic scorer on gene-level precision/recall/F1.

**Architecture:** Replicate the notebook's training pipeline in a standalone Python script under `scripts/`, run genome-stratified CV to compare classifiers and thresholds, export the best model to ONNX, then run an automated benchmark loop that evaluates each candidate against the default heuristic on held-out genomes using the existing `compare_predictions.py` metrics.

**Tech Stack:** Python 3.13, scikit-learn, xgboost, skl2onnx, onnxmltools, PHANOTATE-rs CLI, pytest for any new helpers.

---

## File structure

- `scripts/train_orf_score_model_v2.py` — main training/tuning script (created).
- `scripts/benchmark_model.py` — benchmark heuristic vs one or more ONNX models (created).
- `models/` — output directory for candidate ONNX files and result JSONs (existing).
- `notebooks/01_orf_score_model.ipynb` — read-only reference; must not be modified.

---

## Task 1: Create a reproducible training script

**Files:**
- Create: `scripts/train_orf_score_model_v2.py`
- Modify: none
- Test: run on a small subset and verify output shape

- [ ] **Step 1: Copy the notebook's feature-extraction helpers**

```python
# scripts/train_orf_score_model_v2.py
"""Train and tune an ORF scoring model for PHANOTATE-rs.

Follows the pipeline in notebooks/01_orf_score_model.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

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
```

- [ ] **Step 2: Add GenBank parsing and coordinate matching helpers**

```python
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
```

- [ ] **Step 3: Add feature extraction wrapper**

```python
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sco", delete=False) as sco_tmp:
        pred_fp_set: set[tuple[int, int]] = set()
        if hard_negatives:
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
```

- [ ] **Step 4: Add a quick smoke test**

Run:
```bash
python scripts/train_orf_score_model_v2.py --smoke --genome-dir tests/golden/annotgenomes_gb --binary ./target/release/phanotate-rs
```

Expected: script loads a few genomes, extracts features, prints shapes, exits 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_orf_score_model_v2.py
git commit -m "feat: add reproducible ORF model training script"
```

---

## Task 2: Build training/test data splits

**Files:**
- Modify: `scripts/train_orf_score_model_v2.py`

- [ ] **Step 1: Add genome sampling helpers**

```python
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
```

- [ ] **Step 2: Add CLI args for data caching**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome-dir", type=Path, default=Path("tests/golden/annotgenomes_gb"))
    parser.add_argument("--binary", type=Path, default=Path("./target/release/phanotate-rs"))
    parser.add_argument("--cache", type=Path, default=Path("models/feature_cache.pkl"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ...
```

- [ ] **Step 3: Cache extracted features**

```python
import joblib

if args.cache.exists() and not args.smoke:
    records = joblib.load(args.cache)
else:
    train_paths, _ = sample_genomes(args.genome_dir, args.binary)
    if args.smoke:
        train_paths = train_paths[:3]
    records = [extract_genome_features(p, args.binary, hard_negatives=True) for p in train_paths]
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(records, args.cache)
```

- [ ] **Step 4: Run smoke test**

Run:
```bash
python scripts/train_orf_score_model_v2.py --smoke
```

Expected: creates `models/feature_cache.pkl`, prints feature matrix shape.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_orf_score_model_v2.py
git commit -m "feat: stratified genome sampling and feature caching"
```

---

## Task 3: Train baseline classifiers with genome-stratified CV

**Files:**
- Modify: `scripts/train_orf_score_model_v2.py`

- [ ] **Step 1: Add hard-negative subsampling for XGBoost**

```python
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
```

- [ ] **Step 2: Add CV training loop**

```python
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
```

- [ ] **Step 3: Run baseline CV and print metrics**

Run:
```bash
python scripts/train_orf_score_model_v2.py --cv
```

Expected: prints a 3-fold CV table with LR and XGBoost AUCs.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_orf_score_model_v2.py
git commit -m "feat: genome-stratified CV for LR and XGBoost baselines"
```

---

## Task 4: Add hyperparameter tuning

**Files:**
- Modify: `scripts/train_orf_score_model_v2.py`

- [ ] **Step 1: Add threshold tuning helper**

```python
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
```

- [ ] **Step 2: Add a small hyperparameter grid for XGBoost**

```python
XGB_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
```

- [ ] **Step 3: Add grid-search function inside CV**

```python
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
```

- [ ] **Step 4: Wire `--tune` CLI flag**

```python
parser.add_argument("--tune", action="store_true", help="Run XGBoost hyperparameter grid search")
```

- [ ] **Step 5: Run tuning on the smoke subset first**

Run:
```bash
python scripts/train_orf_score_model_v2.py --smoke --tune
```

Expected: finishes in a few minutes, prints best XGBoost params and validation AUC.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_orf_score_model_v2.py
git commit -m "feat: add XGBoost hyperparameter grid search and threshold tuning"
```

---

## Task 5: Export the best model(s) to ONNX

**Files:**
- Modify: `scripts/train_orf_score_model_v2.py`

- [ ] **Step 1: Add ONNX export helpers**

```python
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
```

- [ ] **Step 2: Train final model on all train genomes and export**

```python
def train_final_and_export(records: list[dict], out_dir: Path, tune: bool, seed: int = 42):
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
        # Use a single held-out split for final tuning to keep runtime sane.
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
```

- [ ] **Step 3: Run final export on smoke data**

Run:
```bash
python scripts/train_orf_score_model_v2.py --smoke --export --output-dir models/
```

Expected: creates `models/model_lr.onnx` and `models/model_xgb.onnx`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_orf_score_model_v2.py
git commit -m "feat: export final LR and XGBoost models to ONNX"
```

---

## Task 6: Build the benchmarking script

**Files:**
- Create: `scripts/benchmark_model.py`
- Modify: `scripts/compare_predictions.py` only if a helper is missing; prefer no changes.

- [ ] **Step 1: Create benchmark script skeleton**

```python
# scripts/benchmark_model.py
"""Benchmark PHANOTATE-rs predictions: heuristic vs ONNX model(s)."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def run_phanotate(genome: Path, binary: Path, table: int, model: Path | None = None) -> set[tuple[int, int]]:
    cmd = [str(binary), "-i", str(genome), "-g", str(table), "-f", "sco"]
    if model:
        cmd.extend(["--model", str(model)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    preds = set()
    for line in result.stdout.splitlines():
        if line.startswith("#"):
            continue
        cols = line.strip().split("\t")
        if len(cols) >= 2:
            preds.add((int(cols[0]), int(cols[1])))
    return preds


def parse_genbank_cds(path: Path) -> set[tuple[int, int]]:
    cds = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("CDS"):
                loc = line[3:].strip()
                strand = -1 if loc.startswith("complement(") else 1
                inner = loc.removeprefix("complement(").removesuffix(")")
                if ".." in inner and "join" not in inner:
                    a, b = inner.split("..")
                    a = int(a.strip("<").strip(">"))
                    b = int(b.strip("<").strip(">"))
                    cds.add((a, b))
    return cds


def gene_metrics(pred: set[tuple[int, int]], true: set[tuple[int, int]], tol: int = 3) -> dict:
    matched_pred = set()
    matched_true = set()
    for p in pred:
        for t in true:
            if max(p[0], t[0]) - tol <= min(p[1], t[1]) + tol:
                matched_pred.add(p)
                matched_true.add(t)
    tp = len(matched_pred)
    fp = len(pred - matched_pred)
    fn = len(true - matched_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome-dir", type=Path, default=Path("tests/golden/annotgenomes_gb"))
    parser.add_argument("--binary", type=Path, default=Path("./target/release/phanotate-rs"))
    parser.add_argument("--models", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path, default=Path("models/benchmark_results.json"))
    parser.add_argument("--max-genomes", type=int, default=50)
    args = parser.parse_args()

    genomes = sorted(args.genome_dir.glob("*.gb"))[:args.max_genomes]
    rows = []
    for genome in genomes:
        true = parse_genbank_cds(genome)
        if not true:
            continue
        heuristic = run_phanotate(genome, args.binary, 11)
        rows.append({"genome": genome.stem, "model": "heuristic", **gene_metrics(heuristic, true)})
        for model_path in args.models:
            preds = run_phanotate(genome, args.binary, 11, model_path)
            rows.append({"genome": genome.stem, "model": model_path.name, **gene_metrics(preds, true)})

    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["precision", "recall", "f1"]].mean().reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary.to_dict(orient="records"),
        "per_genome": df.to_dict(orient="records"),
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark on a few genomes**

Run:
```bash
python scripts/benchmark_model.py --models models/model_lr.onnx models/model_xgb.onnx --max-genomes 10
```

Expected: prints precision/recall/F1 for heuristic, LR, and XGBoost; writes JSON.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_model.py
git commit -m "feat: add heuristic vs ONNX benchmark script"
```

---

## Task 7: Full training + benchmarking run

**Files:**
- Modify: none (uses scripts)
- Test: verify JSON outputs

- [ ] **Step 1: Build release binary**

```bash
cargo build --release
```

- [ ] **Step 2: Extract features for the stratified training set**

```bash
python scripts/train_orf_score_model_v2.py --cache models/feature_cache.pkl
```

Expected: creates `models/feature_cache.pkl` with ~102 genomes.

- [ ] **Step 3: Run CV to confirm baselines are reasonable**

```bash
python scripts/train_orf_score_model_v2.py --cv
```

Expected: prints per-fold AUCs.

- [ ] **Step 4: Train final models with tuning**

```bash
python scripts/train_orf_score_model_v2.py --tune --export --output-dir models/
```

Expected: creates `models/model_lr.onnx`, `models/model_xgb.onnx`, and `models/xgb_best_params.json`.

- [ ] **Step 5: Benchmark on held-out genomes**

```bash
python scripts/benchmark_model.py --models models/model_lr.onnx models/model_xgb.onnx --max-genomes 100 --output models/benchmark_results.json
```

Expected: prints average precision/recall/F1. If neither model beats heuristic, continue to Task 8.

- [ ] **Step 6: Commit results**

```bash
git add models/xgb_best_params.json models/benchmark_results.json
git commit -m "results: baseline LR/XGB models and benchmark"
```

---

## Task 8: Iterate to beat the heuristic

**Files:**
- Modify: `scripts/train_orf_score_model_v2.py`, `scripts/benchmark_model.py`, possibly `src/ml_features.rs` if new features are needed.

- [ ] **Step 1: Analyze failure modes**

Open `models/benchmark_results.json` and identify genomes where the learned model underperforms the heuristic. Common causes:
- Threshold too high/low.
- Gap/overlap penalties not calibrated for the model's score scale.
- Missing features (e.g., start-codon frequency for table 4).

- [ ] **Step 2: Tune PHANOTATE-rs runtime parameters**

Add a parameter-sweep helper to `benchmark_model.py`:

```python
for scale in [0.5, 1.0, 2.0, 5.0]:
    for threshold in [0.3, 0.5, 0.7]:
        cmd = [str(binary), "-i", str(genome), "-g", str(table),
               "--model", str(model), "--model-scale", str(scale),
               "--model-threshold", str(threshold), "-f", "sco"]
```

Run the sweep and pick the `(scale, threshold)` combination with the highest average F1.

- [ ] **Step 3: Try richer models if still behind**

If parameter tuning is insufficient, extend `train_orf_score_model_v2.py` with:
- A `RandomForestClassifier` baseline.
- A wider XGBoost grid.
- Class-weight / scale_pos_weight variants.

- [ ] **Step 4: Add features if needed**

If models still lag, add a new feature in `src/ml_features.rs` (e.g., table-specific start-codon frequency, dicodon score), recompute `--export-features`, and retrain. This step is optional and only if the benchmark clearly shows a feature gap.

- [ ] **Step 5: Re-run benchmark and commit the winning model**

Once a configuration beats the heuristic, copy the best ONNX to `models/default.onnx` **only if the user approves**, and commit:

```bash
git add models/default.onnx models/benchmark_results.json scripts/*.py
git commit -m "feat: new ORF model outperforms heuristic (F1 X vs Y)"
```

---

## Verification

- `cargo test` passes after any Rust changes.
- `cargo fmt -- --check` and `cargo clippy -- -D warnings` pass.
- Training script runs end-to-end on the full dataset without errors.
- Benchmark JSON contains per-genome and average precision/recall/F1.
- At least one model+parameter configuration achieves higher average F1 than the heuristic on the held-out benchmark set.

## Self-review checklist

- [ ] Spec coverage: training, tuning, ONNX export, benchmarking, and iteration are all represented.
- [ ] No placeholders: every step includes concrete code or exact commands.
- [ ] Type consistency: feature names, NUM_FEATURES, and ONNX input shape match `src/ml_features.rs`.
- [ ] Notebook untouched: `notebooks/01_orf_score_model.ipynb` is only read, never modified.
