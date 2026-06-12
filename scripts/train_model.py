#!/usr/bin/env python3
"""
Training pipeline for PHANOTATE-rs hybrid ML scorer.

Reads ORF feature TSVs (generated via `phanotate-rs --export-features`),
trains an XGBoost regressor to predict log-scale adjustment factors,
and exports the model to ONNX format for Rust inference.

Usage:
    python scripts/train_model.py features.tsv labels.tsv --output model.onnx

The label file should contain one row per ORF with a 'is_gene' column
(1 = true gene, 0 = false ORF).  Rows must align with the feature TSV.
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: numpy and pandas are required.", file=sys.stderr)
    print("  pip install numpy pandas", file=sys.stderr)
    sys.exit(1)


def load_data(features_path: str, labels_path: str | None):
    """Load feature and label data."""
    features = pd.read_csv(features_path, sep="\t")

    if labels_path is not None:
        labels = pd.read_csv(labels_path, sep="\t")
        if len(features) != len(labels):
            raise ValueError(
                f"Feature/label row count mismatch: {len(features)} vs {len(labels)}"
            )
        # Expect 'is_gene' column in labels
        if "is_gene" not in labels.columns:
            raise ValueError("Label file must contain an 'is_gene' column")
        y = labels["is_gene"].values.astype(np.float32)
    else:
        # Unsupervised mode: train on all data with pseudo-labels
        # derived from heuristic score quartiles
        print("Warning: No labels provided. Using heuristic-based pseudo-labels.",
              file=sys.stderr)
        y = None

    # Drop any non-feature columns if present
    feature_cols = [c for c in features.columns if c not in ("is_gene", "genome_id", "orf_id")]
    X = features[feature_cols].values.astype(np.float32)

    return X, y, feature_cols


def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, max_depth: int = 4):
    """Train an XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("Error: xgboost is required.", file=sys.stderr)
        print("  pip install xgboost", file=sys.stderr)
        sys.exit(1)

    # Target: log-adjustment that would make heuristic score optimal
    # For genes (y=1): we want adjustment > 1 (boost score)
    # For non-genes (y=0): we want adjustment < 1 (reduce score)
    # We model this as a log-odds-like target:
    #   target = log( (y + epsilon) / (1 - y + epsilon) ) * scale
    epsilon = 0.01
    scale = 0.5  # Keep adjustments moderate
    target = np.log((y + epsilon) / (1.0 - y + epsilon)) * scale

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, target)

    # Report training metrics
    preds = model.predict(X)
    mse = np.mean((preds - target) ** 2)
    print(f"Training MSE: {mse:.4f}")

    # Show feature importances
    importances = model.feature_importances_
    print("\nFeature importances:")
    for name, imp in zip(FEATURE_NAMES, importances):
        print(f"  {name:20s}: {imp:.4f}")

    return model


def export_onnx(model, feature_names: list[str], output_path: str):
    """Export the trained model to ONNX format."""
    try:
        from skl2onnx import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("Error: skl2onnx is required for ONNX export.", file=sys.stderr)
        print("  pip install skl2onnx", file=sys.stderr)
        sys.exit(1)

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 2},
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\nModel exported to: {output_path}")
    print(f"  Input shape: [batch, {len(feature_names)}]")
    print(f"  Output shape: [batch, 1]")


def create_dummy_model(output_path: str):
    """Create a dummy ONNX model that always outputs 0 (neutral adjustment)."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("Error: onnx is required.", file=sys.stderr)
        print("  pip install onnx", file=sys.stderr)
        sys.exit(1)

    # Create a model that outputs 0 regardless of input
    # Use fixed batch dimension for tract compatibility
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, NUM_FEATURES]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]
    )

    # MatMul with zero weights + zero bias = always zero output
    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [NUM_FEATURES, 1],
        [0.0] * NUM_FEATURES
    )
    bias = helper.make_tensor(
        "B", TensorProto.FLOAT, [1], [0.0]
    )

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "B"], ["output"])

    graph = helper.make_graph(
        [matmul_node, add_node],
        "dummy_model",
        [input_tensor],
        [output_tensor],
        [weight, bias],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Dummy neutral model saved to: {output_path}")


# Feature names must match src/ml_features.rs FEATURE_NAMES
NUM_FEATURES = 13
FEATURE_NAMES = [
    "log_length",
    "rbs_score_norm",
    "log_hold",
    "pstop",
    "weight_rbs_log",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
]


def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid ML scorer for PHANOTATE-rs"
    )
    parser.add_argument(
        "features",
        nargs="?",
        help="Path to feature TSV (from --export-features)",
    )
    parser.add_argument(
        "--labels",
        help="Path to label TSV with 'is_gene' column",
        default=None,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output ONNX model path",
        default="model.onnx",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Create a dummy neutral model instead of training",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of XGBoost trees (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Max tree depth (default: 4)",
    )
    args = parser.parse_args()

    if args.dummy:
        create_dummy_model(args.output)
        return

    if args.features is None:
        parser.error("features TSV is required unless --dummy is used")

    print(f"Loading features from: {args.features}")
    X, y, feature_cols = load_data(args.features, args.labels)
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")

    if y is not None:
        n_genes = int(y.sum())
        print(f"  Genes: {n_genes}, Non-genes: {len(y) - n_genes}")

    print("\nTraining XGBoost regressor...")
    model = train_model(X, y, args.n_estimators, args.max_depth)

    print("\nExporting to ONNX...")
    export_onnx(model, feature_cols, args.output)

    print("\nDone! To use the model:")
    print(f"  cargo run --features ml -- -i genome.fasta --ml-model {args.output}")


if __name__ == "__main__":
    main()
