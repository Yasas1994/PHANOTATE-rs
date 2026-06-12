#!/usr/bin/env python3
"""Export a scikit-learn tree-based regressor to ONNX for PHANOTATE-rs.

This script trains a RandomForestRegressor on synthetic or real data and
exports it to ONNX format. The resulting model uses the TreeEnsembleRegressor
operator, which is supported by ONNX Runtime (ort) but NOT by tract-onnx.

PHANOTATE-rs now uses the `ort` crate (ONNX Runtime) for inference, which
supports all ONNX operators including tree ensembles.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("ERROR: skl2onnx and onnx required. Install with: pip install skl2onnx onnx")
    sys.exit(1)

OUTPUT_DIR = Path("notebooks/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load or generate training data
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "log_length", "rbs_score_norm", "log_hold", "pstop",
    "weight_rbs_log", "start_codon_ATG", "start_codon_GTG", "start_codon_TTG",
    "gc_content", "frame_fwd", "frame_1", "frame_2", "frame_3",
]
NUM_FEATURES = len(FEATURE_NAMES)

features_path = Path("notebooks/models/training_features.tsv")
if features_path.exists():
    print(f"Loading features from {features_path}")
    import pandas as pd
    df = pd.read_csv(features_path, sep='\t')
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df['is_gene'].values.astype(np.float32)
else:
    print("Generating synthetic training data")
    np.random.seed(42)
    n_samples = 2000
    X = np.random.randn(n_samples, NUM_FEATURES).astype(np.float32)
    # Target: log-ratio in range [ln(0.5), ln(2.0)]
    y = np.random.normal(0.0, 0.3, n_samples).astype(np.float32)
    y = np.clip(y, np.log(0.5), np.log(2.0))

print(f"Training data: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Train RandomForestRegressor
# ---------------------------------------------------------------------------
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf_reg.fit(X_train_s, y_train)

y_pred = rf_reg.predict(X_test_s)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nRandomForestRegressor — MSE: {mse:.4f}, R²: {r2:.4f}")

# ---------------------------------------------------------------------------
# 3. Export to ONNX
# ---------------------------------------------------------------------------
initial_type = [("input", FloatTensorType([None, NUM_FEATURES]))]

onnx_model = convert_sklearn(rf_reg, initial_types=initial_type, target_opset=15)

output_path = OUTPUT_DIR / "model_random_forest_regressor.onnx"
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nExported ONNX model: {output_path}")

# Verify
model = onnx.load(output_path)
onnx.checker.check_model(model)

print("\nModel I/O:")
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value else '?' for d in inp.type.tensor_type.shape.dim]
    print(f"  Input:  {inp.name} -> shape={shape}, type={inp.type.tensor_type.elem_type}")
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value else '?' for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name} -> shape={shape}, type={out.type.tensor_type.elem_type}")

print("\nOperators:")
for node in model.graph.node:
    print(f"  {node.op_type}")

# Save scaler
scaler_path = OUTPUT_DIR / "scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\nSaved scaler to {scaler_path}")

print("\n" + "="*60)
print("USAGE:")
print(f"  cargo run --release --features ml -- \\")
print(f"    -i tests/golden/phiX174.fasta_out \\")
print(f"    --ml-model {output_path}")
print("="*60)
