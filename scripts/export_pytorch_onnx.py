#!/usr/bin/env python3
"""Create a simple ONNX regression model using PyTorch for tract-onnx compatibility.

tract-onnx supports standard ONNX ops like MatMul, Add, Relu, Sigmoid, etc.
We export with FIXED batch size to avoid symbolic dimension issues.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required. Install with: pip install torch")
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Define a simple MLP in PyTorch
# ---------------------------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        return self.net(x)

model = MLPRegressor(NUM_FEATURES)

# Train
device = torch.device('cpu')
model = model.to(device)
X_train_t = torch.from_numpy(X_train_s)
y_train_t = torch.from_numpy(y_train).view(-1, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(torch.from_numpy(X_test_s)).numpy().flatten()

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMLP Regressor — MSE: {mse:.4f}, R²: {r2:.4f}")

# ---------------------------------------------------------------------------
# 3. Export to ONNX with FIXED batch size (no dynamic axes)
# ---------------------------------------------------------------------------
# tract-onnx has trouble with symbolic batch dimensions.
# We export with batch=1 and will use with_input_fact in Rust to override.
output_path = OUTPUT_DIR / "model_mlp_regressor.onnx"

dummy_input = torch.randn(1, NUM_FEATURES)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    # NO dynamic_axes — fixed batch size
    opset_version=13,
)

print(f"\nExported ONNX model: {output_path}")

# Verify
import onnx
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)

print("\nModel I/O:")
for inp in onnx_model.graph.input:
    shape = [d.dim_value if d.dim_value else '?' for d in inp.type.tensor_type.shape.dim]
    print(f"  Input:  {inp.name} -> shape={shape}, type={inp.type.tensor_type.elem_type}")
for out in onnx_model.graph.output:
    shape = [d.dim_value if d.dim_value else '?' for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name} -> shape={shape}, type={out.type.tensor_type.elem_type}")

print("\nOperators:")
for node in onnx_model.graph.node:
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
