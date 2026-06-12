# ONNX Model Export for PHANOTATE-rs ML Scorer

## Overview

PHANOTATE-rs supports ONNX-based ML models for hybrid ORF scoring. We use the
**`ort` crate** (ONNX Runtime) for inference, which supports all ONNX operators
including tree-based models (Random Forest, XGBoost) and neural networks.

## Supported Model Types

| Model Type | ONNX Operator | Supported |
|-----------|--------------|-----------|
| Random Forest Regressor | `TreeEnsembleRegressor` | ✅ |
| XGBoost Regressor | `TreeEnsembleRegressor` | ✅ |
| Neural Network (MLP) | `Gemm`, `Relu`, etc. | ✅ |
| LightGBM | `TreeEnsembleRegressor` | ✅ |
| Any ONNX model | All ops | ✅ |

## Quick Start

### 1. Export a Tree-based Model (Random Forest)

```bash
python3 scripts/export_tree_onnx.py
```

This creates:
- `notebooks/models/model_random_forest_regressor.onnx` — ONNX model with `TreeEnsembleRegressor`
- `notebooks/models/scaler.pkl` — StandardScaler for feature normalization

### 2. Export a Neural Network (PyTorch MLP)

```bash
python3 scripts/export_pytorch_onnx.py
```

This creates:
- `notebooks/models/model_mlp_regressor.onnx` — ONNX model with `Gemm`/`Relu`
- `notebooks/models/scaler.pkl` — StandardScaler

### 3. Run PHANOTATE-rs with ML

```bash
cargo run --release --features ml -- \
  -i tests/golden/phiX174.fasta_out \
  --ml-model notebooks/models/model_random_forest_regressor.onnx
```

## Model Requirements

### Input
- **Shape**: `[batch_size, 13]` where `batch_size` can be any positive integer
- **Type**: `float32`
- **Features**: See `src/ml_features.rs` for feature definitions

### Output
- **Shape**: `[batch_size, 1]`
- **Type**: `float32`
- **Interpretation**: Log-scale adjustment value. The Rust code clamps to `[ln(0.5), ln(2.0)]` and exponentiates to get a multiplicative factor in `[0.5, 2.0]`.

## Feature List (13 features)

| # | Feature | Description |
|---|---------|-------------|
| 0 | `log_length` | ln(ORF length in bp) |
| 1 | `rbs_score_norm` | RBS score / 27.0 |
| 2 | `log_hold` | ln(hold value) |
| 3 | `pstop` | Stop codon probability |
| 4 | `weight_rbs_log` | ln(weight_rbs) |
| 5 | `start_codon_ATG` | 1 if ATG start, else 0 |
| 6 | `start_codon_GTG` | 1 if GTG start, else 0 |
| 7 | `start_codon_TTG` | 1 if TTG start, else 0 |
| 8 | `gc_content` | G+C fraction of ORF |
| 9 | `frame_fwd` | 1 if forward strand, 0 if reverse |
| 10 | `frame_1` | 1 if reading frame 1 |
| 11 | `frame_2` | 1 if reading frame 2 |
| 12 | `frame_3` | 1 if reading frame 3 |

## Training with Real Data

1. **Export features** from a genome:
   ```bash
   cargo run --release --features ml -- \
     -i your_genome.fasta \
     --export-features notebooks/models/training_features.tsv
   ```

2. **Generate labels** using DIAMOND blastp against known proteins:
   ```bash
   python3 scripts/generate_training_labels.py \
     --genome your_genome.fasta \
     --output notebooks/models/training_labels.tsv
   ```

3. **Merge features and labels**, then train and export:
   ```python
   import pandas as pd
   features = pd.read_csv('notebooks/models/training_features.tsv', sep='\t')
   labels = pd.read_csv('notebooks/models/training_labels.tsv', sep='\t')
   df = features.merge(labels, on='orf_id')
   df.to_csv('notebooks/models/training_data.tsv', sep='\t', index=False)
   ```

4. **Update the export script** to load `training_data.tsv` instead of generating synthetic data.

## Implementation Details

### Why `ort` instead of `tract-onnx`?

We originally used `tract-onnx` for its lightweight, pure-Rust implementation.
However, `tract-onnx` does not support the `TreeEnsembleRegressor` operator
used by scikit-learn and XGBoost ONNX exports. 

The `ort` crate (ONNX Runtime) supports **all** ONNX operators and is the
official Microsoft implementation. It downloads prebuilt binaries automatically.

### Thread Safety

`Session::run()` requires `&mut self` in the `ort` API, but the underlying
ONNX Runtime C session is thread-safe. We wrap the session in a `std::sync::Mutex`
to allow concurrent inference from multiple threads (via `rayon` parallel genome
processing). The mutex is uncontended in practice since inference is very fast.

### Feature Scaling

The exported scaler (`scaler.pkl`) is currently **not used** in Rust inference.
For production models, you should either:
1. Embed scaling parameters in the ONNX model (e.g., using sklearn's `Pipeline`)
2. Or implement StandardScaler in Rust as a preprocessing step

## Troubleshooting

### "Failed to load ONNX model"
- Check the model path is correct
- Verify the model has input shape `[?, 13]` and output shape `[?, 1]`
- Ensure the model is a regressor, not a classifier

### "ONNX model inference failed"
- Check that input features are `float32` (not `float64` or `int64`)
- Verify batch dimension is correct (dynamic batch is supported)

### Architecture mismatch (macOS)
If you see `libomp.dylib` architecture errors when training with XGBoost:
```bash
# Check your Python architecture
python3 -c "import platform; print(platform.machine())"

# If running under Rosetta (x86_64), install x86_64 libomp:
arch -x86_64 brew install libomp
```
