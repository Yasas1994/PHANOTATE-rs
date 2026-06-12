# PHANOTATE-rs ML Training Notebooks

This directory contains Jupyter notebooks for training and evaluating machine learning models that predict ORF score adjustments for PHANOTATE-rs.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build PHANOTATE-rs with ML support
cd ..
cargo build --release --features ml

# 3. Generate training labels (using UniProt + DIAMOND)
#    This creates labels.tsv by comparing ORFs against reviewed phage proteins
python scripts/generate_training_labels.py -i genome.fasta -o notebooks/labels.tsv

# 4. Generate training features
./target/release/phanotate-rs -i genome.fasta --export-features notebooks/features.tsv

# 5. Launch Jupyter
jupyter notebook

# 6. Open `01_train_hybrid_scorer.ipynb` and run all cells
```

## Notebooks

### `01_train_hybrid_scorer.ipynb`

End-to-end training pipeline:

1. **Load & inspect** ORF features exported from PHANOTATE-rs
2. **Explore** feature distributions, correlations, and label relationships
3. **Train** multiple models (Logistic Regression, Random Forest, XGBoost)
4. **Evaluate** with ROC, PR curves, and calibration plots
5. **Export** the best model to ONNX for Rust inference
6. **Validate** end-to-end by running PHANOTATE-rs with the trained model

## Feature Description

The 13 features extracted per ORF (from `src/ml_features.rs`):

| Feature | Description | Range |
|---------|-------------|-------|
| `log_length` | Natural log of ORF length (nt) | ~4–10 |
| `rbs_score_norm` | Shine-Dalgarno score / 27 | 0–1 |
| `log_hold` | Log of GC frame plot product | varies |
| `pstop` | Stop codon probability | 0–1 |
| `weight_rbs_log` | Log of RBS likelihood ratio | varies |
| `start_codon_atg` | 1 if start is ATG, else 0 | 0 or 1 |
| `start_codon_gtg` | 1 if start is GTG, else 0 | 0 or 1 |
| `start_codon_ttg` | 1 if start is TTG, else 0 | 0 or 1 |
| `gc_content` | G+C fraction in ORF | 0–1 |
| `frame_fwd` | 1 if forward strand, else 0 | 0 or 1 |
| `frame_1` | 1 if \|frame\| == 1, else 0 | 0 or 1 |
| `frame_2` | 1 if \|frame\| == 2, else 0 | 0 or 1 |
| `frame_3` | 1 if \|frame\| == 3, else 0 | 0 or 1 |

## Training Data

### Automated Label Generation (Recommended)

The easiest way to generate labels is using the provided script, which downloads reviewed phage proteins from UniProt and uses DIAMOND to label ORFs with significant hits:

```bash
# Full pipeline: download UniProt proteins, run DIAMOND, generate labels
python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv

# Use existing UniProt download (faster for subsequent runs)
python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv \
    --uniprot phage_reviewed.fasta

# Stricter thresholds for higher-confidence labels
python scripts/generate_training_labels.py -i genome.fasta -o labels.tsv \
    --evalue 1e-5 --identity 50 --query-cover 60
```

This script:
1. Downloads ~1,600 reviewed bacteriophage proteins from UniProt
2. Builds a DIAMOND database
3. Finds all ORFs in your genome (matching PHANOTATE-rs logic)
4. Translates ORFs and runs DIAMOND blastp against phage proteins
5. Labels ORFs with hits as genes (`1`), others as non-genes (`0`)

### With Manual Labels

If you have annotated genomes (e.g., RefSeq with curated gene predictions):

1. Run PHANOTATE-rs on each genome to get ORF features:
   ```bash
   phanotate-rs -i genome.fasta --export-features features.tsv
   ```

2. Create a label file with matching rows and an `is_gene` column:
   ```bash
   # is_gene: 1 = true gene, 0 = false ORF
   echo -e "is_gene\n1\n0\n1\n..." > labels.tsv
   ```

3. The notebook will use these labels for supervised training.

### Without Labels (Pseudo-Labels)

If you don't have annotations, the notebook can generate pseudo-labels based on heuristic score quartiles. This is less accurate but lets you experiment with the pipeline.

## Model Export

The notebook exports trained models to ONNX format, which PHANOTATE-rs loads at runtime:

```python
# In the notebook
export_regressor_to_onnx(model, FEATURE_NAMES, "model.onnx")
```

Then use with PHANOTATE-rs:

```bash
phanotate-rs -i genome.fasta --ml-model model.onnx
```

## Outputs

Each run produces:

- `outputs/feature_distributions.png` — Histogram of each feature
- `outputs/feature_correlations.png` — Correlation heatmap
- `outputs/features_by_label.png` — Boxplots by gene/non-gene
- `outputs/rf_importances.png` — Random Forest feature importance
- `outputs/xgb_importances.png` — XGBoost feature importance
- `outputs/model_comparison.png` — ROC and PR curves
- `outputs/model_*.onnx` — Exported ONNX models
- `outputs/experiment_summary.json` — Metrics and metadata

## Tips

- **Start small**: Use a single phage genome (~5kb) for initial experiments
- **Feature engineering**: Try adding codon usage bias, protein length, or amino acid composition
- **Model selection**: XGBoost regressor works best for the hybrid scorer; Random Forest is more interpretable
- **Adjustment bounds**: The Rust scorer clamps adjustments to [0.5, 2.0], so the model only tweaks the heuristic
