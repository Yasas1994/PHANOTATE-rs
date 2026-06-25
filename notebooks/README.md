# PHANOTATE-rs ML Training Notebooks

This directory contains Jupyter notebooks for training and evaluating machine learning models that predict ORF scores for PHANOTATE-rs.

> **Note:** The runtime ONNX/ML scorer (`--ml-model`) has been removed. The current pipeline uses the lightweight JSON model loaded via `--model`. For a ready-to-use training script, see `scripts/train_orf_score_model.py`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build PHANOTATE-rs
cd ..
cargo build --release

# 3. Train a JSON model directly from annotated GenBank files
python scripts/train_orf_score_model.py -i annotated_genomes/ -o notebooks/model.json

# 4. Use the model
./target/release/phanotate-rs -i genome.fasta --model notebooks/model.json -f sco

# 5. Launch Jupyter for exploratory training
jupyter notebook

# 6. Open `01_train_hybrid_scorer.ipynb` and run all cells
```

## Notebooks

### `01_train_hybrid_scorer.ipynb`

End-to-end exploratory training pipeline:

1. **Load & inspect** ORF features exported from PHANOTATE-rs
2. **Explore** feature distributions, correlations, and label relationships
3. **Train** multiple models (Logistic Regression, Random Forest, XGBoost)
4. **Evaluate** with ROC, PR curves, and calibration plots
5. **Export** a JSON logistic-regression model compatible with `--model`
6. **Validate** end-to-end by running PHANOTATE-rs with the trained model

## Feature Description

The 14 features extracted per ORF (from `src/ml_features.rs`):

| Feature | Description | Range |
|---------|-------------|-------|
| `log_length` | Natural log of ORF length (nt) | ~4–10 |
| `rbs_score_norm` | Shine-Dalgarno score / 27 | 0–1 |
| `log_hold` | Log of GC frame plot product | varies |
| `pstop` | Stop codon probability | 0–1 |
| `log_sd_rbs_score` | Log of SD RBS likelihood ratio | varies |
| `start_codon_atg` | 1 if start is ATG, else 0 | 0 or 1 |
| `start_codon_gtg` | 1 if start is GTG, else 0 | 0 or 1 |
| `start_codon_ttg` | 1 if start is TTG, else 0 | 0 or 1 |
| `gc_content` | G+C fraction in ORF | 0–1 |
| `frame_fwd` | 1 if forward strand, else 0 | 0 or 1 |
| `frame_1` | 1 if \|frame\| == 1, else 0 | 0 or 1 |
| `frame_2` | 1 if \|frame\| == 2, else 0 | 0 or 1 |
| `frame_3` | 1 if \|frame\| == 3, else 0 | 0 or 1 |
| `log_non_sd_rbs_score` | Natural log of non-SD motif score | varies |

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

The recommended path is to use `scripts/train_orf_score_model.py`, which exports the standardised coefficients, means, and standard deviations as JSON:

```bash
python scripts/train_orf_score_model.py -i annotated_genomes/ -o model.json
```

Then use with PHANOTATE-rs:

```bash
phanotate-rs -i genome.fasta --model model.json -f sco
```

The notebook can also produce a JSON logistic-regression model for `--model`; ONNX export is no longer supported by the runtime.

## Outputs

Each run produces:

- `outputs/feature_distributions.png` — Histogram of each feature
- `outputs/feature_correlations.png` — Correlation heatmap
- `outputs/features_by_label.png` — Boxplots by gene/non-gene
- `outputs/rf_importances.png` — Random Forest feature importance
- `outputs/xgb_importances.png` — XGBoost feature importance
- `outputs/model_comparison.png` — ROC and PR curves
- `outputs/model_*.json` — Exported JSON models compatible with `--model`
- `outputs/experiment_summary.json` — Metrics and metadata

## Tips

- **Start small**: Use a single phage genome (~5kb) for initial experiments
- **Feature engineering**: Try adding codon usage bias, protein length, or amino acid composition
- **Model selection**: Logistic regression is the format natively consumed by `--model`; the notebook can train and export it
- **Scoring range**: The Rust scorer clamps the learned log-odds score to [-10, 10]
