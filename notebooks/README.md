# PHANOTATE-rs ML Training Notebooks

This directory contains Jupyter notebooks for experimenting with learned ORF
scoring models for PHANOTATE-rs.

> **Note:** The runtime ONNX/ML scorer (`--ml-model`) has been removed. The
> current pipeline uses the lightweight JSON model loaded via `--model`.
> For a ready-to-use training script, see `scripts/train_orf_score_model.py`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build PHANOTATE-rs
cd ..
cargo build --release

# 3. Launch Jupyter
jupyter notebook

# 4. Open `01_orf_score_model.ipynb` and run all cells
```

## Notebooks

### `01_orf_score_model.ipynb`

End-to-end experimental pipeline:

1. **Load** ORF features and labels from annotated GenBank files.
2. **Explore** feature distributions, correlations, and label relationships.
3. **Train** candidate models (Logistic Regression, Random Forest, XGBoost, etc.).
4. **Evaluate** genome-stratified cross-validation with ROC/PR curves.
5. **Export** a JSON logistic-regression model compatible with `--model`.
6. **Validate** end-to-end by running PHANOTATE-rs with the model and comparing
   predictions to the GenBank annotations.

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

The recommended data source is a directory of annotated GenBank files (one per
genome). The notebook uses `src/ml_features.rs` (via `--export-features`) to
extract per-ORF features and derives labels from the `CDS` coordinates in each
GenBank record.

## Cross-Validation Workflow

The notebook performs **genome-stratified** cross-validation:

1. Split genomes into training and validation folds (not individual ORFs).
2. Train a model on the training genomes.
3. Export the model to a JSON file compatible with `--model`.
4. Run `phanotate-rs --model <json>` on each validation genome.
5. Compare predicted genes to the GenBank `CDS` annotations using start/stop
   coordinate overlap.

This workflow prevents information leakage across genomes and measures the
model's impact on the full gene-calling pipeline, not just ORF classification.

## Model Export

The notebook exports a JSON file with the standardised coefficients, means, and
standard deviations required by `src/orf_score_model.rs`:

```json
{
  "version": 1,
  "num_features": 14,
  "coeffs": [...],
  "mean": [...],
  "std": [...]
}
```

Use it with PHANOTATE-rs:

```bash
phanotate-rs -i genome.fasta --model model.json -f sco
```

## Outputs

Each run produces under `outputs/`:

- `feature_distributions.png` — Histogram of each feature
- `feature_correlations.png` — Correlation heatmap
- `features_by_label.png` — Boxplots by gene/non-gene
- `model_comparison.png` — ROC and PR curves across candidate models
- `model_*.json` — Exported JSON models compatible with `--model`
- `experiment_summary.json` — Metrics and metadata

## Tips

- **Start small**: Use a few phage genomes for initial experiments.
- **Genome-stratified CV is essential**: ORFs from the same genome are highly
  correlated; splitting by genome gives a realistic estimate.
- **Model selection**: Only the exported logistic-regression JSON is natively
  consumed by `--model`, but you can benchmark any classifier/regressor inside
  the notebook and re-train the best-performing linear model for export.
- **Scoring range**: The Rust scorer clamps the learned log-odds score to
  `[-10, 10]`.
