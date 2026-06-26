# PHANOTATE-rs ML Training Notebooks

This directory contains Jupyter notebooks for experimenting with learned ORF
scoring models for PHANOTATE-rs.

> **Note:** `--model` accepts ONNX models only. For a ready-to-use training
> script, see `scripts/train_orf_score_model.py`.

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
3. **Train** candidate models (Logistic Regression, XGBoost, etc.).
4. **Evaluate** genome-stratified cross-validation with ROC/PR curves.
5. **Export** the best model to ONNX for use with `--model`.
6. **Validate** end-to-end by running PHANOTATE-rs with the model and comparing
   predictions to the GenBank annotations.

## Feature Description

The 34 features extracted per ORF (from `src/ml_features.rs`):

| Feature | Description | Range |
|---------|-------------|-------|
| `log_length` | Natural log of ORF length (nt) | ~4–10 |
| `rbs_bin` | RBS bin / raw Shine–Dalgarno score (0–27) | varies |
| `log_hold` | Log of GC frame plot product | varies |
| `pstop` | Stop codon probability | 0–1 |
| `sd_rbs_score` | SD RBS likelihood ratio | varies |
| `start_codon_atg` | 1 if start is ATG, else 0 | 0 or 1 |
| `start_codon_gtg` | 1 if start is GTG, else 0 | 0 or 1 |
| `start_codon_ttg` | 1 if start is TTG, else 0 | 0 or 1 |
| `gc_content` | G+C fraction in ORF | 0–1 |
| `frame_fwd` | 1 if forward strand, else 0 | 0 or 1 |
| `frame_1` | 1 if \|frame\| == 1, else 0 | 0 or 1 |
| `frame_2` | 1 if \|frame\| == 2, else 0 | 0 or 1 |
| `frame_3` | 1 if \|frame\| == 3, else 0 | 0 or 1 |
| `non_sd_rbs_score` | Non-SD motif score | varies |
| `dicodon_log_likelihood` | Natural log of Prodigal-style dicodon coding potential | varies |
| `cai` | Codon adaptation index vs. genome-wide usage | 0–1 |
| `gc1` | GC content at codon position 1 | 0–1 |
| `gc2` | GC content at codon position 2 | 0–1 |
| `gc3` | GC content at codon position 3 | 0–1 |
| `overlap_upstream_length` | Bases overlapping nearest upstream ORF | ≥0 |
| `overlap_upstream_same_strand` | 1 if overlapping upstream ORF is same strand | 0 or 1 |
| `overlap_downstream_length` | Bases overlapping nearest downstream ORF | ≥0 |
| `overlap_downstream_same_strand` | 1 if overlapping downstream ORF is same strand | 0 or 1 |
| `stop_sharing_count` | Number of ORFs sharing this stop codon | ≥1 |
| `gc_skew` | (G−C)/(G+C) of the ORF sequence | −1–1 |
| `truncation_penalty` | Prodigal-style sharpening penalty | varies |
| `upstream_pwm_score` | Log-likelihood of start upstream region | varies |
| `rbs_spacer` | Bases between detected RBS motif and start codon | varies |
| `heuristic_score` | PHANOTATE-style heuristic score | varies |
| `best_alt_pwm_score` | Highest PWM score among alternative starts for the same stop | varies |
| `pwm_ratio` | `upstream_pwm_score / best_alt_pwm_score` | ≥0 |
| `start_rank` | Rank of chosen start by PWM among alternatives (1 = best) | ≥1 |
| `num_alt_starts` | Number of alternative starts considered | ≥1 |
| `start_codon_log_freq` | Log frequency of start codon among high-confidence ORFs | ≤0 |

## Training Data

The recommended data source is a directory of annotated GenBank files (one per
genome). The notebook uses `src/ml_features.rs` (via `--export-features`) to
extract per-ORF features and derives labels from the `CDS` coordinates in each
GenBank record.

## Cross-Validation Workflow

The notebook performs **genome-stratified** cross-validation:

1. Split genomes into training and validation folds (not individual ORFs).
2. Train a model on the training genomes.
3. Export the model to ONNX.
4. Run `phanotate-rs --model <onnx>` on each validation genome.
5. Compare predicted genes to the GenBank `CDS` annotations using start/stop
   coordinate overlap.

This workflow prevents information leakage across genomes and measures the
model's impact on the full gene-calling pipeline, not just ORF classification.

## Model Export

The notebook exports an ONNX model consumed by `src/onnx_scorer.rs`:

```bash
phanotate-rs -i genome.fasta --model outputs/model_final.onnx -f sco
```

Linear models are exported via `skl2onnx`; XGBoost models are exported via
`onnxmltools`.

## Outputs

Each run produces under `outputs/`:

- `feature_distributions.png` — Histogram of each feature
- `feature_correlations.png` — Correlation heatmap
- `features_by_label.png` — Boxplots by gene/non-gene
- `model_comparison.png` — ROC and PR curves across candidate models
- `model_final.onnx` — Exported ONNX model compatible with `--model`
- `experiment_summary.json` — Metrics and metadata

## Tips

- **Start small**: Use a few phage genomes for initial experiments.
- **Genome-stratified CV is essential**: ORFs from the same genome are highly
  correlated; splitting by genome gives a realistic estimate.
- **Model selection**: Any classifier that can be exported to ONNX can be used
  at runtime. Benchmark multiple models inside the notebook, then export the
  best performer to ONNX.
- **Scoring range**: The Rust scorer clamps the learned log-odds score to
  `[-10, 10]`.
