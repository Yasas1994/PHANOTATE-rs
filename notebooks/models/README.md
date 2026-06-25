# ONNX Model Export (Historical)

> **This directory is historical.** PHANOTATE-rs no longer uses ONNX Runtime for
> ORF scoring. The `--ml-model` flag and the `ml` Cargo feature have been removed.
>
> The current learned scorer is a lightweight JSON logistic-regression model
> loaded via `--model`. See:
> - `scripts/train_orf_score_model.py` — train a JSON model from annotated GenBank files
> - `src/orf_score_model.rs` — Rust runtime loader and scorer
> - `notebooks/README.md` — updated training documentation
>
> Old `.onnx` files and export scripts (`export_tree_onnx.py`,
> `export_regressor_onnx.py`, `export_pytorch_onnx.py`) are kept for reference
> but are not consumed by the current binary.
