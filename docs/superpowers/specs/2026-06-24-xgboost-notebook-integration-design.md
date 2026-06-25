# XGBoost ORF Score Model — Notebook Integration Design

## Goal
Add an XGBoost classifier as a learned ORF-scoring candidate in the `notebooks/01_orf_score_model.ipynb` benchmark, replacing the current Random Forest cell.

## Background
The notebook currently trains two learned models:
- Logistic regression (standardised features, exported to ONNX).
- Random forest (raw features, exported to ONNX).

On T4 the random forest underperforms the default PHANOTATE-rs heuristic, while on NC_001365 it improves over the heuristic. XGBoost is a stronger tree-based model and may outperform Random Forest on the same feature set.

## Changes

### 1. Dependency
Install `xgboost` in the project venv:

```bash
pip install xgboost
```

### 2. ONNX export helper
Add `export_xgboost_to_onnx()` next to the existing `export_rf_to_onnx()`:

```python
def export_xgboost_to_onnx(model, path: Path, n_features: int = NUM_FEATURES) -> None:
    """Export an XGBoost classifier to ONNX."""
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxmltools

    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, n_features]))]
    )
    path.write_bytes(onnx_model.SerializeToString())
```

### 3. Model training cell
Replace the `RandomForestClassifier` training cell with `XGBClassifier`:

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_all) - y_all.sum()) / y_all.sum(),
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
)
xgb.fit(X_all, y_all)

xgb_path = Path("outputs/model_xgb.onnx")
export_xgboost_to_onnx(xgb, xgb_path)
print(f"Exported XGBoost model to {xgb_path}")

# Feature importances
importance_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": xgb.feature_importances_})
importance_df = importance_df.sort_values("importance", ascending=False)
print(importance_df)
```

### 4. Final benchmark cell
Update the comparison cell to evaluate:
- Default PHANOTATE-rs 0.1.3
- PHANOTATE-rs 0.1.2
- PHANOTATE-rs (conda)
- Logistic regression
- **XGBoost** (instead of Random forest)
- PHANOTATE.py
- prodigal-gv

## Non-goals
- No changes to the Rust code.
- No changes to the 15-feature vector.
- No changes to the CV loop beyond replacing RF with XGB for ORF-level AUC comparison.
- No changes to the default heuristic.

## Success criteria
- Notebook executes end-to-end without errors.
- `outputs/model_xgb.onnx` is produced and loads with PHANOTATE-rs `--model`.
- Benchmark table includes XGBoost results.
