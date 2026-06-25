# XGBoost Notebook Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Random Forest model in `notebooks/01_orf_score_model.ipynb` with an XGBoost classifier and include it in the benchmark comparison.

**Architecture:** Add an ONNX export helper for XGBoost, swap the RF training cell for XGBoost, and update the final benchmark cell to evaluate the XGBoost model. No Rust code changes.

**Tech Stack:** Python, Jupyter, xgboost, onnxmltools, phanotate-rs CLI.

---

### Task 1: Install xgboost in the venv

**Files:** none (environment change)

- [ ] **Step 1: Install xgboost**

  Run:
  ```bash
  cd /Users/javis/Documents/UMG/PHANOTATE-rs
  unset CONDA_PREFIX && . .venv/bin/activate && pip install xgboost
  ```

- [ ] **Step 2: Verify import**

  Run:
  ```bash
  unset CONDA_PREFIX && . .venv/bin/activate && python -c "from xgboost import XGBClassifier; print('xgboost ok')"
  ```
  Expected: `xgboost ok`

---

### Task 2: Add export helper for XGBoost

**Files:**
- Modify: `notebooks/01_orf_score_model.ipynb`

- [ ] **Step 1: Insert `export_xgboost_to_onnx()` next to `export_rf_to_onnx()`**

  Add a new code cell or merge into the existing export-helper cell:
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

- [ ] **Step 2: Remove `export_rf_to_onnx()` and the RF import if they are no longer referenced**

  If the RF cell is replaced, the RF export helper can also be removed to keep the notebook clean.

---

### Task 3: Replace Random Forest training cell with XGBoost

**Files:**
- Modify: `notebooks/01_orf_score_model.ipynb`

- [ ] **Step 1: Replace the RF training cell source with XGBoost**

  ```python
  from xgboost import XGBClassifier

  # Train an XGBoost classifier on all genomes (raw, unstandardised features)
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

- [ ] **Step 2: Update any markdown heading for the cell from "Random forest" to "XGBoost"**

---

### Task 4: Update CV loop to evaluate XGBoost instead of Random Forest

**Files:**
- Modify: `notebooks/01_orf_score_model.ipynb`

- [ ] **Step 1: Replace the RandomForestClassifier in the CV loop with XGBClassifier**

  In the CV cell, change:
  ```python
  rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
  rf.fit(X_train, y_train)
  val_prob_rf = rf.predict_proba(X_val)[:, 1]
  ```
  to:
  ```python
  xgb_fold = XGBClassifier(
      n_estimators=100,
      max_depth=6,
      learning_rate=0.1,
      scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
      n_jobs=-1,
      random_state=42,
      eval_metric="logloss",
  )
  xgb_fold.fit(X_train, y_train)
  val_prob_xgb = xgb_fold.predict_proba(X_val)[:, 1]
  ```

- [ ] **Step 2: Rename metrics keys from `rf_auc` to `xgb_auc`**

  Update:
  ```python
  orf_metrics = {
      ...
      "lr_auc": roc_auc_score(y_val, val_prob_lr),
      "xgb_auc": roc_auc_score(y_val, val_prob_xgb),
  }
  ```

---

### Task 5: Update final benchmark cell

**Files:**
- Modify: `notebooks/01_orf_score_model.ipynb`

- [ ] **Step 1: Replace RF predictions with XGBoost predictions**

  In the comparison cell, change:
  ```python
  rf_pred = predict_genes(example_genome, rf_path, example_table)
  ```
  to:
  ```python
  xgb_pred = predict_genes(example_genome, xgb_path, example_table)
  ```

- [ ] **Step 2: Update print statements**

  Replace:
  ```python
  print("Random forest:          ", gene_metrics(rf_pred, true_set))
  ```
  with:
  ```python
  print("XGBoost:                ", gene_metrics(xgb_pred, true_set))
  ```

- [ ] **Step 3: Update NC_001365 benchmark cell similarly**

  Replace `nc_rf = predict_genes(..., rf_path, ...)` with `nc_xgb = predict_genes(..., xgb_path, ...)` and update the print/table row from "Random forest" to "XGBoost".

---

### Task 6: Run the notebook end-to-end

**Files:** none

- [ ] **Step 1: Execute the notebook**

  Run:
  ```bash
  cd /Users/javis/Documents/UMG/PHANOTATE-rs/notebooks
  unset CONDA_PREFIX && . ../.venv/bin/activate
  cargo build --release
  jupyter nbconvert --to notebook --execute --inplace 01_orf_score_model.ipynb
  ```
  Expected: exit code 0.

- [ ] **Step 2: Verify the benchmark output includes XGBoost rows**

  Check the final comparison cells in the executed notebook show XGBoost precision/recall/F1.

---

### Task 7: Commit changes

**Files:**
- `notebooks/01_orf_score_model.ipynb`
- `docs/superpowers/specs/2026-06-24-xgboost-notebook-integration-design.md`
- `docs/superpowers/plans/2026-06-24-xgboost-notebook-integration.md`

- [ ] **Step 1: Stage and commit**

  ```bash
  git add notebooks/01_orf_score_model.ipynb docs/superpowers/specs/2026-06-24-xgboost-notebook-integration-design.md docs/superpowers/plans/2026-06-24-xgboost-notebook-integration.md
  git commit -m "feat(notebook): replace Random Forest with XGBoost ORF score model"
  ```

---

## Spec coverage check

- ONNX export helper for XGBoost → Task 2
- XGBoost training on all data → Task 3
- CV loop uses XGBoost → Task 4
- Final benchmark includes XGBoost → Task 5
- No Rust changes → no Rust tasks
- Notebook executes end-to-end → Task 6
