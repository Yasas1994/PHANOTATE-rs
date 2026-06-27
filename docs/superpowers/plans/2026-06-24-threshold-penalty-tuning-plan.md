# Per-Genome Threshold + Penalty Retuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-tune gap/overlap penalties for the new 34-feature model and add per-genome `--auto-threshold` calibration to improve `--model` path F1.

**Architecture:** A new `AutoThresholdMode` enum selects between a fixed threshold, a genome-length-based target gene count, and a score percentile. `OnnxScorer` exposes per-ORF probabilities so `process_genome` can compute an effective threshold before scoring. Penalty ratios remain in `src/penalty_calibration.rs` and are retuned via the existing tuner.

**Tech Stack:** Rust 2021, Python 3, ONNX.

---

### Task 1: Expose per-ORF probability from `OnnxScorer`

**Files:**
- Modify: `src/onnx_scorer.rs:39-84`
- Test: `src/onnx_scorer.rs` unit tests

- [ ] **Step 1: Make `probability` public and add `probability_for_orf`**

Change:

```rust
    fn probability(&self, features: &OrfFeatures) -> f64 {
```

to:

```rust
    pub fn probability(&self, features: &OrfFeatures) -> f64 {
```

Add immediately after the `probability` method:

```rust
    /// Convenience helper: compute the positive-class probability for an ORF.
    pub fn probability_for_orf(&self, orf: &crate::orf::Orf) -> f64 {
        let features = OrfFeatures::from_orf(orf);
        self.probability(&features)
    }
```

- [ ] **Step 2: Compile**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/onnx_scorer.rs
git commit -m "feat(onnx): expose probability_for_orf for threshold calibration"
```

---

### Task 2: Add `--auto-threshold` CLI flag and helper

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Add the enum and CLI args**

Near the top of `src/main.rs` (after the other `use` statements), add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum AutoThresholdMode {
    #[default]
    None,
    Length,
    Percentile,
}
```

Inside the `Cli` struct, add:

```rust
    /// Automatically calibrate the model decision threshold per genome.
    #[arg(long = "auto-threshold", value_enum, default_value = "none")]
    auto_threshold: AutoThresholdMode,
```

- [ ] **Step 2: Add the threshold calibration helper**

Add this function in `src/main.rs`, near `process_genome`:

```rust
fn compute_effective_model_threshold(
    orfs: &[phanotate_rs::orf::Orf],
    orf_model: &phanotate_rs::onnx_scorer::OnnxScorer,
    contig_length: usize,
    mode: AutoThresholdMode,
    cli_threshold: f64,
) -> f64 {
    if mode == AutoThresholdMode::None || orfs.is_empty() {
        return cli_threshold;
    }

    let mut probs: Vec<f64> = orfs
        .iter()
        .map(|o| orf_model.probability_for_orf(o))
        .collect();
    probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let computed = match mode {
        AutoThresholdMode::Length => {
            let genes_per_kb: f64 = std::env::var("PHANOTATE_MODEL_GENES_PER_KB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let target = ((contig_length as f64 / 1000.0) * genes_per_kb).round() as usize;
            let idx = target.saturating_sub(1).min(probs.len().saturating_sub(1));
            probs[idx]
        }
        AutoThresholdMode::Percentile => {
            let pct: f64 = std::env::var("PHANOTATE_MODEL_THRESHOLD_PERCENTILE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(80.0);
            let idx = ((pct / 100.0) * probs.len().saturating_sub(1) as f64)
                .round()
                .min(probs.len().saturating_sub(1) as f64) as usize;
            probs[idx]
        }
        AutoThresholdMode::None => cli_threshold,
    };

    computed.max(cli_threshold)
}
```

- [ ] **Step 3: Use the effective threshold when scoring ORFs**

In `process_genome`, after ORFs are scored with the default heuristic (but before graph construction), insert:

```rust
    let effective_model_threshold = orf_model.map_or(cli.model_threshold, |m| {
        compute_effective_model_threshold(
            &orfs,
            m,
            contig_length,
            cli.auto_threshold,
            cli.model_threshold,
        )
    });
```

Then change the ORF scoring loop to use `effective_model_threshold`:

```rust
        orf.score(
            start_codons_map,
            orf_model,
            cli.model_scale,
            effective_model_threshold,
        );
```

- [ ] **Step 4: Compile and run a quick smoke test**

```bash
cargo build --release
./target/release/phanotate-rs --help | grep -A1 auto-threshold
```

Expected: help text shows `--auto-threshold` with `none|length|percentile`.

- [ ] **Step 5: Run a quick benchmark**

```bash
.venv/bin/python - <<'PY'
import subprocess
r = subprocess.run(['target/release/phanotate-rs','-i','tests/golden/annotgenomes/NC_000866.gb','--detect-table','--yes','-f','sco','--model','tests/golden/orf_model.onnx','--auto-threshold','length'], stdout=subprocess.PIPE, text=True)
print(len(r.stdout.splitlines()))
PY
```

Expected: command runs and prints a gene count.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --auto-threshold per-genome calibration"
```

---

### Task 3: Re-tune gap/overlap penalties for the new model

**Files:**
- Modify: `src/penalty_calibration.rs`
- Tool: `scripts/tune_penalty_balance.py`

- [ ] **Step 1: Ensure release binary is built with the new model**

```bash
cargo build --release
```

- [ ] **Step 2: Run the penalty tuner**

```bash
.venv/bin/python scripts/tune_penalty_balance.py
```

Wait for it to finish. It will print the best `(gap_target, overlap_target)` and F1.

- [ ] **Step 3: Update the constants in `src/penalty_calibration.rs`**

Replace:

```rust
const MODEL_GAP_TARGET_RATIO: f64 = 0.5;
const MODEL_OVERLAP_TARGET_RATIO: f64 = 0.3;
```

with the values reported by the tuner.

- [ ] **Step 4: Commit**

```bash
git add src/penalty_calibration.rs
git commit -m "feat(penalty): retune gap/overlap ratios for 34-feature model"
```

---

### Task 4: Create the threshold tuner

**Files:**
- Create: `scripts/tune_threshold.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Grid-search per-genome auto-threshold parameters."""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from compare_predictions import _parse_genbank_cds, compare

ANNOT_DIR = Path("tests/golden/annotgenomes")
BINARY = Path("target/release/phanotate-rs")
MODEL = Path("tests/golden/orf_model.onnx")


def paths(limit: Optional[int] = None) -> list[Path]:
    p = sorted(ANNOT_DIR.glob("*.gb"))
    if limit:
        p = p[:limit]
    return p


def evaluate(mode: str, param: float, genomes: list[Path]) -> dict:
    env = os.environ.copy()
    if mode == "length":
        env["PHANOTATE_MODEL_GENES_PER_KB"] = str(param)
    elif mode == "percentile":
        env["PHANOTATE_MODEL_THRESHOLD_PERCENTILE"] = str(param)

    agg = {"tp": 0, "fp": 0, "fn": 0}
    for genome in genomes:
        cmd = [
            str(BINARY),
            "-i",
            str(genome),
            "--detect-table",
            "--yes",
            "-f",
            "sco",
            "--model",
            str(MODEL),
            "--auto-threshold",
            mode,
        ]
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        ref = _parse_genbank_cds(str(genome))
        pred = set()
        for line in r.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                pred.add((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
        c = compare(pred, ref, 3)
        for k in ("tp", "fp", "fn"):
            agg[k] += c[k]
    p = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
    r = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, **agg}


def main() -> None:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}", file=sys.stderr)
        sys.exit(1)
    genomes = paths(50)
    print(f"Tuning on {len(genomes)} genomes")
    best = None

    for genes_per_kb in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
        res = evaluate("length", genes_per_kb, genomes)
        print(f"length genes_per_kb={genes_per_kb:.2f} -> F1={res['f1']:.4f}")
        if best is None or res["f1"] > best["f1"]:
            best = {"mode": "length", "param": genes_per_kb, **res}

    for pct in [70.0, 75.0, 80.0, 85.0, 90.0]:
        res = evaluate("percentile", pct, genomes)
        print(f"percentile pct={pct:.1f} -> F1={res['f1']:.4f}")
        if best is None or res["f1"] > best["f1"]:
            best = {"mode": "percentile", "param": pct, **res}

    # baseline fixed threshold
    res = evaluate("none", 0.0, genomes)
    print(f"none -> F1={res['f1']:.4f}")

    print("\nBest:")
    print(f"  mode={best['mode']}")
    print(f"  param={best['param']}")
    print(f"  F1={best['f1']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/tune_threshold.py
git add scripts/tune_threshold.py
git commit -m "feat(scripts): add grid-search tuner for per-genome threshold"
```

---

### Task 5: Run the threshold tuner and set defaults

**Files:**
- Modify: `src/main.rs` defaults for `PHANOTATE_MODEL_GENES_PER_KB` and `PHANOTATE_MODEL_THRESHOLD_PERCENTILE`

- [ ] **Step 1: Run the tuner**

```bash
.venv/bin/python scripts/tune_threshold.py
```

Wait for it to finish. Note the best mode and parameter.

- [ ] **Step 2: Update the default constants in `compute_effective_model_threshold`**

In `src/main.rs`, update the `unwrap_or(...)` defaults to the tuned values, e.g.:

```rust
.unwrap_or(1.15)
```

and/or

```rust
.unwrap_or(80.0)
```

Also update the doc comment for `--auto-threshold` to mention the default mode if it is no longer `none`.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): set learned per-genome threshold defaults"
```

---

### Task 6: Verification and final benchmark

**Files:**
- All modified Rust files
- `tests/golden/orf_model.onnx`

- [ ] **Step 1: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean.

- [ ] **Step 2: Run Rust tests**

```bash
cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_
cargo test --test cli_tests
```

Expected: all pass except the pre-existing external-file detect_table failures.

- [ ] **Step 3: Run Python tests**

```bash
unset CONDA_PREFIX
VIRTUAL_ENV=.venv .venv/bin/maturin develop --features python
.venv/bin/pytest tests/test_python_bindings.py -q
```

Expected: 59 passed.

- [ ] **Step 4: Final benchmark**

```bash
.venv/bin/python - <<'PY'
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from compare_predictions import _parse_genbank_cds, compare

paths = sorted(Path('tests/golden/annotgenomes').glob('*.gb'))[:50]
model = 'tests/golden/orf_model.onnx'

def run(auto_mode='none'):
    agg = {'tp':0,'fp':0,'fn':0}
    for p in paths:
        cmd = ['target/release/phanotate-rs','-i',str(p),'--detect-table','--yes','-f','sco','--model',model]
        if auto_mode != 'none':
            cmd += ['--auto-threshold', auto_mode]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        ref = _parse_genbank_cds(str(p))
        pred = set()
        for line in r.stdout.splitlines():
            if line.startswith('#') or not line.strip(): continue
            x = line.split('\t')
            if len(x) >= 2:
                try: pred.add((int(x[0]), int(x[1])))
                except: pass
        c = compare(pred, ref, 3)
        for k in ('tp','fp','fn'): agg[k] += c[k]
    p = agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
    r = agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
    return 2*p*r/(p+r) if p+r else 0, p, r

for m in ['none','length','percentile']:
    f1, p, r = run(m)
    print(f"{m:10s} F1={f1:.4f} P={p:.4f} R={r:.4f}")
PY
```

Expected: the best auto-threshold mode beats `none` (0.7025).

- [ ] **Step 5: Commit verification**

```bash
git add -A
git commit -m "test(threshold): verify per-genome calibration improves F1"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every design requirement maps to a task.
- [ ] **No placeholders:** Every task shows exact code/commands.
- [ ] **Type consistency:** `AutoThresholdMode`, `compute_effective_model_threshold`, and `OnnxScorer` signatures are consistent across tasks.
- [ ] **Test coverage:** Unit tests, integration tests, Python tests, and benchmark all included.
