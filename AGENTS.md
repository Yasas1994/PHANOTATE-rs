# PHANOTATE-rs — Agent Guide

> This file is written for AI coding agents. It assumes you know nothing about the project. All facts below are derived from the actual source tree, not from general knowledge.

---

## 1. Project Overview

**PHANOTATE-rs** is a fast Rust reimplementation of **PHANOTATE**, a gene caller optimised for bacteriophage genomes. It ships as a command-line tool, a Rust library (`phanotate_rs`), and a Python package (`phanotate-rs`) distributed via GitHub Releases, PyPI, Cargo, Bioconda, and Homebrew.

* **Repository**: `https://github.com/Yasas1994/PHANOTATE-rs`
* **License**: GPL-3.0 (see `LICENSE`; `Cargo.toml` uses `GPL-3.0`, Conda recipe uses `GPL-3.0-only`)
* **Current version**: `0.1.3` (tracked in `VERSION`, `Cargo.toml`, `pyproject.toml`)
* **MSRV**: Rust 1.70
* **Authors**: Yasas Wijesekara, Lars Kaderali (University of Greifswald)
* **Python requirement**: `pyproject.toml` specifies `requires-python = ">=3.8"`; the README advertises Python 3.9+

### What it does
1. Reads FASTA or GenBank input.
2. Enumerates all ORFs in six reading frames using a selected NCBI translation table.
3. Scores Shine–Dalgarno (RBS) motifs with a byte-based pattern matcher.
4. Computes a GC frame plot to distinguish coding vs non-coding regions.
5. Builds a weighted directed graph of ORFs, gaps, and overlaps.
6. Finds the maximum-weight path (shortest path on negated weights) via topological relaxation, falling back to Bellman-Ford when cycles exist.
7. Writes predicted genes in GenBank (`gbk`), GFF3 (`gff`), or simple coordinate (`sco`) format, plus optional protein (`-a`) and nucleotide (`-d`) FASTA side outputs.
8. Optionally detects the most likely genetic code automatically (`--detect-table`, `--detect-table-batch`).
9. Optionally adjusts ORF scores with an ONNX regression model (`--ml-model`, feature `ml`).

### Key differentiators from the original Python PHANOTATE
* ~76× faster single-threaded and ~386× faster multi-threaded on the 100-genome validation set.
* Six supported NCBI translation tables (1, 4, 6, 11, 15, 25); automatic detection among tables 1, 4, 11, 15, 25.
* Optional ML-adjusted ORF scoring via ONNX models.
* Full Python API via PyO3 bindings.

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Core language | Rust (edition 2021) | Algorithm implementation |
| CLI framework | `clap` (derive feature) | Argument parsing |
| Error handling | `anyhow` | Ergonomic error propagation |
| Stdin detection | `atty` | Distinguish pipe from terminal input |
| Parallelism | `rayon` + `indicatif` | Multi-threaded contig processing with progress bars |
| Big integers | `num-bigint` | Graph edge weights (prevents overflow on long ORFs) |
| Python bindings | `pyo3` 0.22 (optional, feature `python`) | `import phanotate_rs` |
| ML inference | `ort` 2.0.0-rc.12 + `ndarray` (optional, feature `ml`) | Hybrid ORF scoring |
| Dev benchmarks | `criterion` | Criterion.rs benchmark harness |
| Dev assertions | `pretty_assertions`, `tempfile` | Integration test helpers |
| Python packaging | `maturin` | Wheel builds for PyPI |
| CI/CD | GitHub Actions | Tests, releases, PyPI publishing |

---

## 3. Build & Test Commands

### Rust (native binary + library)
```bash
# Debug build
cargo build

# Release build (used for benchmarking)
cargo build --release

# Run all Rust tests (unit + integration; default features only)
cargo test

# Run only unit tests (skip debug/regression tests that need ../PHANOTATE/ files)
cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_

# Run only integration tests
cargo test --test cli_tests
cargo test --test detect_table_tests

# Run with Clippy (zero warnings policy — CI enforces this)
cargo clippy -- -D warnings

# Check formatting (CI enforces this)
cargo fmt -- --check

# Run Criterion benchmarks
cargo bench
```

### Python bindings
```bash
# Build and install locally for development
pip install maturin
maturin develop

# Run Python tests
pytest tests/test_python_bindings.py -v
```

### ML feature build
```bash
# Build with ONNX Runtime support for --ml-model
cargo build --release --features ml

# Build with both Python and ML
cargo build --release --features "python ml"
```

### Feature flags
| Feature | Dependencies enabled | What it does |
|---------|----------------------|--------------|
| `default` | none | CLI + library only |
| `python` | `pyo3` | Python bindings via PyO3 |
| `ml` | `ort`, `ndarray` | ONNX-based hybrid ORF scorer |

---

## 4. Code Organisation (`src/`)

| File | Lines | Responsibility |
|------|-------|----------------|
| `main.rs` | 733 | CLI entry point. Parses args, orchestrates the full pipeline, handles `--detect-table`, `--detect-table-batch`, `--ml-model`, `--export-features`, `--progress`, `-a`, `-d`, `-c`, `-m`, and multi-threading via `rayon`. |
| `lib.rs` | 23 | Library root. Re-exports all internal modules as `pub` so integration tests can access crate-private items. Conditionally includes `ml_scorer` (feature `ml`) and `lib_python` (feature `python`). |
| `lib_python.rs` | 1,091 | PyO3 Python bindings. Exposes `phanotate()`, `find_orfs()`, `detect_table()`, `score_rbs()`, `translate()`, plus utility functions and Python classes (`Orf`, `Gene`, `TableScore`). Mirrors the CLI pipeline logic. |
| `orf.rs` | 1,286 | ORF data structure (`Orf` struct) and ORF finding logic. `find_orfs_with_rc()` enumerates all ORFs in 6 reading frames. Includes RBS scoring, P(stop) computation, heuristic scoring (`score()`), and hybrid ML scoring (`score_hybrid()` gated by `cfg(feature = "ml")`). |
| `graph.rs` | 525 | Graph construction from ORFs. Defines `Node`, `Edge`, and `Graph` structs. `Graph::from_orfs()` builds a directed graph where nodes are start/stop codons and edges represent ORFs, gaps, and overlaps. Uses `num_bigint::BigInt` for weights. |
| `bellman_ford.rs` | 258 | Shortest path solver. Tries topological-order relaxation (O(V+E)) first; falls back to Bellman-Ford if backward edges (cycles from strand switches) are detected. |
| `codon_table.rs` | 452 | NCBI translation tables 1, 4, 6, 11, 15, 25. Each supported table has its own `translate_tableN()` function. Also provides `start_codons()`, `stop_codons()`, `is_supported_table()`, `table_name()`. |
| `detect_table.rs` | 1,259 | Automatic genetic code detection. Uses mean ORF length ratio and reassigned-codon signal heuristics. `CANDIDATE_TABLES = [1, 4, 11, 15, 25]` (table 6 is supported manually but excluded from auto-detection). Includes batch detection and confidence scoring. |
| `genome.rs` | 191 | Genome I/O: FASTA and GenBank parsing. `Genome` struct holds `id`, `seq` (lowercase), and pre-computed `rc_seq`. |
| `gcfp.rs` | 221 | GC Frame Plot computation. Sliding 120-bp window (40 codons) over 3 frames to compute per-position GC content arrays. |
| `output.rs` | 543 | Output formatting: GenBank (`gbk`), GFF3 (`gff`), and SCO (`sco`) formats. Also writes protein and nucleotide FASTA side outputs. |
| `weights.rs` | 133 | Edge weight formulas: `score_overlap()` and `score_gap()` for graph edges. Includes unit tests. |
| `ml_features.rs` | 233 | Feature extraction for ML. `OrfFeatures` is a fixed 14-feature vector (`NUM_FEATURES = 14`). Includes TSV export (`write_features_tsv()`) for training data generation. |
| `ml_scorer.rs` | 202 | ONNX-based ML scorer (gated by `cfg(feature = "ml")`). Wraps `ort::Session` in a `Mutex` for thread safety. Loads ONNX models, runs inference, clamps adjustment to [0.5, 2.0]. |

---

## 5. Testing Strategy

The project uses a **three-tier testing** approach. The exact number of compiled tests depends on the enabled features (`python` / `ml`).

### Tier 1 — Unit tests (inline in source files)
* Located in `#[cfg(test)]` modules inside `src/` files.
* With default features, `cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_` runs **98 tests**.
* With the `ml` feature enabled, the same skip filter runs **100 tests**.
* With both `python` and `ml` features enabled, the total unit-test count rises to roughly **136** (some tests are defined in `lib_python.rs`).
* Running `cargo test --lib` without skips currently yields **102 passed; 22 failed** — the 22 failures are `debug_`, `regression_`, `lambda_`, and `table4_` tests that require files from an external `../PHANOTATE/` directory.
* Examples: `weights.rs` has 12 tests for overlap/gap scoring; `genome.rs` has tests for `rev_comp()` and `normalize_seq()`; `bellman_ford.rs` has tests for linear and cyclic graphs; `output.rs` has 28 tests for formatters.
* Run with: `cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_`

### Tier 2 — Integration tests (spawn the real binary)
* `tests/cli_tests.rs` (365 lines): Tests CLI flag parsing, output format validation, golden-file comparisons against the Python reference, stdin input, closed ends, N-masking, and combo flags. A full run without external files yields **7 passed; 14 failed**; the failures need `../PHANOTATE/tests/` files.
* `tests/detect_table_tests.rs` (640 lines): Tests genetic-code detection on real genomes (lambda, phiX174, SpV4) and synthetic sequences. Covers confidence levels, batch mode, pipe mode, and deterministic LCG-based synthetic benchmarks. A full run without external files yields **17 passed; 6 failed**; the failures need `../PHANOTATE/tests/` files.
* Golden test data lives in `tests/golden/` (e.g. `phiX174.tabular`, `NC_000866.1.tabular`, `phiX174.fasta_out`).
* Some integration tests require files from an external `../PHANOTATE/` directory; CI skips these via `--skip` flags.
* Run with: `cargo test --test cli_tests` and `cargo test --test detect_table_tests` (expect failures if the external repo is absent).

### Tier 3 — Python binding tests
* `tests/test_python_bindings.py` (451 lines): `pytest` suite covering all PyO3 API functions (`phanotate`, `find_orfs`, `detect_table`, `translate`, `score_rbs`, data classes), edge cases (short sequences, ambiguous bases, mixed case, multiline FASTA), and protein translation without internal stops.
* Run with: `pytest tests/test_python_bindings.py -v` (requires `maturin develop` first).

### CI behaviour
* **Rust CI** (`.github/workflows/rust-tests.yml`): runs on Ubuntu, macOS, Windows. Skips external-file-dependent tests and a subset of integration tests. Enforces `cargo fmt --check` and `cargo clippy -- -D warnings`.
* **Python CI** (`.github/workflows/python-tests.yml`): runs on Ubuntu/macOS/Windows × Python 3.9/3.12 (plus macOS 3.13). Builds with `maturin develop`, runs `pytest`, then runs Rust unit tests.

---

## 6. Code Style Guidelines

* **Formatting**: Enforced via `cargo fmt --check` in CI. Run `cargo fmt` before committing.
* **Linting**: `cargo clippy -- -D warnings` (warnings are treated as errors in CI).
* **Error handling**: Use `anyhow` with `.context()` for ergonomic error propagation. Avoid `unwrap()` in production code; it is acceptable only in tests.
* **Conditional compilation**: Use `#[cfg(feature = "ml")]` and `#[cfg(feature = "python")]` to gate optional functionality. The `ml` and `python` features are **not** enabled by default.
* **Documentation**: Module-level doc comments (`//!`) with detailed algorithm explanations. Public APIs should have doc comments.
* **Unsafe code**: Minimal. One instance exists in `detect_table_tests.rs` using `unsafe { seq.as_bytes_mut() }` for synthetic test data mutation. No unsafe in production code.
* **Naming**: Follow standard Rust conventions (`snake_case` for functions/variables, `PascalCase` for types/structs, `SCREAMING_SNAKE_CASE` for constants).
* **BigInt weights**: Graph edge weights use `num_bigint::BigInt` to prevent overflow on very long ORFs. Conversion helper `f64_to_bigint_weight()` is in `graph.rs`.
* **Parallelism**: Use `rayon` parallel iterators (`par_iter()`) for multi-genome processing. The `indicatif` progress bar integrates with Rayon when `--progress` is used.
* **Thread-safe ONNX**: The `MlScorer` wraps `ort::Session` in a `Mutex` because `Session::run()` requires `&mut self`, but inference is called from multiple threads.

---

## 7. Release & Deployment Process

### Version bumping
Use `./bump-version.sh <new-version>` (e.g. `./bump-version.sh 0.1.4`). It updates:
* `VERSION`
* `Cargo.toml` (package version only, first five lines)
* `packaging/conda/meta.yaml` (version and SHA256 placeholder)
* `packaging/homebrew/phanotate-rs.rb` (URL and SHA256 placeholder)
* `Cargo.lock` (via `cargo update -w`)

It does **not** update `pyproject.toml`; that version is kept in sync manually.

After bumping, commit, tag (`git tag v0.1.4`), and push. GitHub Actions triggers automatically on tag pushes.

### Automated release workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `release.yml` | Tag push `v*` | Builds native binaries for 6 platforms (Linux x86_64/musl/aarch64, macOS x86_64/aarch64, Windows x86_64), packages as `.tar.gz`/`.zip`, creates GitHub Release. |
| `pypi.yml` | Tag push `v*` or manual dispatch | Builds Python wheels (Linux x86_64, macOS universal2, Windows x86_64) and source distribution via `maturin-action`. Publishes to PyPI via **trusted publishing (OIDC)**. |
| `bioconda.yml.disabled` | Disabled | Placeholder workflow that would compute the release tarball SHA256 and open a PR to `bioconda-recipes`. Disabled until Bioconda automation is finalised. |

### Distribution channels
* **GitHub Releases**: Pre-built binaries for Linux (x86_64, aarch64, musl), macOS (Intel, Apple Silicon), Windows (x86_64).
* **PyPI**: `pip install phanotate-rs` (wheels for Linux, macOS, Windows; `pyproject.toml` requires Python ≥3.8, README says 3.9+).
* **Bioconda**: `conda install -c bioconda phanotate-rs`.
* **Homebrew**: `brew tap yasas1994/bioinformatics && brew install phanotate-rs`.
* **Cargo**: `cargo install phanotate-rs`.
* **One-liner installer**: `curl -fsSL https://raw.githubusercontent.com/Yasas1994/PHANOTATE-rs/main/install.sh | bash`.

---

## 8. ML / Hybrid Scoring (Optional Feature)

The `ml` feature enables an optional ONNX-based ORF score adjustment:

1. **Feature extraction** (`src/ml_features.rs`): 14 features per ORF (log length, RBS score, log hold, P(stop), start-codon one-hot, GC content, frame indicators, log motif score).
2. **Training** (`scripts/train_model.py`, `notebooks/01_train_hybrid_scorer.ipynb`): Train XGBoost / Random Forest / MLP regressors on labeled ORF data.
3. **Label generation** (`scripts/generate_training_labels.py`): Uses UniProt reviewed phage proteins + DIAMOND blastp to label ORFs as genes/non-genes.
4. **Export to ONNX** (`scripts/export_tree_onnx.py`, `scripts/export_regressor_onnx.py`, `scripts/export_pytorch_onnx.py`): Export trained models to ONNX format.
5. **Runtime inference** (`src/ml_scorer.rs`): Loads ONNX model, predicts adjustment factor per ORF, clamps to [0.5, 2.0], multiplies heuristic weight.

Trained example models and training plots are stored under `notebooks/models/`.

Usage:
```bash
# Build with ML support
cargo build --release --features ml

# Run with a trained model
./target/release/phanotate-rs -i genome.fasta --ml-model model.onnx

# Export features for training data generation
./target/release/phanotate-rs -i genome.fasta --export-features features.tsv
```

---

## 9. Security Considerations

* **Trusted publishing (OIDC)**: The PyPI workflow uses OIDC instead of long-lived API tokens. Setup is documented in `pypi.yml` comments.
* **Unsafe code**: Minimal and confined to test code (`detect_table_tests.rs`). No unsafe in production code.
* **Input validation**: The CLI validates translation table numbers, output formats, and file paths. The Python bindings validate all inputs and raise `PyValueError` for invalid arguments.
* **Binary distribution**: The `install.sh` script downloads release artifacts from GitHub and verifies the binary runs (`--version`) after installation. It prefers `musl` builds on Linux for maximum portability.
* **License compliance**: `cargo-bundle-licenses` is integrated in the Conda recipe for Rust dependency license bundling.

---

## 10. Key Files for Agents

| File | Why it matters |
|------|----------------|
| `Cargo.toml` | Rust package config, dependencies, features, bench harness. |
| `pyproject.toml` | Python packaging config for maturin. |
| `VERSION` | Single source of truth for version number. |
| `bump-version.sh` | Automated version bumper for packaging files (does not touch `pyproject.toml`). |
| `install.sh` | One-line installer script for end users. |
| `src/lib.rs` | Library root — all modules are re-exported here. |
| `src/main.rs` | CLI entry point — the full pipeline orchestration. |
| `src/lib_python.rs` | PyO3 bindings — mirrors CLI logic for Python API. |
| `.github/workflows/rust-tests.yml` | Rust CI — tests, fmt, clippy. |
| `.github/workflows/python-tests.yml` | Python CI — pytest + maturin. |
| `.github/workflows/pypi.yml` | PyPI wheel building and publishing. |
| `.github/workflows/release.yml` | Native binary release building. |
| `.github/workflows/bioconda.yml.disabled` | Disabled Bioconda recipe automation placeholder. |
| `tests/golden/` | Golden test files for CLI regression testing. |
| `notebooks/README.md` | ML training pipeline documentation. |
| `memory/progress_2026-06-06.md` | Project progress tracker (feature status, benchmarks). |

---

## 11. Quick Reference: Common Tasks

| Task | Command |
|------|---------|
| Build release binary | `cargo build --release` |
| Run all tests | `cargo test` |
| Run unit tests (skip external-file tests) | `cargo test --lib -- --skip debug_ --skip regression_ --skip lambda_ --skip table4_` |
| Run with Clippy | `cargo clippy -- -D warnings` |
| Format code | `cargo fmt` |
| Build Python bindings | `maturin develop` |
| Run Python tests | `pytest tests/test_python_bindings.py -v` |
| Run benchmarks | `cargo bench` |
| Build with ML | `cargo build --release --features ml` |
| Annotate a genome | `./target/release/phanotate-rs -i genome.fasta -f sco` |
| Detect genetic code | `./target/release/phanotate-rs -i genome.fasta --detect-table --yes` |
| Export ML features | `./target/release/phanotate-rs -i genome.fasta --export-features features.tsv` |
| Bump version | `./bump-version.sh 0.1.4` |

---

*Last updated: 2026-06-23*
