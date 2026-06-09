# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Python bindings** via PyO3 — `import phanotate_rs` from Python
  - `phanotate()` — full gene-calling pipeline with all CLI options exposed as kwargs
  - `find_orfs()` — low-level ORF finder returning structured `Orf` objects
  - `detect_table()` — automatic translation-table detection returning `TableScore` objects
  - `translate()` — DNA-to-protein translation using NCBI tables
  - `score_rbs()` — Shine-Dalgarno scoring
  - `supported_tables()`, `stop_codons()`, `start_codons()`, `table_name()` — codon-table utilities
  - Python classes: `Orf`, `Gene`, `TableScore` with full attribute access
- `pyproject.toml` — maturin configuration for Python wheel builds
- `src/lib_python.rs` — PyO3 module implementation
- `tests/test_python_bindings.py` — 51 Python tests covering all bindings
- `.github/workflows/pypi.yml` — automated PyPI publishing via trusted publishing (OIDC)
- `.github/workflows/python-tests.yml` — CI for Python bindings across platforms
- `bump-version.sh` now bumps `pyproject.toml` version and documents PyPI setup steps

### Changed

- `.github/workflows/release.yml` — added release summary mentioning PyPI and bioconda workflows
- `src/lib.rs` — includes `lib_python` module for PyO3 bindings
- `Cargo.toml` — added `pyo3` dependency and `cdylib` crate-type for Python extension builds

## [0.1.3] — 2025-06-08

### Added

- GitHub Actions workflow for PyPI publishing (`pypi.yml`)
- GitHub Actions workflow for Python bindings CI testing (`python-tests.yml`)
- Python test suite (`tests/test_python_bindings.py`)

### Fixed

- macOS wheel deployment target set to `14.0` for modern system compatibility

## [0.1.2] — 2026-06-02

### Added

- `bump-version.sh` — automated version bumper for all packaging files
- `VERSION` file — single source of truth for version number
- `.github/workflows/bioconda.yml` — automated bioconda recipe update PRs
- `packaging/conda/` — Conda recipe template
- `packaging/homebrew/` — Homebrew formula template
- `cargo-bundle-licenses` integration for compliance

### Changed

- README badges and documentation improvements
- Benchmark results split into single-thread vs multi-thread comparisons
- Added Prodigal-gv to large-scale benchmark comparison

## [0.1.1] — 2026-06-02

### Added

- Large-scale benchmark results (100 phage genomes)
- 100-genome benchmark dataset accession list
- Intersection and unique predictions analysis in benchmark table

### Fixed

- Bioconda recipe: removed standalone cargo dependency, added build scripts

## [0.1.0] — 2026-06-02

### Added

- Initial Rust implementation of PHANOTATE gene caller
- Automatic genetic-code detection (`--detect-table`, `--detect-table-batch`)
- Support for 6 NCBI translation tables: 1, 4, 6, 11, 15, 25
- Multi-threaded processing via Rayon
- Three output formats: GenBank (`gbk`), GFF3 (`gff`), SCO (`sco`)
- Protein (`-a`) and nucleotide (`-d`) FASTA side outputs
- Closed ends (`-c`) and N-mask (`-m`) options
- Progress bar (`--progress`) for batch processing
- Overlapping gene detection (`--find-overlaps`)
- GitHub Actions release workflow with cross-platform binaries
- Integration tests and golden file comparisons
- Benchmark suite with Criterion

### Performance

- ~10× faster than original Python PHANOTATE
- Single-thread and multi-thread benchmarks against Prodigal and Prodigal-gv

---

## Release Checklist

When cutting a new release:

1. Run `./bump-version.sh X.Y.Z`
2. Review `git diff`
3. Commit: `git add -A && git commit -m "chore: bump version to X.Y.Z"`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push origin main && git push origin vX.Y.Z`
6. GitHub Actions will automatically:
   - Build binaries for all platforms (`release.yml`)
   - Build and publish Python wheels to PyPI (`pypi.yml`)
   - Update bioconda recipe (`bioconda.yml`)
