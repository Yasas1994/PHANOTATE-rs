# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] — 2026-11-18

### Added

- Full NCBI translation table support: tables 1–6, 9–16, and 21–31 (previously 1, 4, 6, 11, 15, 25)
- `--rbs-mode auto|sd|non-sd|prodigal` — control how upstream start-codon motifs are scored:
  auto-detection of Shine–Dalgarno vs non-SD motifs, legacy SD scoring, non-SD motif
  discovery, or Prodigal-style SD bins
- Prodigal-style hexamer (cscore) coding-potential scoring
- Learned ORF scoring behind `--model`: trains on 34 per-ORF features exported via
  `--export-features`, runs ONNX inference to replace the heuristic ORF edge weight
- Per-genome calibration on the `--model` path: `--auto-threshold` plus learned
  gap/overlap penalty ratios
- `--circular` flag: circular genomes are annotated on the doubled sequence and genes
  spanning the origin are reported as a single joined feature
- Overlap rescue flags to recover overlapping gene calls dropped by the default scoring
- Start-codon signature and context prior to improve automatic genetic-code detection

### Fixed

- Genomes whose largest non-coding gap fell in a ~5 bp window at 500 bp (covered 496–500)
  annotated zero genes: the inter-ORF edge cutoff and the long-gap bridge used
  inconsistent distance metrics, leaving a dead zone with no path across the gap
  (part of issue #6)
- Ambiguous IUPAC bases (Y, K, R, N, …) were silently converted to defined bases
  (mostly adenine) before translation, so proteins showed a defined amino acid where
  the ambiguity admits no decision. Ambiguity now survives normalization and reverse
  complementation, and undecidable codons translate to `X` on both strands
  (part of issue #6)
- RNA input: `U`/`u` accepted as thymine
- Table-aware heuristic scoring: `P(stop)` and start-codon weights now follow the
  active translation table (improves table-4 genomes)
- Windows builds: upgraded `ort` to 2.0.0-rc.15 — the pinned ONNX Runtime 1.24.2
  artifacts used by rc.12 were no longer served for all platforms
- CI is self-contained: unit and integration tests no longer depend on a local
  checkout of the original PHANOTATE repository or untracked fixture files

### Changed

- Release binaries: Linux x86_64/aarch64 (glibc), macOS Apple Silicon, Windows x86_64.
  musl and Intel-mac builds are dropped because ONNX Runtime publishes no binaries
  for those targets; the macOS PyPI wheel is arm64-only (`install.sh` already falls
  back to the glibc Linux artifact)
- Rust MSRV raised from 1.70 to 1.83 (required by pyo3 0.29)
- Dependencies: pyo3 0.22 → 0.29.3 (addresses three security advisories, including a
  high-severity out-of-bounds read); unmaintained `atty` replaced with
  `std::io::IsTerminal`

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
