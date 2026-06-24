# Dicodon Filter Design

> **Goal:** Add an optional, annotation-driven dicodon filter to PHANOTATE-rs so low-coding-potential ORFs are removed before the shortest-path step, improving precision on annotated genomes such as `NC_001365.gb`.

## Context

* A learned start-site model (`--start-model`) already improves start-coordinate choice on `NC_001365.gb`.
* On that genome, start-model-alone precision is ~63% because extra false-positive ORFs are still kept.
* The existing `--dicodon` scoring **multiplier** does **not** help: combining `--dicodon` with the start model dropped precision to ~57% on `NC_001365.gb`.
* A dicodon model trained from the *known* coding sequences and used as a **hard filter** is conceptually different from the multiplier: it asks "is this ORF coding?" instead of "how should this ORF be weighted relative to others?"

## Recommended approach: trained dicodon filter (`--dicodon-filter`)

When `--dicodon-filter` is enabled:

1. Build a `DicodonModel` from trusted seed ORFs.
2. Score every ORF with the model.
3. Drop ORFs whose dicodon score is below a configurable threshold before graph construction.

Seeds come from annotations when available (GenBank input), otherwise from the existing heuristic seed selection used by `DicodonModel::train`.

## Alternatives considered

| Approach | Why it is not recommended |
|---|---|
| Use the existing `--dicodon` multiplier as a filter | Already observed to reduce precision on `NC_001365.gb`; it re-weights rather than removes false positives. |
| Use dicodon score only to choose between overlapping start candidates | Does not address the core false-positive problem; the start model already handles start choice. |

## Architecture

```text
Load genome
  │
  ▼
Find all ORFs
  │
  ▼
Score RBS / motif / start-model
  │
  ▼
[--dicodon-filter] ?
  ├─ Train DicodonModel from annotated CDS (GenBank) or heuristic seeds (FASTA)
  ├─ Score each ORF
  └─ Retain only ORFs with dicodon_score >= threshold
  │
  ▼
Build graph → shortest path → output
```

## Components

### `src/dicodon.rs` (existing, minor additions)

* `DicodonModel::train` already trains from heuristic seeds.
* Add `DicodonModel::from_annotated_orfs(&[Orf]) -> Self` that skips the seed-selection logic and counts dicodons directly from the supplied annotated ORFs.
* `DicodonModel::score_orf` already returns a positive multiplier; values well below 1.0 indicate non-coding sequence.

### `src/genome.rs`

* Extend `Genome` to optionally carry a list of annotated CDS coordinates: `pub cds: Vec<(usize, usize, i8)>`.
* Populate this list in `read_genbank` by parsing `CDS` feature locations and their strand.
* FASTA input leaves `cds` empty.

### `src/main.rs`

* Add CLI flags:
  * `--dicodon-filter` — enable the filter.
  * `--dicodon-filter-threshold <FLOAT>` — minimum dicodon score to keep an ORF (default `0.5`).
* In `process_genome`, after start-model scoring:
  * If `--dicodon-filter` is set and `genome.cds` is non-empty, build `DicodonModel::from_annotated_orfs(&annotated_orfs)`.
  * Else if the flag is set but no annotations exist, fall back to `DicodonModel::train(&orfs, dna, rc_dna)` using heuristic seeds and emit a one-line warning.
  * Apply `model.score_orf` to each ORF and `retain` those with score >= threshold.

### `src/lib_python.rs`

* Add `dicodon_filter: bool` and `dicodon_filter_threshold: Option<f64>` parameters to `phanotate()` and `find_orfs()`.
* Mirror the CLI logic when either parameter is true/set.

### `tests/cli_tests.rs`

* Add a smoke test that `--dicodon-filter` runs on `tests/golden/NC_001365.gb`.
* Add a regression test that precision on `NC_001365.gb` is at least as good as start-model alone.

## Data flow

1. `load_genomes` returns `Vec<Genome>`; GenBank records include annotated CDS ranges.
2. `process_genome` receives the full `Genome`, including annotations.
3. After ORF finding and scoring, the optional filter:
   * collects `Orf` objects matching annotated CDS coordinates (GenBank) or long/signal-rich ORFs (FASTA fallback);
   * trains the dicodon model;
   * scores and filters the full ORF list;
4. The surviving ORFs feed into `Graph::from_orfs` as before.

## Error handling

* If `--dicodon-filter` is enabled but the model cannot be trained (no seeds), emit a warning and continue without filtering.
* `--dicodon-filter-threshold` must be a positive finite float; `clap` validates this.
* Invalid/unknown bases in a dicodon cause the k-mer to be skipped, as in the existing implementation.

## Testing

| Level | Test | Expected |
|---|---|---|
| Unit | `DicodonModel::from_annotated_orfs` on synthetic ORFs | Produces finite, discriminative scores |
| Unit | `score_orf` returns value below threshold for random sequence | Filter would drop it |
| Integration | `--dicodon-filter -i tests/golden/NC_001365.gb -f sco` | Runs successfully |
| Integration | Precision/recall comparison vs start-model-only on `NC_001365.gb` | Precision improves, recall stays at 12/12 |
| Python | `phanotate(..., dicodon_filter=True)` | Runs and returns fewer or equal ORFs |

## Open questions / tuning

* Default threshold `0.5` is a starting point; it should be evaluated on the cross-validation genome set.
* Whether to clamp the dicodon score before filtering (already clamped to `[0.1, 10.0]`).
* Whether to combine `start_score` and `dicodon_score` into a single product filter (future work).
