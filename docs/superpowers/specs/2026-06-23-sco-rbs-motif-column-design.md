# SCO RBS Motif Column — Design Spec

**Project:** PHANOTATE-rs  
**Date:** 2026-06-23  
**Feature:** Add a detected RBS/non-SD motif column to the SCO table output.

---

## 1. Goal

Extend the SCO output format so each predicted gene includes the upstream motif that contributed to its start-codon score. The column must report:

- SD mode: the canonical Shine–Dalgarno motif that matched (e.g., `AGGAGG`).
- Non-SD mode: the discovered motif (e.g., `AAAAAA`).
- No match: `not detected`.

Other output formats (GFF3, GenBank) remain unchanged.

---

## 2. Motivation

Users currently see a numeric RBS score but cannot tell which motif drove it. Exposing the actual motif improves interpretability and makes it easier to validate that the scorer behaved as expected.

---

## 3. Design

### 3.1 Data model

Add an optional motif string to `Orf`:

```rust
pub struct Orf {
    // ... existing fields ...
    pub rbs_motif: Option<String>,
}
```

`None` means no motif was detected. The SCO formatter renders `None` as `not detected`.

### 3.2 SD motif detection

Add a new helper `detect_rbs_motif(seq: &[u8]) -> Option<String>` in `src/orf.rs` that mirrors the pattern logic of `score_rbs` but returns the matched motif string instead of the score. Patterns are checked in the same priority order so the reported motif corresponds to the highest-scoring match.

Example patterns (with canonical spacing):

| Score | Motif |
|-------|-------|
| 27 | `AGGAGG` |
| 26 | `AGGAGG` (far) |
| 24 | `GGAGG` |
| 22 | `GAGGA` |
| 19 | `GGAAGA`, `GGATGA`, `GGACGA`, `GGCGGA`, `GGGGGA`, `GGTGGA` |
| 16 | `GGAG`, `GAGG` |
| ... | ... |

The helper returns the first (highest-priority) motif whose byte pattern is found in the reversed upstream window.

### 3.3 Non-SD motif detection

For non-SD mode, after training the model, call `find_best_motif` for each ORF. Decode `hit.ndx` with `hit.len` using `kmer_decode`. If `hit.len == 0`, set `rbs_motif = None`.

### 3.4 Population timing

- In `find_orfs_with_rc` (SD mode): set `rbs_motif` when `rbs_score` is computed.
- In `main.rs` / `lib_python.rs` (non-SD mode): after training and scoring, set `rbs_motif` for each ORF.
- In hybrid/ML path: the motif is already set by the time `score_hybrid` runs.

### 3.5 SCO output format

Change `write_sco` to emit five tab-separated columns:

```text
# uses_sd: 1
start\tstop\tstrand\tweight\tmotif\n
```

Example lines:

```text
156	368	+	-3.90E1	AGGAGG
96	368	+	-2.32E0	AAAAAA
200	500	+	-1.50E0	not detected
```

### 3.6 Other formats

GFF3 and GenBank keep their existing columns/fields. Adding a motif attribute to GFF3 was considered but rejected to keep this change focused on SCO only.

---

## 4. Files Changed

| File | Change |
|------|--------|
| `src/orf.rs` | Add `rbs_motif` field; add `detect_rbs_motif`; populate in constructors. |
| `src/nonsd_motif.rs` | Ensure `kmer_decode` is public; add `format_motif_hit` helper. |
| `src/main.rs` | Populate `rbs_motif` in non-SD mode. |
| `src/lib_python.rs` | Populate `rbs_motif` in non-SD mode. |
| `src/output.rs` | Add motif column to `write_sco`. |
| `tests/cli_tests.rs` | Update expectations / add non-SD motif test. |
| `src/output.rs` tests | Update SCO golden strings. |

---

## 5. Testing

- **Unit test:** `detect_rbs_motif` returns `Some("AGGAGG")` for a window containing `aaggaggtg...`.
- **Unit test:** `detect_rbs_motif` returns `None` for a window with no SD-like sequence.
- **Unit test:** `write_sco` includes the motif column.
- **CLI test:** `--non-sd -f sco` output contains a discovered motif or `not detected` in the fifth column.
- **Regression:** default SD mode SCO output still has five columns and the same coordinates.

---

## 6. Backwards Compatibility

The SCO format gains one column. Any parser expecting exactly four columns will break. This is an intentional format change; SCO is described as a "simple coordinate" format and is the most permissible place to add metadata.
