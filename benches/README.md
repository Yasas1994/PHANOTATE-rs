# Optimization Benchmark Results

## Baseline (pre-optimization)

| Genome | Time (ms) | Notes |
|--------|-----------|-------|
| phiX174 | 15.037 | 5,386 bp |
| NC_001416.1 | 127.28 | 48,502 bp |
| NC_000866.1 | 373.19 | 168,903 bp |

### Top 3 Hotspots from Flamegraph
1. `score_rbs` closure + memcmp — ~48.73%
2. `score_rbs` function + from_utf8 — ~25.96%
3. `__memcmp_avx2_movbe` (string compare) — ~23.56%

Combined, `score_rbs` accounted for ~75% of runtime.

---

## Final Results

| Genome | Baseline | Final | Speedup |
|--------|----------|-------|---------|
| phiX174 | 15.0 ms | 6.0 ms | **2.5×** |
| NC_001416.1 | 127 ms | 46 ms | **2.8×** |
| NC_000866.1 | 373 ms | 101 ms | **3.7×** |

---

## Optimization Log

### score_rbs byte-based optimization (#1 hotspot, not in original task list)
- **Status**: ✅ Done — eliminated `from_utf8` and string `contains()`, using byte patterns directly
- **Measured gain**:
  - phiX174: 15.0ms → 7.1ms (-53%)
  - NC_001416.1: 127ms → 53ms (-58%)
  - NC_000866.1: 373ms → 116ms (-69%)

### OPT-1: Topological-order relaxation (replaces Bellman-Ford)
- **Status**: ✅ Done
- **Result**: No measurable gain for these test genomes (graphs are small, DAG path always taken)
- **Note**: Fallback to Bellman-Ford preserved for genomes with cycles

### OPT-2: Single-pass O(n) ORF enumeration
- **Status**: Partially already implemented — current code accumulates starts and flushes on stops
- **Note**: No significant change needed; algorithm is already O(n) per frame

### OPT-3: Log-space arithmetic for weights
- **Status**: ✅ Done
- **Measured gain**:
  - phiX174: 7.1ms → 6.2ms (-13%)
  - NC_001416.1: 53ms → 49ms (-8%)
  - NC_000866.1: 116ms → 107ms (-7%)
- **Note**: Also fixes potential underflow bug on long ORFs

### OPT-4: Pre-computed reverse complement
- **Status**: ✅ Done — added `rc_seq` to Genome, used in background RBS and ORF finding
- **Measured gain** (cumulative):
  - phiX174: 6.2ms → 5.9ms (-5%)
  - NC_001416.1: 49ms → 47ms (-4%)
  - NC_000866.1: 107ms → 101ms (-4%)

### OPT-5: CSR flat adjacency list
- **Status**: ❌ Reverted — showed slight regression for small graphs
- **Note**: CSR structures remain in `graph.rs` for future use with larger graphs

### OPT-6: 64-entry codon log-probability lookup table
- **Status**: Not implemented — current `compute_pstop` uses ORF-specific frequencies, not genome-wide
- **Note**: Would require semantic change; skipped to preserve correctness

### OPT-7: Prefix-sum GCFP
- **Status**: ❌ Reverted — ring-buffer implementation showed no gain; `remove(0)` in `get()` was O(n)
- **Note**: VecDeque version is already reasonably efficient

### OPT-8: Parallelize weight calculation with Rayon
- **Status**: Not implemented — weight calculation is already fast after other optimizations
- **Note**: Main parallelism is already at genome level (OPT-9)

### OPT-9: Work-stealing batch processing
- **Status**: ✅ Already implemented — `main.rs` uses `par_iter()` over genomes

### OPT-10: Profile-guided optimization
- **Status**: Not implemented
- **Note**: Could yield 5–15% additional gain; left as future work

---

## Summary

The dominant optimization was **byte-based `score_rbs`** (~60% speedup), which eliminated UTF-8 string conversion and `memcmp` calls. **Log-space arithmetic** and **pre-computed RC** contributed additional ~15% and ~5% respectively.

Total speedup: **2.4× to 3.7×** depending on genome size.

All golden-file tests pass. `cargo test` is clean (62 unit + 16 integration tests).
