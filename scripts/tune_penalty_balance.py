#!/usr/bin/env python3
"""Grid-search the global gap/overlap target ratios on annotated genomes."""
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


def evaluate(gap_target: float, overlap_target: float, genomes: list[Path]) -> dict:
    env = os.environ.copy()
    env["PHANOTATE_MODEL_GAP_TARGET_RATIO"] = str(gap_target)
    env["PHANOTATE_MODEL_OVERLAP_TARGET_RATIO"] = str(overlap_target)
    agg = {"tp": 0, "fp": 0, "fn": 0}
    for genome in genomes:
        r = subprocess.run(
            [
                str(BINARY),
                "-i",
                str(genome),
                "--detect-table",
                "--yes",
                "-f",
                "sco",
                "--model",
                str(MODEL),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        print(f"Binary not found: {BINARY}; run cargo build --release --features ml", file=sys.stderr)
        sys.exit(1)
    genomes = paths(50)
    print(f"Tuning on {len(genomes)} genomes")
    best = None
    for gap in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        for overlap in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            res = evaluate(gap, overlap, genomes)
            print(f"gap={gap:.2f} overlap={overlap:.2f} -> F1={res['f1']:.4f}")
            if best is None or res["f1"] > best["f1"]:
                best = {"gap": gap, "overlap": overlap, **res}
    print("\nBest:")
    print(f"  gap_target={best['gap']:.2f}")
    print(f"  overlap_target={best['overlap']:.2f}")
    print(f"  F1={best['f1']:.4f}")


if __name__ == "__main__":
    main()
