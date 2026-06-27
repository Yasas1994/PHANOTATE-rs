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
    if mode not in {"none", "length", "percentile"}:
        raise ValueError(
            f"Invalid mode: {mode!r}; expected 'none', 'length', or 'percentile'"
        )

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
        if r.returncode != 0:
            print(
                f"Warning: phanotate-rs failed for mode={mode} param={param} "
                f"genome={genome} (return code {r.returncode})",
                file=sys.stderr,
            )
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, **agg}
        ref = _parse_genbank_cds(str(genome))
        pred = set()
        for line in r.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
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
    if not MODEL.is_file():
        print(f"Model file not found: {MODEL}", file=sys.stderr)
        sys.exit(1)
    if not ANNOT_DIR.is_dir() or not any(ANNOT_DIR.glob("*.gb")):
        print(
            f"Annotation directory is empty or missing .gb files: {ANNOT_DIR}",
            file=sys.stderr,
        )
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
