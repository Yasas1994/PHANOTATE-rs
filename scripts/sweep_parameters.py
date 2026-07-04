#!/usr/bin/env python3
"""Grid search over PHANOTATE-rs ONNX runtime parameters.

Runs ``scripts/benchmark_model.py`` for each combination of
``--model-scale`` and ``--model-threshold`` (and auto-threshold variants)
and reports the configuration with the highest average gene-level F1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_benchmark(
    *,
    binary: Path,
    genome_dir: Path,
    model: Path | None,
    output: Path,
    max_genomes: int,
    scale: float | None = None,
    threshold: float | None = None,
    auto_threshold: str | None = None,
    timeout: int = 180,
) -> dict | None:
    """Run benchmark_model.py for a single parameter set and return the summary.

    Returns ``None`` if the run times out or the subprocess is killed (e.g. OOM).
    """
    cmd = [
        sys.executable,
        "scripts/benchmark_model.py",
        "--binary", str(binary),
        "--genome-dir", str(genome_dir),
        "--output", str(output),
        "--max-genomes", str(max_genomes),
    ]
    if model is not None:
        cmd.extend(["--models", str(model)])
    if scale is not None:
        cmd.extend(["--model-scale", str(scale)])
    if threshold is not None:
        cmd.extend(["--model-threshold", str(threshold)])
    if auto_threshold is not None:
        cmd.extend(["--auto-threshold", auto_threshold])

    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        label = f"scale={scale} threshold={threshold} auto={auto_threshold}"
        print(f"  {label} failed: {e}", file=sys.stderr)
        return None
    with output.open() as fh:
        data = json.load(fh)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep PHANOTATE-rs ONNX runtime parameters.")
    parser.add_argument("--genome-dir", type=Path, default=Path("tests/golden/annotgenomes_gb"))
    parser.add_argument("--binary", type=Path, default=Path("./target/release/phanotate-rs"))
    parser.add_argument("--model", type=Path, default=Path("models/model_lr.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/sweep_results.json"))
    parser.add_argument("--max-genomes", type=int, default=50)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument(
        "--auto-thresholds",
        type=str,
        nargs="+",
        default=["length", "percentile"],
        help="Auto-threshold modes to test with default scale",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.genome_dir.exists():
        print(f"Error: genome directory not found: {args.genome_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    tmp_output = args.output.parent / "_sweep_tmp.json"

    # Baseline heuristic (no model) for reference.
    print("\n=== Baseline heuristic ===")
    data = run_benchmark(
        binary=args.binary,
        genome_dir=args.genome_dir,
        model=None,
        output=tmp_output,
        max_genomes=args.max_genomes,
        scale=None,
        threshold=None,
        auto_threshold=None,
    )
    if data is None:
        print("Error: baseline heuristic run failed", file=sys.stderr)
        return 1
    heuristic = data["summary"]["heuristic"]
    print(f"heuristic: precision={heuristic['precision']:.4f} recall={heuristic['recall']:.4f} f1={heuristic['f1']:.4f}")

    # Fixed-threshold grid.
    total = len(args.scales) * len(args.thresholds) + len(args.auto_thresholds)
    print(f"\n=== Sweeping {total} configurations ===")
    for scale, threshold in itertools.product(args.scales, args.thresholds):
        print(f"\nRunning scale={scale} threshold={threshold} ...")
        data = run_benchmark(
            binary=args.binary,
            genome_dir=args.genome_dir,
            model=args.model,
            output=tmp_output,
            max_genomes=args.max_genomes,
            scale=scale,
            threshold=threshold,
            auto_threshold=None,
        )
        if data is None:
            continue
        model_metrics = data["summary"]["models"][args.model.name]
        row = {
            "scale": scale,
            "threshold": threshold,
            "auto_threshold": None,
            **model_metrics,
        }
        rows.append(row)
        print(f"  precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}")

    # Auto-threshold variants with default scale.
    for auto in args.auto_thresholds:
        print(f"\nRunning auto-threshold={auto} (default scale) ...")
        data = run_benchmark(
            binary=args.binary,
            genome_dir=args.genome_dir,
            model=args.model,
            output=tmp_output,
            max_genomes=args.max_genomes,
            scale=None,
            threshold=None,
            auto_threshold=auto,
        )
        if data is None:
            continue
        model_metrics = data["summary"]["models"][args.model.name]
        row = {
            "scale": None,
            "threshold": None,
            "auto_threshold": auto,
            **model_metrics,
        }
        rows.append(row)
        print(f"  precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}")

    tmp_output.unlink(missing_ok=True)

    if not rows:
        print("Error: all model configurations failed", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    best_idx = df["f1"].idxmax()
    best = {
        k: (None if isinstance(v, float) and pd.isna(v) else v)
        for k, v in df.loc[best_idx].to_dict().items()
    }

    results = {
        "heuristic": heuristic,
        "best": best,
        "all_configs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== Sweep summary ===")
    print(df.to_string(index=False))
    print(f"\nBest configuration: {best}")
    print(f"Heuristic F1: {heuristic['f1']:.4f}; Best model F1: {best['f1']:.4f}")
    if best["f1"] > heuristic["f1"]:
        print("Model configuration beats heuristic.")
    else:
        print("Model configuration does NOT beat heuristic.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
