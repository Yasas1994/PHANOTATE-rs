#!/usr/bin/env python3
"""Compare PHANOTATE-rs predictions to reference GenBank CDS annotations.

This is the evaluation step used in the model-training notebook: after running
`phanotate-rs --model model.json -f sco`, compare the predicted gene
coordinates to the `CDS` coordinates in the annotated GenBank file.

Usage
-----
    # Exact coordinate match
    python scripts/compare_predictions.py -p preds.sco -r ref.gb

    # Allow start/stod differences up to 3 bp
    python scripts/compare_predictions.py -p preds.sco -r ref.gb --tolerance 3

Metrics
-------
* Precision = TP / (TP + FP)
* Recall    = TP / (TP + FN)
* F1        = 2 * P * R / (P + R)

A predicted gene is a true positive (TP) when both its start and stop
coordinates are within `--tolerance` bp of a reference CDS. Coordinates on
the reverse strand are compared as written in the SCO file (start > stop).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


def _parse_genbank_cds(path: str) -> Set[Tuple[int, int]]:
    """Return the CDS coordinate set from a GenBank file."""
    text = Path(path).read_text()

    feat_match = re.search(
        r"FEATURES\s+Location/Qualifiers\s+(.*?)(?=^ORIGIN)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not feat_match:
        return set()

    block = feat_match.group(1)
    cds: Set[Tuple[int, int]] = set()
    current: dict | None = None

    def parse_loc(loc: str) -> Tuple[int, int] | None:
        loc = loc.strip()
        strand = "+"
        if loc.startswith("complement("):
            strand = "-"
            loc = loc[len("complement(") :].rstrip(")")
        if loc.startswith("join("):
            loc = loc[len("join(") :].rstrip(")")
        coords: List[Tuple[int, int]] = []
        for part in loc.split(","):
            part = part.strip()
            m = re.match(r"(\d+)\.\.(\d+)", part)
            if m:
                coords.append((int(m.group(1)), int(m.group(2))))
        if not coords:
            return None
        low = min(s for s, _ in coords)
        high = max(e for _, e in coords)
        return (low, high) if strand == "+" else (high, low)

    for raw_line in block.splitlines():
        if not raw_line.strip():
            continue

        qual_match = re.match(r"^\s{21}/([^=]+)(?:=(.*))?$", raw_line)
        if qual_match and current is not None:
            continue

        cont_match = re.match(r"^\s{21}(\S.*)$", raw_line)
        if cont_match and current is not None:
            current["location"] += cont_match.group(1)
            continue

        feat_key_match = re.match(r"^\s{5}(\S+)\s+(\S.*)$", raw_line)
        if feat_key_match:
            if current is not None and current["key"] == "CDS":
                parsed = parse_loc(current["location"])
                if parsed is not None:
                    cds.add(parsed)
            current = {"key": feat_key_match.group(1), "location": feat_key_match.group(2)}

    if current is not None and current["key"] == "CDS":
        parsed = parse_loc(current["location"])
        if parsed is not None:
            cds.add(parsed)

    return cds


def _parse_sco(path: str) -> Set[Tuple[int, int]]:
    """Return the gene coordinate set from a PHANOTATE SCO file."""
    coords: Set[Tuple[int, int]] = set()
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0])
            stop = int(parts[1])
        except ValueError:
            continue
        coords.add((start, stop))
    return coords


def _match(pred: Tuple[int, int], true_set: Set[Tuple[int, int]], tolerance: int) -> bool:
    """Check whether a predicted gene matches any reference CDS within tolerance."""
    ps, pe = pred
    for ts, te in true_set:
        if abs(ps - ts) <= tolerance and abs(pe - te) <= tolerance:
            return True
    return False


def compare(pred_set: Set[Tuple[int, int]], true_set: Set[Tuple[int, int]], tolerance: int) -> dict:
    tp = sum(1 for p in pred_set if _match(p, true_set, tolerance))
    fp = len(pred_set) - tp
    fn = sum(1 for t in true_set if not _match(t, pred_set, tolerance))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": len(pred_set),
        "reference": len(true_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PHANOTATE-rs predictions to GenBank CDS annotations."
    )
    parser.add_argument(
        "-p", "--predictions", required=True, help="PHANOTATE SCO output file."
    )
    parser.add_argument(
        "-r", "--reference", required=True, help="Reference GenBank file with CDS features."
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=0,
        help="Allowed start/stop coordinate difference (default: 0).",
    )
    args = parser.parse_args()

    pred = _parse_sco(args.predictions)
    true = _parse_genbank_cds(args.reference)

    if not true:
        print("Error: no CDS features found in reference.", file=sys.stderr)
        return 1

    m = compare(pred, true, args.tolerance)

    print(f"Predicted genes: {m['predicted']}")
    print(f"Reference genes: {m['reference']}")
    print(f"TP: {m['tp']}  FP: {m['fp']}  FN: {m['fn']}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1:        {m['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
