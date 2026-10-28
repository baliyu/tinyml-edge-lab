#!/usr/bin/env python3
"""Quick look at one recorded microphone session: for each class, the median and the 10th to 90th percentile of every
summary number, over the valid windows only. A look at the data, not a model and not a result.
Built with AI assistance (Claude).

    py tools\\mic_peek.py --session s1
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

FEATS = ("rms", "peak", "zcr", "hf", "clip")


def read_file(path: Path):
    label, rows, header = "", [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# label="):
            label = line.split("=", 1)[1].strip()
        elif line.startswith("#") or not line.strip():
            continue
        elif header is None:
            header = line.split(",")
        else:
            rows.append(dict(zip(header, line.split(","))))
    return label, [r for r in rows if r.get("valid") == "1"]


def pct(vals, lo, hi):
    if len(vals) < 2:
        return vals[0], vals[0]
    q = statistics.quantiles(vals, n=10)
    return q[0], q[8]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--dir", default="data/own/mic")
    a = ap.parse_args(argv)
    files = sorted(Path(a.dir).glob(f"{a.session}_*.csv"))
    if not files:
        print(f"No files for session {a.session} in {a.dir}", file=sys.stderr)
        return 1
    print(f"Session {a.session}: median (10th to 90th percentile) over valid windows\n")
    print(f"{'class':8}{'n':>5}  " + "  ".join(f"{f:>18}" for f in FEATS))
    for f in files:
        label, rows = read_file(f)
        if not rows:
            print(f"{label:8}{0:>5}  (no valid windows)")
            continue
        cells = []
        for k in FEATS:
            v = [int(r[k]) for r in rows]
            lo, hi = pct(v, 10, 90)
            cells.append(f"{statistics.median(v):>6.0f} ({lo:>4.0f}-{hi:>5.0f})")
        print(f"{label:8}{len(rows):>5}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
