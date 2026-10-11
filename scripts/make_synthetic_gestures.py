#!/usr/bin/env python3
"""Write SYNTHETIC desk-gesture recordings (NOT real data) so the pipeline can be tried before any real session exists.
Usage: python scripts/make_synthetic_gestures.py [--out data/synthetic/edge_gestures] [--sessions 3] [--seconds 45] [--seed 0]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab.edge_synth import write_synthetic_gestures  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/synthetic/edge_gestures")
p.add_argument("--sessions", type=int, default=3)
p.add_argument("--seconds", type=float, default=45.0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
paths = write_synthetic_gestures(a.out, sessions=a.sessions, seconds=a.seconds, seed=a.seed)
print(f"wrote {len(paths)} SYNTHETIC recordings to {a.out}")
print(f"try:  python scripts/run_gestures.py --data {a.out} --out results/synthetic")
