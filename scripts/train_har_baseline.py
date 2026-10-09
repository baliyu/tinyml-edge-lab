#!/usr/bin/env python3
"""Train and evaluate the decision-tree baseline. Usage: python scripts/train_har_baseline.py [--root data/raw] [--depth 6]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab.har_baseline import run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/raw")
p.add_argument("--depth", type=int, default=6)
p.add_argument("--out", default="results")
a = p.parse_args()
r = run(a.root, chosen_depth=a.depth, out_dir=a.out)
print(f"test accuracy {r['test_accuracy']:.3f}, macro-F1 {r['test_macro_f1']:.3f}, "
      f"{r['nodes']} nodes, {r['leaves']} leaves")
print("per-class recall:", {k: round(v, 3) for k, v in r["per_class_recall"].items()})
print("wrote", Path(a.out) / "har_baseline.json", Path(a.out) / "har_tree.c", Path(a.out) / "depth_sweep.md")
