#!/usr/bin/env python3
"""Draw the report figures from results/har_baseline.json. Run train_har_baseline.py first."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab.plots import build_report_figures  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/raw")
p.add_argument("--results", default="results")
a = p.parse_args()
names = build_report_figures(a.root, results_dir=a.results)
print("wrote", len(names), "figures to", Path(a.results) / "figures", ":", ", ".join(names))
