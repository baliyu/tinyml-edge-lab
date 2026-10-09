#!/usr/bin/env python3
"""Grouped model selection, richer features and a random-forest comparison on the real UCI HAR data.
Usage: python scripts/compare_models.py [--root data/raw] [--out results] [--readme README.md] [--no-readme]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab.har_compare import run_compare  # noqa: E402
from tinyml_lab.plots import build_compare_figures  # noqa: E402
from tinyml_lab.readme_block import render_block, update_readme  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/raw")
p.add_argument("--out", default="results")
p.add_argument("--readme", default="README.md")
p.add_argument("--no-readme", action="store_true")
a = p.parse_args()

report = run_compare(a.root, out_dir=a.out)
print()
print((Path(a.out) / "compare.md").read_text(encoding="utf-8"))
print("figures:", ", ".join(build_compare_figures(a.out)))
if not a.no_readme and Path(a.readme).exists():
    update_readme(a.readme, render_block(report))
    print("README updated from the results")
