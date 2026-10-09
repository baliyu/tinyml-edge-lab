#!/usr/bin/env python3
"""Run the CWRU evaluation, draw the figures and write the results into the README.
Usage: python scripts/run_cwru.py [--data data/raw/cwru] [--out results] [--readme README.md] [--no-readme]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import cwru  # noqa: E402
from tinyml_lab.cwru_eval import run_cwru  # noqa: E402
from tinyml_lab.cwru_plots import build_cwru_figures  # noqa: E402
from tinyml_lab.cwru_readme import render_block, update_readme  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/raw/cwru")
p.add_argument("--out", default="results")
p.add_argument("--readme", default="README.md")
p.add_argument("--no-readme", action="store_true")
a = p.parse_args()

problems = cwru.check_dir(a.data)
if problems:
    print("The data folder is not ready:", "; ".join(f"{n}: {r}" for n, r in problems[:8]), "...")
    print("Run python scripts/download_cwru.py first.")
    sys.exit(1)
report = run_cwru(a.data, out_dir=a.out)
print()
print((Path(a.out) / "cwru.md").read_text(encoding="utf-8"))
print("figures:", ", ".join(build_cwru_figures(a.data, a.out)))
if not a.no_readme and Path(a.readme).exists():
    update_readme(a.readme, render_block(report))
    print("README updated from the results")
