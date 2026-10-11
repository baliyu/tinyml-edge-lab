#!/usr/bin/env python3
"""Evaluate the desk-gesture recordings honestly, draw the figures, and (for real data) write the results into the README.

Usage: python scripts/run_gestures.py [--data data/own/edge_gestures] [--out results] [--readme README.md]
                                      [--no-readme] [--export-c] [--allow-problems]
For a dry run on synthetic data:
    python scripts/make_synthetic_gestures.py
    python scripts/run_gestures.py --data data/synthetic/edge_gestures --out results/synthetic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import edge_gestures as eg  # noqa: E402
from tinyml_lab.edge_gesture_eval import export_tree, run_gestures  # noqa: E402
from tinyml_lab.edge_gesture_report import build_gesture_figures, render_block, update_readme  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/own/edge_gestures")
p.add_argument("--out", default="results")
p.add_argument("--readme", default="README.md")
p.add_argument("--no-readme", action="store_true")
p.add_argument("--export-c", action="store_true", help="also write results/gesture_tree.c (a tree trained on every session)")
p.add_argument("--allow-problems", action="store_true", help="also use recordings the capture tool flagged with PROBLEMS")
a = p.parse_args()

try:
    report = run_gestures(a.data, out_dir=a.out, allow_problems=a.allow_problems)
    ds = eg.load_dataset(a.data, allow_problems=a.allow_problems)
except (FileNotFoundError, ValueError) as e:
    print("Cannot run:", e)
    sys.exit(1)
print()
print((Path(a.out) / "gestures.md").read_text(encoding="utf-8"))
print("figures:", ", ".join(build_gesture_figures(ds, a.out)))
if a.export_c:
    clf, _, _, labels = export_tree(a.data, a.out, report=report)
    print(f"wrote {a.out}/gesture_tree.c ({clf.tree_.node_count} nodes) and gesture_model.json; classes: {', '.join(labels)}")
if report["data"]["synthetic"]:
    print("These are SYNTHETIC recordings: the numbers prove the code works, not that gestures can be recognised. "
          "The README was left alone.")
elif a.no_readme or not Path(a.readme).exists():
    pass
elif Path(a.out).as_posix().rstrip("/") != "results":
    print("README not updated: its figure links point at results/figures, so only --out results is written into it.")
else:
    update_readme(a.readme, render_block(report))
    print("README updated from the results")
