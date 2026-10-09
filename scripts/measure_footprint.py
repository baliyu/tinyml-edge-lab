#!/usr/bin/env python3
"""Measure the real flash and static RAM of the exported UCI models on Cortex-M0+ and Cortex-M4F with the ARM compiler.
Needs results/compare.json (run scripts/compare_models.py first) and the UCI data in data/raw.
Usage: python scripts/measure_footprint.py [--root data/raw] [--results results] [--readme README.md] [--no-readme]
If the compiler is not found: set ARM_GCC=/path/to/arm-zephyr-eabi-gcc (or ZEPHYR_SDK_INSTALL_DIR=/path/to/sdk)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import footprint, har_compare  # noqa: E402
from tinyml_lab.footprint_report import build_report, footprint_fig, render_block, update_readme, write_outputs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/raw")
p.add_argument("--results", default="results")
p.add_argument("--readme", default="README.md")
p.add_argument("--no-readme", action="store_true")
a = p.parse_args()

tool = footprint.find_toolchain()
if tool is None:
    print("No ARM compiler found. Tell me where it is, for example:")
    print("  ARM_GCC=$HOME/zephyr-sdk-1.0.1/gnu/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc python scripts/measure_footprint.py")
    print("  (find it with:  find ~ -name arm-zephyr-eabi-gcc 2>/dev/null | head -3)")
    print("or install one:  sudo apt install gcc-arm-none-eabi")
    sys.exit(2)
print("compiler:", tool["version"])
cmp_path = Path(a.results) / "compare.json"
if not cmp_path.exists():
    print("results/compare.json is missing: run scripts/compare_models.py first.")
    sys.exit(1)
cmp_rep = json.loads(cmp_path.read_text(encoding="utf-8"))
bands = cmp_rep["protocol"]["n_bands"]
print("computing features (this takes a minute or two) ...")
sets_tr, _, ytr, _ = har_compare.build_features(a.root, "train", bands)
sets_te, _, yte, _ = har_compare.build_features(a.root, "test", bands)
models = footprint.refit_models(cmp_rep, sets_tr, ytr)
match = footprint.refit_matches_saved(cmp_rep, models, sets_te, yte)
print("refitted models reproduce the saved test accuracy:", match)
results = []
for m in cmp_rep["models"]:
    fs, est = models[m["id"]]
    print(f"measuring {m['id']} ({m['nodes']} nodes) ...")
    results.append({"id": m["id"], "label": m["label"], "short": m["short"], "features": fs, "n_features": m["n_features"],
                    "nodes": m["nodes"], "leaves": m["leaves"], "variants": footprint.measure_model(est, sets_te[fs], tool)})
fp = build_report(tool, cmp_rep, results, match, len(yte))
write_outputs(a.results, fp)
footprint_fig(fp, Path(a.results) / "figures" / "footprint.png")
print()
print((Path(a.results) / "footprint.md").read_text(encoding="utf-8"))
if not a.no_readme and Path(a.readme).exists():
    update_readme(a.readme, render_block(fp))
    print("README updated from the results")
