#!/usr/bin/env python3
"""Run recordings through the C gesture chain, sample by sample as the board would receive them, and compare with the Python chain.

Usage: python scripts/check_gesture_pipeline.py [--data data/own/edge_gestures] [--model forest|tree] [--features raw|motion]
                                                [--out results]
Trains the model the same way scripts/export_gesture_model.py does, builds firmware/sparkfun_edge/gesture_demo with the host gcc
as double (must equal Python to machine precision) and as float (what the board runs), and compares every window: the gesture
number and the features. If an ARM compiler is installed it also reports the flash, RAM and stack of the board build.
Writes <out>/gesture_pipeline_check.json. Nothing here runs on the board.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import gesture_demo as gd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/own/edge_gestures")
p.add_argument("--model", choices=gd.MODEL_KINDS, default="forest")
p.add_argument("--features", choices=("raw", "motion"), default="raw")
p.add_argument("--out", default="results")
a = p.parse_args()

try:
    clf, X, y, labels, ds = gd.train_deployable(a.data, a.model, a.features)
    work = Path(tempfile.mkdtemp(prefix="gesture_check_"))
    meta = gd.write_model(clf, labels, a.features, a.model, work / "model", ds)
    runs = {real: gd.check_recordings(a.data, clf, labels, a.features, work / "model", real, workdir=work / real) for real in ("double", "float")}
    board = gd.arm_footprint(work / "model", work / "arm")
except (FileNotFoundError, ValueError, RuntimeError) as e:
    print("Cannot run:", e)
    sys.exit(1)

banner = "SYNTHETIC DATA, NOT REAL RECORDINGS" if ds.synthetic else "real recordings"
f = runs["float"]
print(f"{f['windows']} windows from {f['files']} files ({banner}); sessions: {', '.join(meta['trained_on']['sessions'])}; "
      f"tick rate(s): {', '.join(str(t) for t in sorted({r['tick_hz'] for r in f['per_file']}))} Hz")
print(f"model: {a.model} on {a.features} features, trained on every session ({meta['nodes']} nodes)")
for real, r in runs.items():
    note = "must equal Python to machine precision" if real == "double" else "what the board runs"
    print(f"  C chain as {real:6s} ({note}): gesture differs from Python in {r['class_mismatches']} of {r['windows']} windows; "
          f"worst feature difference {r['worst_feature_error']:.1e} of a feature's size ({r['worst_feature']}); "
          f"different dominant bin in {r['windows_with_a_different_dominant_bin']} windows")
if board:
    fl = board["flash_bytes"]
    print(f"board build ({board['compiler'].split(' (')[0]}, Cortex-M4F, -Os): flash {fl['total']:,} B (chain {fl['pipeline'] + fl['features']:,} B, "
          f"model {fl['model']:,} B), state RAM {board['state_ram_bytes']:,} B, deepest stack about {board['deepest_stack_estimate_bytes']:,} B; "
          f"needs: {', '.join(board['undefined_symbols'])}")
else:
    print("board build: not measured (no arm-none-eabi-gcc)")

committed = gd.DEMO_DIR / "gesture_model.c"
same = (committed.is_file() and committed.read_text(encoding="utf-8") == (work / "model" / "gesture_model.c").read_text(encoding="utf-8"))
if committed.is_file():
    print(f"committed model {'is identical to' if same else 'DIFFERS from'} a fresh export from these recordings")

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
report = {"data": {"files": f["files"], "windows": f["windows"], "synthetic": bool(ds.synthetic), "sessions": meta["trained_on"]["sessions"],
                   "tick_hz": sorted({r["tick_hz"] for r in f["per_file"]})},
          "model": {k: meta[k] for k in ("model", "feature_set", "nodes", "classes") if k in meta} | ({"trees": meta["trees"]} if "trees" in meta else {}),
          "builds": {real: {k: v for k, v in r.items() if k != "per_file"} for real, r in runs.items()},
          "per_file_float": f["per_file"], "board_build": board,
          "committed_model": {"present": committed.is_file(), "identical_to_fresh_export": bool(same)},
          "note": "Agreement between the C and Python chains on recordings the model was trained on. It says nothing about accuracy on new sessions."}
(out / "gesture_pipeline_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote {out / 'gesture_pipeline_check.json'}")
