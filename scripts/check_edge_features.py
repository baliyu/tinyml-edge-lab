#!/usr/bin/env python3
"""Check the C feature code against the Python features on a folder of recordings (real or synthetic).

Usage: python scripts/check_edge_features.py [--data data/own/edge_gestures] [--out results]
It builds firmware/sparkfun_edge/gesture_features/edge_features.c with the host gcc as double (must equal Python to machine
precision) and as float (what the board runs), compares the features of every window, and reports how often a trained
model answers differently on the float C features. Writes <out>/edge_features_check.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import edge_gestures as eg  # noqa: E402
from tinyml_lab.edge_features_c import CFeatures, compare_features, prediction_agreement  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/own/edge_gestures")
p.add_argument("--out", default="results")
a = p.parse_args()

try:
    ds = eg.load_dataset(a.data)
    c64, c32 = CFeatures("double"), CFeatures("float")
except (FileNotFoundError, ValueError, RuntimeError) as e:
    print("Cannot run:", e)
    sys.exit(1)

banner = "SYNTHETIC DATA, NOT REAL RECORDINGS" if ds.synthetic else "real recordings"
print(f"{len(ds.y)} windows from {len(ds.files)} files ({banner}); sessions: {', '.join(sorted(set(ds.session.tolist())))}")
report = {"data": {"windows": int(len(ds.y)), "synthetic": bool(ds.synthetic)}, "feature_sets": {}}
for fset in eg.FEATURE_SETS:
    py = eg.feature_matrix(ds.windows, eg.FS, fset)
    d64 = compare_features(py, c64.compute(ds.windows, fset), fset)
    xc32 = c32.compute(ds.windows, fset)
    d32 = compare_features(py, xc32, fset)
    agree = prediction_agreement(py, xc32, ds.y, ds.session)
    report["feature_sets"][fset] = {"double_vs_python": d64, "float_vs_python": d32, "prediction_agreement": agree}
    print(f"\n[{fset} features]")
    print(f"  C as double vs Python: worst difference {d64['max_error_vs_feature_size']:.1e} of a feature's size ({d64['worst_feature']}); "
          f"different dominant bin in {d64['windows_with_a_different_dominant_bin']} windows")
    print(f"  C as float  vs Python: worst difference {d32['max_error_vs_feature_size']:.1e} of a feature's size ({d32['worst_feature']}); "
          f"different dominant bin in {d32['windows_with_a_different_dominant_bin']} windows")
    for model, r in agree.items():
        s = r["seen"]
        line = f"  {model}: answers differently on the float C features for {s['different']} of {s['windows']} training windows"
        if "unseen sessions" in r:
            u = r["unseen sessions"]
            line += f", and {u['different']} of {u['windows']} windows from a session it never saw"
        print(line)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "edge_features_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {out / 'edge_features_check.json'}")
if ds.synthetic:
    print("These are SYNTHETIC recordings: they show the code agrees with Python, not how it behaves on real motion.")
