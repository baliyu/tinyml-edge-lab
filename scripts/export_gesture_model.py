#!/usr/bin/env python3
"""Train the desk-gesture model on EVERY recording and write it as C for the board.

Usage: python scripts/export_gesture_model.py [--data data/own/edge_gestures] [--model forest|tree] [--features raw|motion]
                                              [--out firmware/sparkfun_edge/gesture_demo]
Writes gesture_model.c, gesture_model.h and gesture_model.json (feature order, the gesture each number means, what it was trained
on). The model has seen every session, so it has no held-out score of its own: the honest numbers are in results/gestures.json,
and the on-device accuracy comes from a session recorded after this model was trained.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import gesture_demo as gd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/own/edge_gestures")
p.add_argument("--model", choices=gd.MODEL_KINDS, default="forest")
p.add_argument("--features", choices=("raw", "motion"), default="raw")
p.add_argument("--out", default=str(gd.DEMO_DIR))
p.add_argument("--allow-synthetic", action="store_true", help="dry run on synthetic recordings; only with --out pointing somewhere else")
a = p.parse_args()

try:
    clf, X, y, labels, ds = gd.train_deployable(a.data, a.model, a.features)
except (FileNotFoundError, ValueError) as e:
    print("Cannot run:", e)
    sys.exit(1)
if ds.synthetic and (not a.allow_synthetic or Path(a.out).resolve() == gd.DEMO_DIR.resolve()):
    print("Refusing: these recordings are SYNTHETIC. A model trained on them says nothing about real gestures and must not go on the board.\n"
          "(For a dry run: --allow-synthetic --out <some other folder>.)")
    sys.exit(1)
meta = gd.write_model(clf, labels, a.features, a.model, a.out, ds)
size = f"{meta['trees']} trees, {meta['nodes']} nodes" if a.model == "forest" else f"{meta['nodes']} nodes"
print(f"{'SYNTHETIC dry run: ' if ds.synthetic else ''}{a.model} on {a.features} features, {size}, trained on {meta['trained_on']['windows']} windows "
      f"from sessions {', '.join(meta['trained_on']['sessions'])}")
print(f"classes: {', '.join(labels)}")
print(f"wrote {a.out}/gesture_model.c, gesture_model.h, gesture_model.json (no held-out score: it has seen every session)")
