#!/usr/bin/env python3
"""Held-out evaluation of the microphone classes (quiet / speech / knock) from the Edge's summary lines.
Built with AI assistance (Claude).

The model is trained on some sittings and tested ONCE on a sitting it never saw. Nothing is tuned on the test sitting.

    python3 tools/mic_eval.py --train s1,s2,s3 --test b
    python3 tools/mic_eval.py --train s1,s2,s3,b,c,d --test e --compare

--compare also runs a calibrated version: each session's first CAL_WINDOWS valid windows of its quiet recording give that
session's quiet level, and the features rms, peak, hf and zcr are divided by it. Those calibration windows are left out of
every score (calibrated or not), so both versions are tested on the same windows. The pass mark for the calibrated version
was fixed before it was run: macro-F1 >= 0.85 for both models AND quiet recall >= 0.80.

Sessions recorded back to back (s1, s2, s3) are ONE sitting: train on them together, never use one to test another.
Only windows marked valid=1 are used. The interval comes from resampling 2-second blocks of the test windows. It reflects
only that one test sitting, not how results would differ between sittings, so treat it as a lower bound on the uncertainty.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

FEATURES = ("rms", "peak", "zcr", "hf", "clip")
RATIO = ("rms", "peak", "zcr", "hf")       # divided by the session's quiet level when calibrating
CLASSES = ("quiet", "speech", "knock")
BLOCK = 8            # windows per resampling block (2 s)
CAL_WINDOWS = 40     # first 10 s of a session's quiet recording
PASS_F1, PASS_QUIET_RECALL = 0.85, 0.80


def read_files(directory: Path, sessions: list[str]):
    out = []
    for s in sessions:
        paths = sorted(directory.glob(f"{s}_*.csv"))
        if not paths:
            raise SystemExit(f"No files found for session {s!r} in {directory}")
        for path in paths:
            label, session, header, rows = "", s, None, []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# label="):
                    label = line.split("=", 1)[1].strip()
                elif line.startswith("# session="):
                    session = line.split("=", 1)[1].strip() or s
                elif line.startswith("#") or not line.strip():
                    continue
                elif header is None:
                    header = line.split(",")
                else:
                    r = dict(zip(header, line.split(",")))
                    if r.get("valid") == "1":
                        rows.append(r)
            rows.sort(key=lambda r: int(r["w"]))
            out.append({"session": s, "label": label, "rows": rows, "path": path})
    return out


def build(directory: Path, sessions: list[str], calibrate: bool):
    files = read_files(directory, sessions)
    cal = {}
    for s in sessions:
        qs = [f for f in files if f["session"] == s and f["label"] == "quiet"]
        if not qs or len(qs[0]["rows"]) <= CAL_WINDOWS:
            raise SystemExit(f"Session {s!r} needs a quiet recording with more than {CAL_WINDOWS} valid windows.")
        first = qs[0]["rows"][:CAL_WINDOWS]
        cal[s] = {k: float(np.median([float(r[k]) for r in first])) for k in RATIO}
    X, y, blk = [], [], []
    for fi, f in enumerate(files):
        rows = f["rows"][CAL_WINDOWS:] if f["label"] == "quiet" else f["rows"]   # calibration windows never scored
        for r in rows:
            v = []
            for k in FEATURES:
                x = float(r[k])
                if calibrate and k in RATIO:
                    x = x / cal[f["session"]][k]
                v.append(x)
            X.append(v)
            y.append(f["label"])
            blk.append((fi, int(r["w"]) // BLOCK))
    return np.array(X), np.array(y), blk


def block_bootstrap_f1(y_true, y_pred, blk, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(blk))
    idx_by = {k: [] for k in keys}
    for i, k in enumerate(blk):
        idx_by[k].append(i)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by[keys[j]] for j in pick])
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=list(CLASSES), average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(name, model, Xtr, ytr, Xte, yte, blk):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    f1 = f1_score(yte, pred, labels=list(CLASSES), average="macro", zero_division=0)
    acc = float((pred == yte).mean())
    lo, hi = block_bootstrap_f1(yte, pred, blk)
    cm = confusion_matrix(yte, pred, labels=list(CLASSES))
    qrec = cm[0][0] / max(1, cm[0].sum())
    print(f"\n{name}: accuracy {acc:.3f}, macro-F1 {f1:.3f} (95% block interval {lo:.3f} to {hi:.3f}; chance is about 0.333)")
    print("  rows = true class, columns = predicted:   " + "  ".join(f"{c:>7}" for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        rec = row[CLASSES.index(c)] / max(1, row.sum())
        print(f"  {c:>7} {' '.join(f'{v:>9}' for v in row)}    recall {rec:.3f}")
    return f1, float(qrec)


def models():
    return (("Decision tree (depth 4)", lambda: DecisionTreeClassifier(max_depth=4, random_state=0)),
            ("Random forest (100 trees)", lambda: RandomForestClassifier(n_estimators=100, random_state=0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="data/own/mic")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--compare", action="store_true", help="also run the calibrated version and print the pass/fail")
    a = ap.parse_args(argv)
    tr, te = a.train.split(","), a.test.split(",")
    if set(tr) & set(te):
        raise SystemExit("A session cannot be in both --train and --test.")
    d = Path(a.dir)
    modes = (("UNCALIBRATED", False), ("CALIBRATED", True)) if a.compare else (("UNCALIBRATED", False),)
    results = {}
    for mname, cal in modes:
        Xtr, ytr, _ = build(d, tr, cal)
        Xte, yte, blk = build(d, te, cal)
        print(f"\n===== {mname} =====")
        print(f"Train {tr}: {len(ytr)} windows. Test {te}: {len(yte)} windows (used once, never tuned on). "
              f"Calibration windows ({CAL_WINDOWS} per session) excluded from every score.")
        print("Features:", ", ".join(FEATURES) + ("  (rms, peak, zcr, hf relative to the session's own quiet)" if cal else ""))
        for name, mk in models():
            results[(mname, name)] = report(name, mk(), Xtr, ytr, Xte, yte, blk)
    if a.compare:
        print("\n===== PASS MARK (fixed before the run): calibrated macro-F1 >= "
              f"{PASS_F1} for both models AND quiet recall >= {PASS_QUIET_RECALL} =====")
        ok = True
        for name, _ in models():
            f1, qr = results[("CALIBRATED", name)]
            good = f1 >= PASS_F1 and qr >= PASS_QUIET_RECALL
            ok &= good
            print(f"  {name}: macro-F1 {f1:.3f}, quiet recall {qr:.3f} -> {'pass' if good else 'FAIL'}")
        print("  OVERALL: " + ("PASS" if ok else "FAIL"))
    print("\nRemember: one held-out sitting; the quiet calibration and quiet test windows come from the same recording "
          "(different windows), which may flatter quiet recall a little.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
