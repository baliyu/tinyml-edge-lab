"""Honest evaluation of desk-gesture classifiers on the Edge's own recordings.

Two protocols, the same models with settings fixed in advance (not tuned), two feature sets:
  random_windows          a random split of windows: LEAKY, shown only to expose the problem (windows cut from one
                          recording are neighbours, and the sitting's orientation, speed and noise leak across the split)
  leave_one_session_out   train on all other sittings, test on the one held out: a model meets a sitting it has never seen

Feature sets: 'raw' (includes the board's orientation through the window means) and 'motion' (orientation removed).
All recordings come from one person and one board, so even the honest protocol does not measure how a different
person or a different board would fare.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import edge_gestures as eg

TREE_DEPTH = 6
FOREST_TREES = 100


def session_splits(sessions):
    """(held-out session, train indices, test indices) for each session."""
    sessions = np.asarray(sessions)
    for held in sorted(set(sessions.tolist())):
        yield held, np.flatnonzero(sessions != held), np.flatnonzero(sessions == held)


def assert_session_disjoint(sessions, train_idx, test_idx):
    """Fail loudly if one sitting is on both sides of a split."""
    sessions = np.asarray(sessions)
    both = set(sessions[train_idx].tolist()) & set(sessions[test_idx].tolist())
    if both:
        raise AssertionError(f"session(s) {sorted(both)} appear in both the training and the test side of a split")


def _models(seed):
    return {"tree": lambda: DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=seed),
            "forest": lambda: RandomForestClassifier(n_estimators=FOREST_TREES, random_state=seed, n_jobs=-1)}


def _scores(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "labels": list(labels),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def _run_folds(X, y, folds, labels, seed, groups=None):
    """Fit every model on every fold. Returns {model: scores} with pooled predictions and per-fold accuracy."""
    out = {}
    for name, make in _models(seed).items():
        pred, accs = {}, []
        for held, tr, te in folds:
            if groups is not None:
                assert_session_disjoint(groups, tr, te)
            p = make().fit(X[tr], y[tr]).predict(X[te])
            pred.update(zip(te.tolist(), p.tolist()))
            accs.append(float(accuracy_score(y[te], p)))
        idx = np.array(sorted(pred))
        res = _scores(y[idx], np.array([pred[i] for i in idx]), labels)
        res["fold_accuracy"] = accs
        out[name] = res
    return out


def _majority_accuracy(y, folds):
    """Accuracy of always answering the most common training label, pooled over the folds."""
    right = total = 0
    for _, tr, te in folds:
        vals, counts = np.unique(y[tr], return_counts=True)
        right += int(np.sum(y[te] == vals[np.argmax(counts)]))
        total += len(te)
    return right / total


def ordered_labels(y):
    present = set(y.tolist())
    return [g for g in eg.GESTURES if g in present] + sorted(present - set(eg.GESTURES))


def run_gestures(data_dir, out_dir="results", fs=eg.FS, win=eg.WINDOW, step=eg.STEP, trim_s=eg.TRIM_S, n_bands=eg.N_BANDS,
                 seed=0, allow_problems=False, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = eg.load_dataset(data_dir, fs, win, step, trim_s, allow_problems)
    problems = eg.design_problems(ds)
    sessions = sorted(set(ds.session.tolist()))
    if len(set(ds.y.tolist())) < 2 or len(sessions) < 2:
        raise ValueError("cannot evaluate: " + "; ".join(problems))
    labels = ordered_labels(ds.y)
    log(f"{len(ds.files)} recordings -> {len(ds.y)} windows; sessions {', '.join(sessions)}; gestures {', '.join(labels)}")
    for p in problems:
        log("  design note: " + p)
    for name, why in ds.skipped:
        log(f"  skipped {name}: {why}")

    y = ds.y
    protocols = {}
    folds_los = list(session_splits(ds.session))
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    folds_rand = [("random 30%", tr, te)]
    protocols["random_windows"] = {
        "title": "Random split of windows (leaky)",
        "folds": [{"held_out": "random 30%", "n_train": int(len(tr)), "n_test": int(len(te))}], "results": {}}
    protocols["leave_one_session_out"] = {
        "title": "Leave one session out",
        "folds": [{"held_out": h, "n_train": int(len(a)), "n_test": int(len(b)),
                   "train_sessions": sorted(set(ds.session[a].tolist()))} for h, a, b in folds_los], "results": {}}
    for fset in eg.FEATURE_SETS:
        log(f"features: {fset} ...")
        X = eg.feature_matrix(ds.windows, fs, fset, n_bands)
        protocols["random_windows"]["results"][fset] = _run_folds(X, y, folds_rand, labels, seed)
        protocols["leave_one_session_out"]["results"][fset] = _run_folds(X, y, folds_los, labels, seed, groups=ds.session)

    counts = {lab: int(np.sum(y == lab)) for lab in labels}
    rates = [f["measured_rate_hz"] for f in ds.files if f["measured_rate_hz"]]
    report = {
        "settings": {"fs_hz": fs, "window_samples": win, "step_samples": step, "trim_s": trim_s, "n_bands": n_bands,
                     "tree_max_depth": TREE_DEPTH, "forest_trees": FOREST_TREES, "seed": seed,
                     "note": "model settings were fixed in advance and not tuned"},
        "data": {"files": len(ds.files), "windows": int(len(y)), "sessions": sessions, "labels": labels,
                 "windows_per_label": counts,
                 "sessions_per_label": {lab: len(set(ds.session[y == lab].tolist())) for lab in labels},
                 "synthetic": bool(ds.synthetic),
                 "recorded_rate_hz": ({"min": float(min(rates)), "mean": float(np.mean(rates)), "max": float(max(rates))}
                                      if rates else None),
                 "skipped": [list(s) for s in ds.skipped], "design_problems": problems,
                 "feature_names": {f: eg.feature_names_for(f, n_bands) for f in eg.FEATURE_SETS}},
        "baselines": {"chance_accuracy": 1.0 / len(labels),
                      "majority_accuracy_leave_one_session_out": float(_majority_accuracy(y, folds_los))},
        "protocols": protocols,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (out_dir / "gestures.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = ["| Evaluation | Features | Tree accuracy | Tree macro-F1 | Forest accuracy | Forest macro-F1 |", "|---|---|---|---|---|---|"]
    for p in protocols.values():
        for fset in eg.FEATURE_SETS:
            t, f = p["results"][fset]["tree"], p["results"][fset]["forest"]
            rows.append(f"| {p['title']} | {fset} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | {f['accuracy']:.3f} | {f['macro_f1']:.3f} |")
    banner = "SYNTHETIC DATA, NOT REAL RECORDINGS\n\n" if ds.synthetic else ""
    (out_dir / "gestures.md").write_text(banner + "\n".join(rows) + "\n", encoding="utf-8")
    return report


def best_honest(report):
    """(feature set, model, scores) with the best leave-one-session-out macro-F1."""
    res = report["protocols"]["leave_one_session_out"]["results"]
    best = max(((f, m, s) for f, models in res.items() for m, s in models.items()), key=lambda t: t[2]["macro_f1"])
    return best


def export_tree(data_dir, out_dir="results", feature_set=None, report=None, fs=eg.FS, win=eg.WINDOW, step=eg.STEP,
                trim_s=eg.TRIM_S, n_bands=eg.N_BANDS, seed=0):
    """Train one tree on ALL recordings and write it as C (float thresholds, exact for float32 inputs) plus a JSON file
    listing the feature order and the gesture each returned number means. Not evaluated: it has seen every session."""
    from .export_c import tree_to_c
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = eg.load_dataset(data_dir, fs, win, step, trim_s)
    labels = ordered_labels(ds.y)
    fset = feature_set or (best_honest(report)[0] if report else "motion")
    X = eg.feature_matrix(ds.windows, fs, fset, n_bands)
    y = np.array([labels.index(v) for v in ds.y])
    clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=seed).fit(X, y)
    (out_dir / "gesture_tree.c").write_text(tree_to_c(clf, "gesture_tree_predict", thresholds="float"), encoding="utf-8")
    meta = {"function": "int gesture_tree_predict(const float *x)", "feature_set": fset,
            "feature_names": eg.feature_names_for(fset, n_bands), "classes": labels,
            "window_samples": win, "fs_hz": fs, "n_bands": n_bands, "tree_max_depth": TREE_DEPTH,
            "nodes": int(clf.tree_.node_count), "trained_on": {"windows": int(len(y)), "sessions": sorted(set(ds.session.tolist()))},
            "note": "trained on every session, so it has no held-out score; the honest numbers are in gestures.json"}
    (out_dir / "gesture_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return clf, X, y, labels
