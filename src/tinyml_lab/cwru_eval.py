"""Honest evaluation of bearing-fault classifiers on CWRU.

Four protocols, same models with settings fixed in advance (not tuned):
  random_windows                 a random split of windows: LEAKY, shown only to expose the problem
  leave_one_load_out             train on three motor loads, test on the fourth (same damaged bearings)
  leave_one_size_out_fault_types train on two fault sizes, test on the third (a different damaged bearing); 3 fault types
  leave_one_load_out_400hz       as leave_one_load_out, after reducing the same signals to 400 Hz
"""
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import cwru
from .cwru_features import batch_features

TREE_DEPTH = 8
FOREST_TREES = 100


def build_table(data_dir, manifest, win, fs_out=None, n_bands=12, log=None):
    """Features and labels for every window of every file in the manifest."""
    feats, cls, kind, size, load, number = [], [], [], [], [], []
    fs = cwru.FS if fs_out is None else float(fs_out)
    for e in manifest:
        x, _ = cwru.load_recording(Path(data_dir) / cwru.file_name(e), e["number"], e["rpm"])
        if fs_out is not None:
            x = cwru.reduce_rate(x, cwru.FS, fs_out)
        wins = cwru.make_windows(x, win)
        if len(wins) == 0:
            continue
        feats.append(batch_features(wins, fs, n_bands))
        n = len(wins)
        cls += [e["cls"]] * n; kind += [e["kind"]] * n; size += [e["size_in"]] * n
        load += [e["load_hp"]] * n; number += [e["number"]] * n
        if log:
            log(f"  file {e['number']:>3} ({e['cls']}, {e['load_hp']} hp): {n} windows")
    return {"X": np.vstack(feats), "cls": np.array(cls), "kind": np.array(kind), "size": np.array(size),
            "load": np.array(load), "file": np.array(number), "fs": fs}


def lolo_splits(loads):
    """(held-out load, train indices, test indices) for each load."""
    loads = np.asarray(loads)
    for held in sorted(set(loads.tolist())):
        yield held, np.flatnonzero(loads != held), np.flatnonzero(loads == held)


def loso_splits(sizes, is_fault):
    """(held-out fault size, train indices, test indices) among fault windows only."""
    sizes, is_fault = np.asarray(sizes), np.asarray(is_fault)
    idx = np.flatnonzero(is_fault)
    for held in sorted(set(sizes[idx].tolist())):
        yield held, idx[sizes[idx] != held], idx[sizes[idx] == held]


def _models(seed):
    return {"tree": lambda: DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=seed),
            "forest": lambda: RandomForestClassifier(n_estimators=FOREST_TREES, random_state=seed, n_jobs=-1)}


def _scores(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
            "labels": list(labels),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def _run_folds(table, y, folds, labels, seed, fold_key):
    """Fit and test every model on every fold; return fold audit and per-model scores."""
    audit = [{fold_key: _py(held), "n_train": int(len(tr)), "n_test": int(len(te))} for held, tr, te in folds]
    out = {}
    for name, make in _models(seed).items():
        pred = {}
        accs = []
        for held, tr, te in folds:
            model = make().fit(table["X"][tr], y[tr])
            p = model.predict(table["X"][te])
            pred.update(zip(te.tolist(), p.tolist()))
            accs.append(float(accuracy_score(y[te], p)))
        idx = np.array(sorted(pred))
        res = _scores(y[idx], np.array([pred[i] for i in idx]), labels)
        res["fold_accuracy"] = accs
        out[name] = res
    return audit, out


def _py(v):
    return v.item() if hasattr(v, "item") else v


def run_cwru(data_dir, out_dir="results", manifest=None, win_48k=4096, win_400=256, n_bands=12, seed=0, log=print):
    manifest = list(cwru.MANIFEST if manifest is None else manifest)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("computing features at 48 kHz ...")
    t48 = build_table(data_dir, manifest, win_48k, None, n_bands)
    log("computing features after reducing the signals to 400 Hz ...")
    t400 = build_table(data_dir, manifest, win_400, 400, n_bands)
    classes = [c for c in cwru.CLASS_ORDER if c in set(t48["cls"])] + sorted(set(t48["cls"]) - set(cwru.CLASS_ORDER))
    kinds = [k for k in ("IR", "B", "OR") if k in set(t48["kind"])]
    protocols = {}

    log("protocol 1: random split of windows (leaky) ...")
    n = len(t48["cls"])
    tr, te = train_test_split(np.arange(n), test_size=0.3, stratify=t48["cls"], random_state=seed)
    audit, models = _run_folds(t48, t48["cls"], [("random 30%", tr, te)], classes, seed, "held_out")
    protocols["random_windows"] = {"title": "Random split of windows (leaky)", "folds": audit, "models": models}

    log("protocol 2: leave one motor load out ...")
    folds = list(lolo_splits(t48["load"]))
    audit, models = _run_folds(t48, t48["cls"], folds, classes, seed, "test_load")
    for a, (held, trn, _) in zip(audit, folds):
        a["train_loads"] = sorted(set(t48["load"][trn].tolist()))
    protocols["leave_one_load_out"] = {"title": "Leave one motor load out", "folds": audit, "models": models}

    log("protocol 3: leave one fault size out (fault types only) ...")
    folds = list(loso_splits(t48["size"], t48["kind"] != "normal"))
    audit, models = _run_folds(t48, t48["kind"], folds, kinds, seed, "test_size_in")
    for a, (held, trn, _) in zip(audit, folds):
        a["train_sizes_in"] = sorted(set(t48["size"][trn].tolist()))
    protocols["leave_one_size_out_fault_types"] = {"title": "Leave one fault size out (3 fault types, no normal class)",
                                                   "folds": audit, "models": models}

    log("protocol 4: leave one motor load out at 400 Hz ...")
    folds = list(lolo_splits(t400["load"]))
    audit, models = _run_folds(t400, t400["cls"], folds, classes, seed, "test_load")
    for a, (held, trn, _) in zip(audit, folds):
        a["train_loads"] = sorted(set(t400["load"][trn].tolist()))
    protocols["leave_one_load_out_400hz"] = {"title": "Leave one motor load out, signals reduced to 400 Hz",
                                             "folds": audit, "models": models}

    report = {
        "settings": {"fs_hz": cwru.FS, "window_48k_samples": win_48k, "window_400hz_samples": win_400, "n_bands": n_bands,
                     "tree_max_depth": TREE_DEPTH, "forest_trees": FOREST_TREES, "seed": seed,
                     "note": "model settings were fixed in advance and not tuned"},
        "data": {"files": len(manifest), "windows_48k": int(n), "windows_400hz": int(len(t400["cls"])),
                 "classes": classes, "kinds": kinds, "loads": sorted(set(t48["load"].tolist())),
                 "sizes_in": sorted(set(t48["size"][t48["kind"] != "normal"].tolist()))},
        "protocols": protocols,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (out_dir / "cwru.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = ["| Evaluation | Tree accuracy | Tree macro-F1 | Forest accuracy | Forest macro-F1 |", "|---|---|---|---|---|"]
    for p in protocols.values():
        t, f = p["models"]["tree"], p["models"]["forest"]
        rows.append(f"| {p['title']} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | {f['accuracy']:.3f} | {f['macro_f1']:.3f} |")
    (out_dir / "cwru.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report
