"""Baseline: decision tree on window features, evaluated on UCI's subject-disjoint test split."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from . import uci_har
from .export_c import tree_to_c
from .features import batch_features, feature_names


def build_features(root, split, n_bands=4):
    windows, y, subjects = uci_har.load_split(root, split)
    mag = np.linalg.norm(windows, axis=-1, keepdims=True)
    windows = np.concatenate([windows, mag], axis=-1)  # x, y, z and magnitude
    return batch_features(windows, uci_har.FS, n_bands), y, subjects


def run(root, depths=(2, 3, 4, 5, 6, 8, 10), out_dir="results", chosen_depth=6, n_bands=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, str_ = build_features(root, "train", n_bands)
    Xte, yte, ste = build_features(root, "test", n_bands)
    uci_har.assert_subject_disjoint(str_, ste)

    sweep = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        sweep.append({"max_depth": d, "nodes": int(clf.tree_.node_count), "leaves": int(clf.get_n_leaves()),
                      "test_accuracy": float(accuracy_score(yte, pred)),
                      "test_macro_f1": float(f1_score(yte, pred, average="macro"))})

    clf = DecisionTreeClassifier(max_depth=chosen_depth, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels = sorted(uci_har.LABELS)
    report = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_subjects": sorted(int(s) for s in np.unique(str_)),
        "test_subjects": sorted(int(s) for s in np.unique(ste)),
        "chosen_depth": chosen_depth,
        "nodes": int(clf.tree_.node_count), "leaves": int(clf.get_n_leaves()),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "per_class_recall": {uci_har.LABELS[l]: float(r) for l, r in
                             zip(labels, recall_score(yte, pred, labels=labels, average=None))},
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        "feature_names": feature_names(["x", "y", "z", "mag"], n_bands),
        "depth_sweep": sweep,
    }
    (out_dir / "har_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "har_tree.c").write_text(tree_to_c(clf, "har_tree_predict"), encoding="utf-8")
    rows = ["| max depth | nodes | leaves | test accuracy | test macro-F1 |", "|---|---|---|---|---|"]
    rows += [f"| {s['max_depth']} | {s['nodes']} | {s['leaves']} | {s['test_accuracy']:.3f} | {s['test_macro_f1']:.3f} |"
             for s in sweep]
    (out_dir / "depth_sweep.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report
