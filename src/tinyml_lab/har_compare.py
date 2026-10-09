"""Compare models on UCI HAR with honest model selection.

* Depth (and forest size) are chosen by cross-validation on the training volunteers only, folds split by volunteer.
* The test volunteers are used once per final model, after the choice is made.
* "baseline" features: total acceleration. "extended": adds UCI's body acceleration, gyroscope and gravity.
"""
import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from . import uci_har
from .export_c import tree_to_c
from .features import batch_features, feature_names, gravity_mean, gravity_names, with_magnitude
from .selection import group_cv_macro_f1, one_se_choice

RF_GRID = [(10, 6), (10, 10), (30, 6), (30, 10), (60, 6), (60, 10)]   # (n_estimators, max_depth)
CHANNELS = ["x", "y", "z", "mag"]


def build_features(root, split, n_bands=4):
    """Return ({'baseline': X, 'extended': X}, {'baseline': names, 'extended': names}, labels, subjects)."""
    total = uci_har.load_signal(root, split, "total_acc")
    body = uci_har.load_signal(root, split, "body_acc")
    gyro = uci_har.load_signal(root, split, "body_gyro")
    y, subjects = uci_har.load_meta(root, split)
    f_total = batch_features(with_magnitude(total), uci_har.FS, n_bands)
    f_body = batch_features(with_magnitude(body), uci_har.FS, n_bands)
    f_gyro = batch_features(with_magnitude(gyro), uci_har.FS, n_bands)
    n_total = feature_names([f"acc_{c}" for c in CHANNELS], n_bands)
    n_body = feature_names([f"body_{c}" for c in CHANNELS], n_bands)
    n_gyro = feature_names([f"gyro_{c}" for c in CHANNELS], n_bands)
    sets = {"baseline": f_total, "extended": np.hstack([f_total, f_body, f_gyro, gravity_mean(total, body)])}
    names = {"baseline": n_total, "extended": n_total + n_body + n_gyro + gravity_names()}
    return sets, names, y, subjects


def forest_nodes(rf):
    return int(sum(t.tree_.node_count for t in rf.estimators_))


def forest_leaves(rf):
    return int(sum(t.get_n_leaves() for t in rf.estimators_))


def _se(scores):
    return float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0


def _select_tree(X, y, groups, depths, n_splits, seed):
    rows, means, ses = [], [], []
    for d in depths:
        sc = group_cv_macro_f1(DecisionTreeClassifier(max_depth=d, random_state=seed), X, y, groups, n_splits)
        means.append(float(sc.mean()))
        ses.append(_se(sc))
        rows.append({"max_depth": int(d), "cv_macro_f1_mean": means[-1], "cv_macro_f1_se": ses[-1]})
    return rows, one_se_choice(means, ses, list(depths))


def _select_forest(X, y, groups, grid, n_splits, seed):
    rows, means, ses, nodes = [], [], [], []
    for n_est, depth in grid:
        est = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed, n_jobs=-1)
        sc = group_cv_macro_f1(est, X, y, groups, n_splits)
        full = clone(est).fit(X, y)
        means.append(float(sc.mean()))
        ses.append(_se(sc))
        nodes.append(forest_nodes(full))
        rows.append({"n_estimators": int(n_est), "max_depth": int(depth), "total_nodes": nodes[-1],
                     "cv_macro_f1_mean": means[-1], "cv_macro_f1_se": ses[-1]})
    return rows, one_se_choice(means, ses, nodes)


def _metrics(model, X, y, labels):
    pred = model.predict(X)
    return {"test_accuracy": float(accuracy_score(y, pred)),
            "test_macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_recall": {uci_har.LABELS[l]: float(r) for l, r in
                                 zip(labels, recall_score(y, pred, labels=labels, average=None))},
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def run_compare(root, out_dir="results", depths=tuple(range(2, 11)), rf_grid=tuple(RF_GRID), n_splits=5,
                fixed_depth=6, seed=0, n_bands=4, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("computing features for the training volunteers ...")
    sets_tr, names, ytr, gtr = build_features(root, "train", n_bands)
    log("computing features for the test volunteers ...")
    sets_te, _, yte, gte = build_features(root, "test", n_bands)
    uci_har.assert_subject_disjoint(gtr, gte)
    labels = sorted(uci_har.LABELS)
    depths = list(depths)

    # ---- selection: training volunteers only
    cv, chosen = {}, {}
    for fs in ("baseline", "extended"):
        log(f"choosing the tree depth by grouped cross-validation ({fs} features) ...")
        rows, idx = _select_tree(sets_tr[fs], ytr, gtr, depths, n_splits, seed)
        cv[f"tree_{fs}"] = rows
        chosen[f"tree_{fs}"] = dict(rows[idx])
    log("choosing the forest size by grouped cross-validation (extended features) ...")
    rows, idx = _select_forest(sets_tr["extended"], ytr, gtr, list(rf_grid), n_splits, seed)
    cv["forest_extended"] = rows
    chosen["forest_extended"] = dict(rows[idx])

    # ---- final fits, each evaluated once on the test volunteers
    models = []

    def add(id_, label, short, fs, est, choice, selected, cv_row):
        model = est.fit(sets_tr[fs], ytr)
        if isinstance(model, RandomForestClassifier):
            nodes, leaves = forest_nodes(model), forest_leaves(model)
        else:
            nodes, leaves = int(model.tree_.node_count), int(model.get_n_leaves())
        m = {"id": id_, "label": label, "short": short, "features": fs, "n_features": int(sets_tr[fs].shape[1]),
             "choice": choice, "selected_by_cv": selected, "nodes": nodes, "leaves": leaves,
             "cv_macro_f1_mean": cv_row["cv_macro_f1_mean"] if cv_row else None,
             "cv_macro_f1_se": cv_row["cv_macro_f1_se"] if cv_row else None}
        m.update(_metrics(model, sets_te[fs], yte, labels))
        models.append(m)
        return model

    fixed_row = next((r for r in cv["tree_baseline"] if r["max_depth"] == fixed_depth), None)
    add("tree_fixed6", f"Tree, depth {fixed_depth} fixed in advance", f"tree d{fixed_depth}", "baseline",
        DecisionTreeClassifier(max_depth=fixed_depth, random_state=seed), f"fixed depth {fixed_depth}", False, fixed_row)
    cb = chosen["tree_baseline"]
    add("tree_baseline_cv", "Tree, depth chosen by CV", "tree CV", "baseline",
        DecisionTreeClassifier(max_depth=cb["max_depth"], random_state=seed), f"depth {cb['max_depth']} (CV)", True, cb)
    ce = chosen["tree_extended"]
    tree_ext = add("tree_extended_cv", "Tree, depth chosen by CV", "tree CV + extra features", "extended",
                   DecisionTreeClassifier(max_depth=ce["max_depth"], random_state=seed),
                   f"depth {ce['max_depth']} (CV)", True, ce)
    cf = chosen["forest_extended"]
    add("forest_extended_cv", "Random forest, size chosen by CV", "forest CV + extra features", "extended",
        RandomForestClassifier(n_estimators=cf["n_estimators"], max_depth=cf["max_depth"], random_state=seed, n_jobs=-1),
        f"{cf['n_estimators']} trees, depth {cf['max_depth']} (CV)", True, cf)

    c_src = tree_to_c(tree_ext, "har_tree_ext_predict")
    (out_dir / "har_tree_extended.c").write_text(c_src, encoding="utf-8")
    imp = tree_ext.feature_importances_
    order = np.argsort(imp)[::-1][:10]

    report = {
        "protocol": {"cv_folds": int(min(n_splits, len(np.unique(gtr)))),
                     "selection_data": "training volunteers only",
                     "selection_rule": "least complex model within one standard error of the best cross-validated macro-F1",
                     "fixed_depth": fixed_depth, "n_bands": n_bands, "seed": seed},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_subjects": sorted(int(s) for s in np.unique(gtr)),
        "test_subjects": sorted(int(s) for s in np.unique(gte)),
        "n_features": {k: int(v.shape[1]) for k, v in sets_tr.items()},
        "cv": cv, "chosen": chosen, "models": models,
        "extended_tree_c_source_bytes": len(c_src.encode("utf-8")),
        "extended_tree_top_features": [{"name": names["extended"][i], "importance": float(imp[i])} for i in order],
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    (out_dir / "compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = ["| Model | Features | Choice | Nodes | CV macro-F1 | Test accuracy | Test macro-F1 |", "|---|---|---|---|---|---|---|"]
    for m in models:
        cvs = "-" if m["cv_macro_f1_mean"] is None else f"{m['cv_macro_f1_mean']:.3f} (SE {m['cv_macro_f1_se']:.3f})"
        rows.append(f"| {m['label']} | {m['features']} | {m['choice']} | {m['nodes']} | {cvs} | "
                    f"{m['test_accuracy']:.3f} | {m['test_macro_f1']:.3f} |")
    (out_dir / "compare.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report
