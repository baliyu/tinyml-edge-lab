"""Figures for the UCI HAR baseline.

Uses a headless matplotlib backend so it runs in WSL and CI without a display. Every figure is drawn from the
saved results (results/har_baseline.json) or from the raw test windows, never from typed-in numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import uci_har  # noqa: E402

SHORT = {"WALKING": "walking", "WALKING_UPSTAIRS": "upstairs", "WALKING_DOWNSTAIRS": "downstairs",
         "SITTING": "sitting", "STANDING": "standing", "LAYING": "laying"}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix_fig(cm, class_names, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted activity")
    ax.set_ylabel("true activity")
    ax.set_title("Confusion matrix on UCI's test volunteers\n(each row sums to 100 % of that true class)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    _save(fig, path)


def per_class_recall_fig(recall, path):
    items = sorted(recall.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b6fb6")
    for i, (_, v) in enumerate(items):
        ax.text(min(v + 0.01, 1.0), i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("recall (share of that activity found)")
    ax.set_title("Per-class recall, test volunteers")
    _save(fig, path)


def depth_sweep_fig(sweep, path, chosen_depth=None):
    depths = [s["max_depth"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(depths, [s["test_accuracy"] for s in sweep], "o-", label="test accuracy", color="#3b6fb6")
    ax.plot(depths, [s["test_macro_f1"] for s in sweep], "s--", label="test macro-F1", color="#c0632b")
    if chosen_depth is not None:
        ax.axvline(chosen_depth, color="grey", linestyle=":", label=f"depth {chosen_depth} (fixed in advance)")
    ax.set_xlabel("maximum tree depth")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(depths, [s["nodes"] for s in sweep], "^-", color="#6a8f3a", label="nodes")
    ax2.set_ylabel("tree size (nodes)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax.set_title("Accuracy and size against depth\n(test scores shown for information; not used to pick the depth)", fontsize=10)
    _save(fig, path)


def class_examples_fig(windows, y, fs, path):
    labels = sorted(uci_har.LABELS)
    t = np.arange(windows.shape[1]) / fs
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0), sharey=True, sharex=True)
    for ax, label in zip(axes.ravel(), labels):
        idx = np.flatnonzero(y == label)
        if len(idx):
            w = windows[idx[0]]
            ax.plot(t, np.linalg.norm(w, axis=-1), color="#3b6fb6")
        ax.set_title(SHORT[uci_har.LABELS[label]], fontsize=10)
        ax.grid(alpha=0.3)
    fig.supxlabel("time in one 2.56 s window (s)")
    fig.supylabel("total acceleration magnitude (g)")
    fig.suptitle("One example window per activity (50 Hz smartphone data)")
    _save(fig, path)


def top_features_fig(top, path):
    names = [t["name"] for t in top][::-1]
    vals = [t["importance"] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(names, vals, color="#6a8f3a")
    ax.set_xlabel("importance in the depth-limited tree")
    ax.set_title("Ten most-used features")
    _save(fig, path)


def build_report_figures(root, results_dir="results", fig_dir=None):
    """Draw every figure from results/har_baseline.json and the raw test windows; return the file names."""
    results_dir = Path(results_dir)
    fig_dir = Path(fig_dir) if fig_dir else results_dir / "figures"
    rep = json.loads((results_dir / "har_baseline.json").read_text(encoding="utf-8"))
    labels = sorted(uci_har.LABELS)
    names = [SHORT[uci_har.LABELS[label]] for label in labels]
    confusion_matrix_fig(rep["confusion_matrix"], names, fig_dir / "confusion_matrix.png")
    per_class_recall_fig({SHORT[k]: v for k, v in rep["per_class_recall"].items()}, fig_dir / "per_class_recall.png")
    depth_sweep_fig(rep["depth_sweep"], fig_dir / "depth_sweep.png", chosen_depth=rep["chosen_depth"])
    windows, y, _ = uci_har.load_split(root, "test")
    class_examples_fig(windows, y, uci_har.FS, fig_dir / "class_examples.png")
    top_features_fig(rep["top_features"], fig_dir / "top_features.png")
    return sorted(p.name for p in fig_dir.glob("*.png"))


def cv_curves_fig(cv, chosen, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for key, label, color in (("tree_baseline", "baseline features", "#3b6fb6"),
                              ("tree_extended", "extended features", "#c0632b")):
        rows = cv[key]
        ax.errorbar([r["max_depth"] for r in rows], [r["cv_macro_f1_mean"] for r in rows],
                    yerr=[r["cv_macro_f1_se"] for r in rows], marker="o", capsize=3, label=label, color=color)
        ax.plot(chosen[key]["max_depth"], chosen[key]["cv_macro_f1_mean"], marker="*", markersize=16,
                linestyle="none", color=color)
    ax.set_xlabel("maximum tree depth")
    ax.set_ylabel("cross-validated macro-F1 (training volunteers)")
    ax.set_title("Choosing the depth without the test set\n(grouped by volunteer; star = chosen by the one-standard-error rule)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def model_comparison_fig(models, n_test_volunteers, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {"tree": "#3b6fb6", "forest": "#6a8f3a"}
    offsets = [(8, 8), (8, -16), (8, 22), (8, -30)]   # staggered so labels of nearby points do not overlap
    for i, m in enumerate(models):
        kind = "forest" if m["id"].startswith("forest") else "tree"
        ax.scatter(m["nodes"], m["test_macro_f1"], s=70, color=colors[kind])
        ax.annotate(m["short"], (m["nodes"], m["test_macro_f1"]), textcoords="offset points",
                    xytext=offsets[i % len(offsets)], fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.6})
    ax.set_xscale("log")
    ax.set_xlabel("model size: total nodes (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_title(f"Size against test macro-F1 (one run, {n_test_volunteers} test volunteers)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.margins(x=0.25, y=0.15)
    _save(fig, path)


def build_compare_figures(results_dir="results", fig_dir=None):
    """Draw the comparison figures from results/compare.json; return the file names."""
    results_dir = Path(results_dir)
    fig_dir = Path(fig_dir) if fig_dir else results_dir / "figures"
    rep = json.loads((results_dir / "compare.json").read_text(encoding="utf-8"))
    cv_curves_fig(rep["cv"], rep["chosen"], fig_dir / "cv_curves.png")
    model_comparison_fig(rep["models"], len(rep["test_subjects"]), fig_dir / "model_comparison.png")
    best = next(m for m in rep["models"] if m["id"] == "tree_extended_cv")
    names = [SHORT[uci_har.LABELS[label]] for label in sorted(uci_har.LABELS)]
    confusion_matrix_fig(best["confusion_matrix"], names, fig_dir / "confusion_matrix_extended_tree.png")
    return ["cv_curves.png", "model_comparison.png", "confusion_matrix_extended_tree.png"]
