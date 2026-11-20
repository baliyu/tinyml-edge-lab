"""Figure for the model 2 PC results (HAR 1D-CNN against the decision tree).

AI-assisted: drafted with Claude, checked by the author.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from litert_quant.figures import ROOT, confusion, wilson_interval  # noqa: E402,F401
from litert_quant.har_data import LABELS  # noqa: E402

SHORT = ("Walk", "Up", "Down", "Sit", "Stand", "Lay")


def disagreement_counts(y, float_pred, int8_pred):
    """Split the windows where float and int8 disagree by which one is right."""
    y, f, q = np.asarray(y), np.asarray(float_pred), np.asarray(int8_pred)
    d = f != q
    return {
        "float_right": int(np.sum(d & (f == y))),
        "int8_right": int(np.sum(d & (q == y))),
        "both_wrong": int(np.sum(d & (f != y) & (q != y))),
        "total": int(np.sum(d)),
    }


def make_model2_figure(pred_npz, result_json, baseline_json, out_png):
    d = np.load(pred_npz)
    r = json.loads(Path(result_json).read_text())
    base = json.loads(Path(baseline_json).read_text())
    y = d["y_test"]
    n = len(y)

    names = ["Decision tree\n(40 features)", "Keras float", "LiteRT float", "LiteRT int8"]
    corrects = [int(round(base["test_accuracy"] * n))] + [int(np.sum(d[k] == y)) for k in ("keras", "float", "int8")]
    accs = [k / n for k in corrects]
    cis = [wilson_interval(k, n) for k in corrects]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    err = [[a - lo for a, (lo, hi) in zip(accs, cis)], [hi - a for a, (lo, hi) in zip(accs, cis)]]
    ax.bar(names, accs, yerr=np.maximum(err, 0.0), capsize=4, color=["#7f7f7f", "#9ecae1", "#4c78a8", "#e45756"])
    for i, a in enumerate(accs):
        ax.text(i, cis[i][1] + 0.003, "%.3f" % a, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("Accuracy on the %d test windows" % n)
    ax.set_title("Accuracy (95% Wilson interval)\naxis starts at 0.70")

    ax = axes[0, 1]
    x = np.arange(len(LABELS))
    tree = [base["per_class_recall"][c] for c in LABELS]
    flo = [r["per_class_recall_float"][c] for c in LABELS]
    q8 = [r["per_class_recall_int8"][c] for c in LABELS]
    ax.bar(x - 0.27, tree, 0.27, label="tree", color="#7f7f7f")
    ax.bar(x, flo, 0.27, label="LiteRT float", color="#4c78a8")
    ax.bar(x + 0.27, q8, 0.27, label="LiteRT int8", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Recall")
    ax.set_title("Recall per class")
    ax.legend(loc="upper center", ncol=3, fontsize=8)

    ax = axes[1, 0]
    m = confusion(y, d["int8"], len(LABELS))
    ax.imshow(m, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(SHORT)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted (int8)")
    ax.set_ylabel("True")
    ax.set_title("int8 confusion matrix")

    ax = axes[1, 1]
    c = disagreement_counts(y, d["float"], d["int8"])
    vals = [c["float_right"], c["int8_right"], c["both_wrong"]]
    labels = ["float right,\nint8 wrong", "int8 right,\nfloat wrong", "both wrong,\npredict differently"]
    ax.bar(labels, vals, color=["#4c78a8", "#e45756", "#bab0ac"])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Windows")
    ax.set_title("%d of %d windows: float and int8 disagree" % (c["total"], n))

    fig.suptitle(
        "Model 2 on the PC: %d parameters, one training run (seed %d, %d epochs); tree from results/har_baseline.json"
        % (r["parameters"], r["seed"], r["epochs"])
    )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with open(out_png, "wb") as f:
        fig.savefig(f, format="png", dpi=150)
    plt.close(fig)
    return out_png
