"""Figures for the model 1 PC results.

AI-assisted: drafted with Claude, checked by the author.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def wilson_interval(k, n, z=1.96):
    """95 percent Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def confusion(y, pred, n_classes=10):
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y, pred):
        m[int(t), int(p)] += 1
    return m


def make_model1_figure(pred_npz, result_json, out_png):
    d = np.load(pred_npz)
    r = json.loads(Path(result_json).read_text())
    y = d["y_test"]
    n = len(y)
    names = ["Keras float", "LiteRT float", "LiteRT int8"]
    preds = [d["keras"], d["float"], d["int8"]]
    accs = [float(np.mean(p == y)) for p in preds]
    cis = [wilson_interval(int(np.sum(p == y)), n) for p in preds]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    err = [
        [a - lo for a, (lo, hi) in zip(accs, cis)],
        [hi - a for a, (lo, hi) in zip(accs, cis)],
    ]
    ax.bar(names, accs, yerr=np.maximum(err, 0.0), capsize=4, color=["#7f7f7f", "#4c78a8", "#e45756"])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Accuracy on the %d test samples" % n)
    ax.set_title("Accuracy (95% Wilson interval)")
    ax.tick_params(axis="x", labelrotation=15)

    ax = axes[1]
    sizes = [r["float_bytes"], r["int8_bytes"]]
    ax.bar(["float .tflite", "int8 .tflite"], sizes, color=["#4c78a8", "#e45756"])
    for i, s in enumerate(sizes):
        ax.text(i, s, str(s), ha="center", va="bottom")
    ax.set_ylabel("File size (bytes)")
    ax.set_title("Model file size")

    ax = axes[2]
    m = confusion(y, d["int8"])
    ax.imshow(m, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted (int8)")
    ax.set_ylabel("True")
    ax.set_title("int8 confusion matrix")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    fig.suptitle(
        "Model 1 on the PC: %d parameters, one training run, seed %d"
        % (r["parameters"], r["seed"])
    )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with open(out_png, "wb") as f:
        fig.savefig(f, format="png", dpi=150)
    plt.close(fig)
    return out_png
