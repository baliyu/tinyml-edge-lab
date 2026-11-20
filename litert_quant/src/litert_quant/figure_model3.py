"""Figure for the model 3 PC results (CWRU autoencoder, float against int8, with an RMS baseline).

AI-assisted: drafted with Claude, checked by the author.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

COLORS = {"float": "#4c78a8", "int8": "#e45756", "rms": "#7f7f7f"}


def _log10(v):
    return np.log10(np.maximum(np.asarray(v, dtype=float), 1e-12))


def _pick(auc, group, key):
    if group is None:
        return auc.get("overall")
    return auc.get(group, {}).get(key)


def make_model3_figure(scores_npz, result_json, out_png):
    d = np.load(scores_npz)
    r = json.loads(Path(result_json).read_text())
    y, f, q = d["y_test"], d["float"], d["int8"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    sets = [
        ("train normal (0, 1, 2 hp)", d["train_float"], "#9ecae1"),
        ("test normal (3 hp)", f[y == 0], "#4c78a8"),
        ("test fault", f[y == 1], "#e45756"),
    ]
    allv = np.concatenate([_log10(s) for _, s, _ in sets])
    bins = np.linspace(allv.min(), allv.max() + 1e-9, 60)
    for name, s, c in sets:
        ax.hist(_log10(s), bins=bins, density=True, histtype="step", color=c, label=name, linewidth=1.5)
    thr = r["scores"]["float"]["operating_point"]["threshold"]
    ax.axvline(float(_log10(thr)), color="k", linestyle="--", linewidth=1, label="threshold: 99th percentile of train")
    ax.set_xlabel("log10 reconstruction error (squared training-std units)")
    ax.set_ylabel("Density")
    ax.set_title("Float model scores")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(4000, len(y)), replace=False)
    ax.scatter(f[idx], q[idx], s=4, c=np.where(y[idx] == 1, "#e45756", "#4c78a8"), alpha=0.5)
    lo = max(min(float(f.min()), float(q.min())), 1e-12)
    hi = max(float(f.max()), float(q.max()))
    ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("float score")
    ax.set_ylabel("int8 score")
    v = r["float_vs_int8"]
    ax.set_title(
        "int8 against float ({} windows; red fault, blue normal)\nSpearman {:.4f}, {} decisions differ".format(
            len(idx), v["spearman"], v["decision_disagree"]
        )
    )

    ax = axes[1, 0]
    auc_f = r["scores"]["float"]["auc"]["overall"]
    auc_q = r["scores"]["int8"]["auc"]["overall"]
    auc_r = r["baselines"]["rms"]["overall"]
    for name, score, a in (("float", f, auc_f), ("int8", q, auc_q), ("rms", d["rms"], auc_r)):
        fpr, tpr, _ = roc_curve(y, score)
        label = "{} AE".format(name) if name != "rms" else "RMS threshold"
        ax.plot(np.maximum(fpr, 1e-4), tpr, color=COLORS[name], label="{} (AUC {:.4f})".format(label, a), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0.99, 1.001)
    ax.set_xlabel("False positive rate (log axis; zero plotted at 1e-4)")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the test windows\ny axis starts at 0.99")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 1]
    spec = (
        [("overall", None, None)]
        + [("kind " + k, "by_kind", k) for k in ("B", "IR", "OR")]
        + [("size " + s + " mil", "by_size_mil", s) for s in ("7", "14", "21")]
        + [("load " + n + " hp", "by_load_hp", n) for n in ("0", "1", "2", "3")]
    )
    cols = [r["scores"]["float"]["auc"], r["scores"]["int8"]["auc"], r["baselines"]["rms"]]
    cells = [["-" if x is None else "{:.4f}".format(x) for x in (_pick(c, g, k) for c in cols)] for _, g, k in spec]
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=[s[0] for s in spec], colLabels=["float AE", "int8 AE", "RMS"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("AUC by fault kind, size and load\n(each group against all {} normal test windows)".format(r.get("n_test_normal", "?")))

    fig.suptitle(
        "Model 3 on the PC: {} parameters, one training run (seed {}, {} epochs); normal test windows are one recording (3 hp)".format(
            r["parameters"], r["seed"], r["epochs"]
        )
    )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with open(out_png, "wb") as fh:
        fig.savefig(fh, format="png", dpi=150)
    plt.close(fig)
    return out_png
