"""Comparison figure: the HAR CNN (float and int8) against the decision tree.

AI-assisted: drafted with Claude, checked by the author.
The tree numbers come from a copy of results/footprint.json (compiled C stubs for
Cortex-M4F) and the tree accuracy from model2_pc.json. The tree has no latency
figure, its RAM excludes the stack, and it needs 40 features that were not
extracted on a board, so panel (c) says what was and was not measured.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from litert_quant.har_device_figure import COLORS, wilson

ROOT = Path(__file__).resolve().parents[2]
TREE_COLOR = "#72B7B2"


def tree_footprint(path, model_id="tree_fixed6", target="cortex-m4f"):
    """Flash and RAM bytes of each variant of one tree model for one target."""
    data = json.loads(Path(path).read_text())
    for m in data["models"]:
        if m["id"] == model_id:
            out = {}
            for v in m["variants"]:
                t = v["targets"][target]
                out[v["variant"]] = {"flash_bytes": t["flash_bytes"], "ram_bytes": t["ram_bytes"]}
            return out
    raise KeyError(model_id)


def make_comparison_figure(pc_json, device_json, flash_json, footprint_json, out_png):
    pc = json.loads(Path(pc_json).read_text())
    dev = json.loads(Path(device_json).read_text())
    fl = json.loads(Path(flash_json).read_text())
    tree = tree_footprint(footprint_json)
    n = pc["n_test"]
    tb = pc["tree_baseline"]
    entries = [
        ("tree d6", round(tb["test_accuracy"] * tb["n_test"]), tb["n_test"], TREE_COLOR),
        ("CNN float", pc["correct_float_tflite"], n, COLORS["float"]),
        ("CNN int8", pc["correct_int8_tflite"], n, COLORS["int8"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1, 1.5, 1.7]})

    ax = axes[0]
    for i, (name, k, nn, col) in enumerate(entries):
        p = k / nn
        lo, hi = wilson(k, nn)
        ax.bar(i, p, color=col, width=0.6)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", capsize=4, lw=1)
        ax.annotate("%.3f" % p, (i, hi), ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0.70, 0.90)
    ax.set_xticks(range(3))
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylabel("accuracy (axis starts at 0.70)")
    ax.set_title("(a) Accuracy, all %s test windows\nWilson 95%% interval, PC run" % format(n, ","), fontsize=9)

    ax = axes[1]
    bars = [
        ("tree\nfloat thr.", tree["float thresholds"]["flash_bytes"], TREE_COLOR, 1.0),
        ("tree\ndouble thr.", tree["double thresholds"]["flash_bytes"], TREE_COLOR, 1.0),
        ("CNN int8\nmodel only", fl["int8"]["model_bytes"], COLORS["int8"], 0.45),
        ("CNN float\nmodel only", fl["float"]["model_bytes"], COLORS["float"], 0.45),
        ("CNN int8\nnet firmware", fl["int8"]["net_firmware_bytes"], COLORS["int8"], 1.0),
        ("CNN float\nnet firmware", fl["float"]["net_firmware_bytes"], COLORS["float"], 1.0),
    ]
    for i, (name, v, col, alpha) in enumerate(bars):
        ax.bar(i, v, color=col, alpha=alpha, width=0.65)
        ax.annotate(format(v, ","), (i, v), ha="center", va="bottom", fontsize=7, xytext=(0, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(100, 1_000_000)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set_ylabel("flash on Cortex-M4F (bytes, log scale)")
    ax.set_title("(b) Flash\nfaded = model only, solid = tree code or whole CNN firmware", fontsize=9)

    ax = axes[2]
    ax.axis("off")
    ms = lambda k: "%.2f ms\n(%.2f M cycles)" % (dev[k]["ms_median_at_64mhz"], dev[k]["cycles"]["median"] / 1e6)
    arena = lambda k: "%s B\ntensor arena" % format(dev[k]["arena_used"], ",")
    rows = [
        ["Latency\n(Nano 33 BLE)", "not measured", ms("float"), ms("int8")],
        ["RAM", "0 B static\n(stack not measured)", arena("float"), arena("int8")],
        ["Input", "40 features\n(extraction not\nmeasured)", "raw 128 x 3\nwindow", "raw 128 x 3\nwindow"],
    ]
    tbl = ax.table(cellText=rows, colLabels=["", "tree d6", "CNN float", "CNN int8"], colWidths=[0.2, 0.28, 0.26, 0.26], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 3.4)
    ax.set_title("(c) What was and was not measured", fontsize=9)

    for a in axes[:2]:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Tree: compiled C stubs for Cortex-M4F from footprint.json; accuracy on the UCI HAR test set. CNN: LiteRT Micro on the Nano 33 BLE; its flash includes the TFLM runtime and Arduino core.\n"
        "CNN accuracy is the PC run on all test windows; the board runs used a 360-window subset and matched the PC exactly. The two are not directly comparable (see README). "
        "One seeded run. AI-assisted: drafted with Claude, checked by the author.",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
