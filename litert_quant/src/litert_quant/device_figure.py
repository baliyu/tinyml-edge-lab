"""Figure comparing the float and int8 models measured on the Nano 33 BLE.

AI-assisted: drafted with Claude, checked by the author.
Reads the board analysis JSON files and the saved compile summaries.
"""
import json
import math
from pathlib import Path

from litert_quant.compile_log import parse_compile_log

COLORS = {"float": "#4C78A8", "int8": "#F58518"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(centre - half, 0.0), min(centre + half, 1.0)


def make_device_figure(float_json, int8_json, float_compile, int8_compile, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for name, jpath, cpath in (
        ("float", float_json, float_compile),
        ("int8", int8_json, int8_compile),
    ):
        runs.append(
            (name, json.loads(Path(jpath).read_text()), parse_compile_log(Path(cpath).read_text()))
        )
    names = [r[0] for r in runs]
    colors = [COLORS[n] for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0][0]
    med = [r[1]["cycles"]["median"] for r in runs]
    below = [r[1]["cycles"]["median"] - r[1]["cycles"]["min"] for r in runs]
    above = [r[1]["cycles"]["max"] - r[1]["cycles"]["median"] for r in runs]
    ax.bar(names, med, color=colors, yerr=[below, above], capsize=4)
    for i, v in enumerate(med):
        ax.text(i, v * 1.02, "{:,.0f}".format(v), ha="center", va="bottom")
    ax.set_ylim(0, max(med) * 1.15)
    ax.set_title("Time per inference (median DWT cycles)\nfloat is {:.2f}x int8".format(med[0] / med[1]))
    ax.set_ylabel("cycles")

    ax = axes[0][1]
    flash = [r[2]["flash_bytes"] for r in runs]
    ax.bar(names, flash, color=colors)
    for i, v in enumerate(flash):
        ax.text(
            i, v * 1.01, "{:,}\nmodel {:,} B".format(v, runs[i][1]["model_bytes"]),
            ha="center", va="bottom",
        )
    ax.set_ylim(0, max(flash) * 1.2)
    ax.set_title("Flash used by the whole sketch (bytes)")
    ax.set_ylabel("bytes")

    ax = axes[1][0]
    arena = [r[1]["arena_used_bytes"] for r in runs]
    ax.bar(names, arena, color=colors)
    for i, v in enumerate(arena):
        ax.text(i, v * 1.02, "{:,}".format(v), ha="center", va="bottom")
    ax.set_ylim(0, max(arena) * 1.2)
    ax.set_title("Tensor arena actually used (bytes)")
    ax.set_ylabel("bytes")

    ax = axes[1][1]
    acc, lows, highs = [], [], []
    for r in runs:
        n = r[1]["count"]
        k = round(r[1]["board_accuracy"] * n)
        lo, hi = wilson(k, n)
        a = k / n
        acc.append(a)
        lows.append(max(a - lo, 0.0))
        highs.append(max(hi - a, 0.0))
    ax.bar(names, acc, color=colors, yerr=[lows, highs], capsize=4)
    for i, v in enumerate(acc):
        ax.text(i, v + highs[i] + 0.003, "{:.2%}".format(v), ha="center", va="bottom")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Accuracy on the board (95% Wilson interval)\naxis starts at 0.85")
    ax.set_ylabel("accuracy")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    agree = ", ".join("{} {}/{}".format(r[0], r[1]["agree_with_pc"], r[1]["count"]) for r in runs)
    fig.suptitle("Digits CNN on the Arduino Nano 33 BLE: float vs int8 (LiteRT Micro)", fontsize=13)
    fig.text(
        0.5, 0.01,
        "One run per model, {} test samples each. Predictions that agree with the PC run: {}.".format(
            runs[0][1]["count"], agree
        ),
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    with open(out_png, "wb") as fh:
        fig.savefig(fh, format="png", dpi=150)
    plt.close(fig)


def main(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    res = root / "results"
    out = res / "figures" / "model1_device.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_device_figure(
        res / "board_float_run1.json",
        res / "board_int8_fixed_run1.json",
        res / "compile_float.txt",
        res / "compile_int8.txt",
        out,
    )
    print("wrote", out)
    return 0
