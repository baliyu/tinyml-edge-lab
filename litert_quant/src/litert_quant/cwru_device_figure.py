"""Device figure for the CWRU autoencoder: decisions, cycles, arena and net flash, float against int8.

AI-assisted: drafted with Claude, checked by the author.
Inputs are results/model3_device.json and results/model3_flash.json.
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from litert_quant.har_device_figure import wilson

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("float", "int8")
COLORS = {"float": "#4C78A8", "int8": "#F58518"}
CAPTION = (
    "CWRU bearing-fault autoencoder (dense 128-16-8-16-128) on the Arduino Nano 33 BLE (LiteRT Micro). "
    "Subset: 100 normal windows (3 hp) and 24 per fault kind and size, seed 0. Threshold: each model's own PC "
    "99th percentile of training scores. Milliseconds assume a 64 MHz core clock; cycles are measured. "
    "Net flash = sketch minus the test arrays. One seeded run. AI-assisted: drafted with Claude, checked by the author."
)
CAPTION_WIDTH = 190


def wrap_caption(text=CAPTION, width=CAPTION_WIDTH):
    return textwrap.fill(text, width=width)


def _label(ax, x, y, text, size=8):
    ax.annotate(text, (x, y), ha="center", va="bottom", fontsize=size, xytext=(0, 3), textcoords="offset points")


def make_device_figure(device_json, flash_json, out_png):
    dev = json.loads(Path(device_json).read_text())
    flash = json.loads(Path(flash_json).read_text())
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    xs = [0, 1]

    ax = axes[0]
    n_all = dev["float"]["n"]
    for i, k in enumerate(KINDS):
        n, c = dev[k]["n"], dev[k]["correct_board_done"]
        p = c / n
        lo, hi = wilson(c, n)
        ax.bar(i, p, color=COLORS[k], width=0.6)
        ax.errorbar(i, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="black", capsize=4, lw=1)
        _label(ax, i, hi, "%d of %d\nboard vs PC: %d differ" % (c, n, dev[k]["board_pc_decision_disagree"]), 7)
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("correct threshold decisions (axis starts at 0.90)")
    ax.set_title("(a) Decisions on the " + str(n_all) + "-window subset\nWilson 95% interval", fontsize=9)

    ax = axes[1]
    med = [dev[k]["cycles"]["median"] for k in KINDS]
    for i, k in enumerate(KINDS):
        ax.bar(i, med[i] / 1e6, color=COLORS[k], width=0.6)
        _label(ax, i, med[i] / 1e6, "%.3f M cycles\n%.2f ms at 64 MHz" % (med[i] / 1e6, dev[k]["ms_median_at_64mhz"]), 7)
    ax.set_ylim(0, max(med) / 1e6 * 1.25)
    ax.set_ylabel("median cycles per inference (millions)")
    ax.set_title("(b) Latency\nint8 is %.1fx faster" % (med[0] / med[1]), fontsize=9)

    ax = axes[2]
    arena = [dev[k]["arena_used"] for k in KINDS]
    for i, k in enumerate(KINDS):
        ax.bar(i, arena[i], color=COLORS[k], width=0.6)
        _label(ax, i, arena[i], "%d B" % arena[i])
    ax.set_ylim(0, max(arena) * 1.2)
    ax.set_ylabel("tensor arena used (bytes)")
    ax.set_title("(c) RAM\ntensor arena used", fontsize=9)

    ax = axes[3]
    model = [flash[k]["model_bytes"] / 1000 for k in KINDS]
    rest = [flash[k]["runtime_and_code_bytes"] / 1000 for k in KINDS]
    ax.bar(xs, model, color="#54A24B", width=0.6, label="model")
    ax.bar(xs, rest, bottom=model, color="#BAB0AC", width=0.6, label="runtime and code")
    top = max(m + r for m, r in zip(model, rest)) * 1.35
    for i, k in enumerate(KINDS):
        _label(ax, i, model[i] + rest[i], "%.1f kB" % (flash[k]["net_firmware_bytes"] / 1000))
        if model[i] >= 0.08 * top:
            ax.text(i, model[i] / 2, "model %.1f" % model[i], ha="center", va="center", fontsize=7)
        else:
            ax.text(i, model[i] + 0.01 * top, "model %.1f" % model[i], ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, top)
    ax.set_ylabel("net firmware flash (kB)")
    ax.set_title("(d) Flash\ntest data excluded", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", ncol=2, frameon=False)

    for a in axes:
        a.set_xticks(xs)
        a.set_xticklabels(KINDS)
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, wrap_caption(), fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
