"""Figures for the CWRU experiments, drawn from results/cwru.json and the raw files."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import cwru  # noqa: E402
from .plots import _save, confusion_matrix_fig  # noqa: E402

SHORT_TITLES = {"random_windows": "random split\nof windows\n(leaky)",
                "leave_one_load_out": "leave one\nmotor load out",
                "leave_one_size_out_fault_types": "leave one\nfault size out\n(3 fault types)",
                "leave_one_load_out_400hz": "leave one load out,\nreduced to 400 Hz"}


def protocol_bars_fig(report, path):
    prots = report["protocols"]
    keys = [k for k in SHORT_TITLES if k in prots]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for off, name, color in ((-0.2, "tree", "#3b6fb6"), (0.2, "forest", "#6a8f3a")):
        vals = [prots[k]["models"][name]["macro_f1"] for k in keys]
        bars = ax.bar(x + off, vals, 0.38, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [SHORT_TITLES[k] for k in keys], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("Same data and models, different evaluation protocols")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def _pick(manifest, kind, load=0):
    cands = [e for e in manifest if e["kind"] == kind and e["load_hp"] == load]
    return max(cands, key=lambda e: e["size_in"])


def signatures_fig(data_dir, manifest, path, seconds=0.1, nfft=16384):
    picks = [_pick(manifest, k) for k in ("normal", "IR", "B", "OR") if any(e["kind"] == k for e in manifest)]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 5.2), squeeze=False)
    for j, e in enumerate(picks):
        x, _ = cwru.load_recording(Path(data_dir) / cwru.file_name(e), e["number"], e["rpm"])
        n = int(seconds * cwru.FS)
        axes[0, j].plot(np.arange(n) / cwru.FS * 1000, x[:n], lw=0.5, color="#3b6fb6")
        axes[0, j].set_title(f"{cwru.KIND_NAMES[e['kind']]}" + ("" if e["kind"] == "normal" else f" {e['size_in']} in") + f", {e['load_hp']} hp", fontsize=9)
        axes[0, j].set_xlabel("time (ms)")
        seg = x[:nfft] - x[:nfft].mean()
        spec = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg) + 1e-12)
        axes[1, j].plot(np.fft.rfftfreq(len(seg), 1 / cwru.FS) / 1000, spec, lw=0.5, color="#c0632b")
        axes[1, j].set_xlabel("frequency (kHz)")
    axes[0, 0].set_ylabel("acceleration (g)")
    axes[1, 0].set_ylabel("spectrum (dB)")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle("Drive-end vibration at 48 kHz: one recording per condition")
    fig.tight_layout()
    _save(fig, path)


def build_cwru_figures(data_dir, results_dir="results", fig_dir=None, manifest=None):
    manifest = list(cwru.MANIFEST if manifest is None else manifest)
    results_dir = Path(results_dir)
    fig_dir = Path(fig_dir) if fig_dir else results_dir / "figures"
    rep = json.loads((results_dir / "cwru.json").read_text(encoding="utf-8"))
    protocol_bars_fig(rep, fig_dir / "cwru_protocols.png")
    signatures_fig(data_dir, manifest, fig_dir / "cwru_signatures.png")
    best = rep["protocols"]["leave_one_load_out"]["models"]["forest"]
    confusion_matrix_fig(best["confusion_matrix"], best["labels"], fig_dir / "cwru_confusion_forest.png",
                         title="Random forest, leave one motor load out\n(each row sums to 100 % of that true condition)",
                         xlabel="predicted condition", ylabel="true condition")
    return ["cwru_protocols.png", "cwru_signatures.png", "cwru_confusion_forest.png"]
