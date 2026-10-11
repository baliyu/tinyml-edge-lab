"""Figures and the README block for the desk-gesture results, generated from results/gestures.json (no typed-in numbers).

Results computed from synthetic recordings are never written into the README: render_block refuses them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import edge_gestures as eg  # noqa: E402
from .edge_gesture_eval import best_honest  # noqa: E402
from .plots import _save, confusion_matrix_fig  # noqa: E402

START = "<!-- gestures:start -->"
END = "<!-- gestures:end -->"
TABLE_ROW = "| `src/tinyml_lab/edge_gesture*.py` | Desk gestures from the Edge's own recordings: loader, resampling, session-wise evaluation |"
QUICK_LINE = "python scripts/run_gestures.py                      # desk gestures: leave-one-session-out evaluation (needs recordings, see docs/GESTURES.md)"
HEADING = "### Desk gestures (SparkFun Edge, run on a PC)"


def _f(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- figures
SHORT = {("random_windows", "raw"): "random split\n(leaky)\nraw features",
         ("random_windows", "motion"): "random split\n(leaky)\nmotion features",
         ("leave_one_session_out", "raw"): "leave one\nsession out\nraw features",
         ("leave_one_session_out", "motion"): "leave one\nsession out\nmotion features"}


def protocol_bars_fig(report, path):
    keys = [k for k in SHORT if k[0] in report["protocols"]]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for off, name, color in ((-0.2, "tree", "#3b6fb6"), (0.2, "forest", "#6a8f3a")):
        vals = [report["protocols"][p]["results"][f][name]["macro_f1"] for p, f in keys]
        bars = ax.bar(x + off, vals, 0.38, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(report["baselines"]["chance_accuracy"], color="grey", ls="--", lw=1)
    ax.set_xlim(-0.6, len(keys) - 0.4 + 0.45)
    ax.text(len(keys) - 0.55, report["baselines"]["chance_accuracy"] + 0.012, "chance", ha="left", fontsize=8, color="grey")
    ax.set_xticks(x, [SHORT[k] for k in keys], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("Same recordings and models, different evaluation protocols")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def examples_fig(ds, path, seconds=5.0):
    """One example per gesture from the middle session, mean removed so the traces share a scale."""
    labels = [g for g in eg.GESTURES if g in set(ds.y.tolist())] + sorted(set(ds.y.tolist()) - set(eg.GESTURES))
    sessions = sorted(set(ds.session.tolist()))
    sess = sessions[len(sessions) // 2]
    n = int(seconds * ds.fs)
    fig, axes = plt.subplots(len(labels), 1, figsize=(7.5, 1.45 * len(labels) + 0.8), sharex=True, squeeze=False)
    for ax, lab in zip(axes[:, 0], labels):
        idx = np.flatnonzero((ds.y == lab) & (ds.session == sess))
        w = ds.windows[idx[len(idx) // 2]][:n]
        w = w - w.mean(axis=0)
        for i, (c, nm) in enumerate(zip(("#3b6fb6", "#c0632b", "#6a8f3a"), "xyz")):
            ax.plot(np.arange(len(w)) / ds.fs, w[:, i], lw=0.8, color=c, label=nm)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", ncol=3, fontsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(f"One example window per gesture (session {sess}), acceleration in mg with the mean removed")
    fig.tight_layout()
    _save(fig, path)


def build_gesture_figures(ds, results_dir="results", fig_dir=None):
    results_dir = Path(results_dir)
    fig_dir = Path(fig_dir) if fig_dir else results_dir / "figures"
    rep = json.loads((results_dir / "gestures.json").read_text(encoding="utf-8"))
    protocol_bars_fig(rep, fig_dir / "gestures_protocols.png")
    fset, model, best = best_honest(rep)
    confusion_matrix_fig(best["confusion_matrix"], best["labels"], fig_dir / "gestures_confusion.png",
                         title=f"{model} on {fset} features, leave one session out\n(each row sums to 100 % of that true gesture)",
                         xlabel="predicted gesture", ylabel="true gesture")
    examples_fig(ds, fig_dir / "gestures_examples.png")
    return ["gestures_protocols.png", "gestures_confusion.png", "gestures_examples.png"]


# ---------------------------------------------------------------- README block
def render_block(report):
    d, s, b = report["data"], report["settings"], report["baselines"]
    if d["synthetic"]:
        raise ValueError("these results come from SYNTHETIC recordings; they must not be written into the README")
    p = report["protocols"]
    los, rnd = p["leave_one_session_out"], p["random_windows"]
    fset, model, best = best_honest(report)
    cm = np.asarray(best["confusion_matrix"], dtype=float)
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1.0)
    weakest = min(zip(best["labels"], recall), key=lambda kv: kv[1])
    rate = d["recorded_rate_hz"]
    if rate:
        rate_text = (f"about {rate['mean']:.1f} Hz" if rate["max"] - rate["min"] < 0.1
                     else f"{rate['min']:.1f} to {rate['max']:.1f} Hz")
    ties = sum(1 for ms in los["results"].values() for sc in ms.values()
               if abs(sc["macro_f1"] - best["macro_f1"]) < 1e-9) - 1
    tie_text = f" ({ties} other setting{'s' if ties != 1 else ''} scored the same)" if ties else ""
    weakest_text = ("That model classified every window of the held-out sessions correctly." if weakest[1] >= 0.9995
                    else f"Its weakest gesture is {weakest[0].replace('_', ' ')} (recall {_f(weakest[1])}).")
    caution = ("Scores this close to perfect on so few sessions say that the gestures are easy to tell apart, or that the "
               "sessions are too alike; they do not show that the model would work for another person or another board. "
               if best["macro_f1"] >= 0.98 else "")
    lines = [START, HEADING, "",
             f"- Data: {d['files']} recordings of {len(d['labels'])} gestures ({', '.join(d['labels'])}) in {len(d['sessions'])} sessions "
             f"({', '.join(d['sessions'])}), one person and one board. Each recording was resampled to exactly {s['fs_hz']:g} Hz"
             + (f" (the accelerometer itself ran at {rate_text} for a configured 50 Hz)" if rate else "")
             + f", {s['trim_s']:g} s was trimmed at both ends, and {s['window_samples']}-sample windows "
             f"({s['window_samples'] / s['fs_hz']:.2f} s) were cut every {s['step_samples']} samples: {d['windows']} windows in total.",
             "- Features: `raw` includes the board's orientation (window means); `motion` removes each window's mean first, so "
             "how the board was held drops out to first order.",
             f"- A decision tree (depth {s['tree_max_depth']}) and a random forest ({s['forest_trees']} trees), settings fixed in advance and not tuned.",
             "", "| Evaluation | Features | Tree accuracy / macro-F1 | Forest accuracy / macro-F1 |", "|---|---|---|---|"]
    for prot in (rnd, los):
        for fs_ in eg.FEATURE_SETS:
            t, f = prot["results"][fs_]["tree"], prot["results"][fs_]["forest"]
            lines.append(f"| {prot['title']} | {fs_} | {_f(t['accuracy'])} / {_f(t['macro_f1'])} | {_f(f['accuracy'])} / {_f(f['macro_f1'])} |")
    folds = ", ".join(f"{f['held_out']} {_f(a)}" for f, a in zip(los["folds"], best["fold_accuracy"]))
    leaky = rnd["results"][fset][model]
    lines += ["",
              f"Chance accuracy is {_f(b['chance_accuracy'])}; always answering the commonest training gesture scores "
              f"{_f(b['majority_accuracy_leave_one_session_out'])} when a session is held out. "
              f"The best honest setting, the {model} on {fset} features, scored macro-F1 {_f(best['macro_f1'])}{tie_text} when a whole "
              f"session was held out; the same model scored {_f(leaky['macro_f1'])} on a random split of windows. Its accuracy per "
              f"held-out session: {folds}. {weakest_text}",
             "",
              f"{caution}These are {len(d['sessions'])} sessions from one person and one board, so the numbers say little about other "
              "people or other boards. Windows overlap by half, so neighbouring windows are not independent.",
              "", "![Protocols](results/figures/gestures_protocols.png)",
              "![Confusion matrix](results/figures/gestures_confusion.png)",
              "![Example windows](results/figures/gestures_examples.png)", END]
    return "\n".join(lines)


def placeholder_block():
    return "\n".join([START, HEADING, "",
                      "Status: the recording tool, loader, evaluation and C export are written and tested on synthetic data only. "
                      "No real gesture recordings exist yet, so there are no results to report. The recording protocol is in "
                      "`docs/GESTURES.md`.", END])


def _insert_after_last_table_row(s, row):
    m = re.search(r"^## What is here\s*$", s, re.M)
    if not m:
        return s
    nxt = re.search(r"^## ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    rows = list(re.finditer(r"^\|.*$", s[m.end():end], re.M))
    if not rows:
        return s
    at = m.end() + rows[-1].end()
    return s[:at] + "\n" + row + s[at:]


def _insert_quick_line(s, line):
    m = re.search(r"^## Quick start\s*$", s, re.M)
    if not m:
        return s
    fence = re.search(r"^```.*$", s[m.end():], re.M)
    if not fence:
        return s
    close = re.search(r"^```\s*$", s[m.end() + fence.end():], re.M)
    if not close:
        return s
    at = m.end() + fence.end() + close.start()
    return s[:at] + line + "\n" + s[at:]


def update_readme(path, block):
    """Replace the gestures block (or add it before '## Roadmap'); add the module-table row and the quick-start line once."""
    p = Path(path)
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if START in s and END in s:
        a, b = s.index(START), s.index(END) + len(END)
        s = s[:a] + block + s[b:]
    else:
        if s.count("## Roadmap") != 1:
            raise SystemExit("README layout changed: cannot find the Roadmap heading")
        s = s.replace("## Roadmap", block + "\n\n## Roadmap")
    if "edge_gesture" not in s.split(START)[0] + s.split(END)[-1]:     # the row, outside the block itself
        s = _insert_after_last_table_row(s, TABLE_ROW)
    if "scripts/run_gestures.py" not in s.split(START)[0] + s.split(END)[-1]:
        s = _insert_quick_line(s, QUICK_LINE)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
