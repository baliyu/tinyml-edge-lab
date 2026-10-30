#!/usr/bin/env python3
"""Draw two figures from the microphone recordings: the quiet-room level of every recording, and the held-out score of every train/test split.
Built with AI assistance (Claude).

    python3 tools/mic_plot.py --dir data/own/mic --out results/figures

Figure 1 (mic_quiet_levels.png) shows, for each session's quiet recording, the median and the 10th to 90th percentile of rms. Figure 2
(mic_heldout_f1.png) re-runs the six held-out tests listed in firmware/sparkfun_edge/mic_logger/README.md with the same models and
settings as tools/mic_eval.py, and draws macro-F1 for the tree and the forest. The scores are printed as well, so they can be
compared with the README table. Power state is not recorded for most sittings: pass --power d=battery,e=battery for those you know.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mic_eval as me  # noqa: E402

A = ["s1", "s2", "s3"]                       # recorded back to back: one sitting
SESSIONS = A + ["b", "c", "d", "e"]
TESTS = [("A→b", A, ["b"]), ("b→A", ["b"], A), ("A+b→c", A + ["b"], ["c"]), ("A+c→b", A + ["c"], ["b"]),
         ("b+c→A", ["b", "c"], A), ("A+b+c→d", A + ["b", "c"], ["d"])]
COLOURS = {"battery": "#d95f02", "plugged in": "#1b9e77", "not recorded": "#7f7f7f"}


def parse_power(text: str) -> dict:
    out = {}
    for item in filter(None, (text or "").split(",")):
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip().replace("_", " ")
    return out


def quiet_levels(directory: Path, sessions: list[str]):
    res = {}
    for f in me.read_files(directory, sessions):
        if f["label"] == "quiet" and f["rows"]:
            v = np.array([float(r["rms"]) for r in f["rows"]])
            res[f["session"]] = (float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)))
    return res


def figure_quiet(levels: dict, power: dict, path: Path) -> None:
    names = [s for s in SESSIONS if s in levels]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, s in enumerate(names):
        med, lo, hi = levels[s]
        state = power.get(s, "not recorded")
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", color=COLOURS.get(state, "#7f7f7f"), capsize=4, markersize=7)
        ax.annotate(f"{med:.0f}", (i, med), textcoords="offset points", xytext=(10, -3), fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("session (s1, s2, s3 were recorded back to back: one sitting)")
    ax.set_ylabel("quiet-room rms (ADC counts)")
    ax.set_title("The quiet-room level changes between sittings")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    states = sorted({power.get(s, "not recorded") for s in names})
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLOURS.get(t, "#7f7f7f"), label=t) for t in states],
              title="laptop power", loc="upper right", fontsize=8)
    fig.text(0.01, 0.01, "Bars: 10th to 90th percentile of the 0.25 s windows. One 45 s recording per session.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heldout_scores(directory: Path, sessions_present: set):
    rows = []
    for name, tr, te in TESTS:
        if not (set(tr) | set(te)) <= sessions_present:
            continue
        Xtr, ytr, _ = me.build(directory, tr, False, drop_cal=False)
        Xte, yte, _ = me.build(directory, te, False, drop_cal=False)
        scores = []
        for _, mk in me.models():
            m = mk().fit(Xtr, ytr)
            scores.append(float(f1_score(yte, m.predict(Xte), labels=list(me.CLASSES), average="macro", zero_division=0)))
        rows.append((name, scores[0], scores[1]))
    return rows


def figure_scores(rows: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="decision tree (depth 4)", color="#4c72b0")
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="random forest (100 trees)", color="#dd8452")
    for xi, r in zip(x, rows):
        ax.text(xi - w / 2, r[1] + 0.01, f"{r[1]:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, r[2] + 0.01, f"{r[2]:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 3, color="k", ls=":", lw=1)
    ax.text(len(rows) - 0.5, 1 / 3 + 0.01, "chance", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlabel("trained on → tested once on a held-out sitting (A = s1, s2, s3)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Held-out score depends on which sittings were used")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="data/own/mic")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--power", default="d=battery,e=battery", help="known laptop power states, e.g. d=battery,e=battery")
    a = ap.parse_args(argv)
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    present = {f["session"] for f in me.read_files(d, [s for s in SESSIONS if list(d.glob(f"{s}_*.csv"))])}
    if not present:
        print(f"No recordings found in {d}.", file=sys.stderr)
        return 1
    levels = quiet_levels(d, sorted(present))
    p1 = out / "mic_quiet_levels.png"
    figure_quiet(levels, parse_power(a.power), p1)
    print(f"Wrote {p1}")
    rows = heldout_scores(d, present)
    if rows:
        p2 = out / "mic_heldout_f1.png"
        figure_scores(rows, p2)
        print(f"Wrote {p2}\n\nMacro-F1 plotted (compare with the table in the mic_logger README):")
        print(f"{'train -> test':>14} {'tree':>7} {'forest':>7}")
        for name, t, f in rows:
            print(f"{name:>14} {t:7.3f} {f:7.3f}")
    else:
        print("Not enough sessions for the held-out figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
