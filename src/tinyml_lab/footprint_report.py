"""Write the flash-size results: JSON, a markdown table, a figure and a README section generated from the numbers."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import footprint  # noqa: E402
from .plots import _save  # noqa: E402

START = "<!-- footprint:start -->"
END = "<!-- footprint:end -->"
TABLE_ROW = "| `src/tinyml_lab/footprint*.py` | Real flash and RAM sizes of the exported models, measured with the ARM compiler |"
QUICK_LINE = "python scripts/measure_footprint.py                # real flash sizes (needs the Zephyr SDK or an arm-none-eabi compiler)"
OLD_SENTENCE = "The forest was not exported to C, so its size is shown in nodes only."
NEW_SENTENCE = "Measured flash sizes, including the forest's, are in the flash-size section below."
DEVICE_FLASH = {"cortex-m4f": ("an nRF52840 (Nano 33 BLE) or Apollo3 (SparkFun Edge)", 1 << 20),
                "cortex-m0plus": ("an STM32L072 (192 KB)", 192 * 1024)}


def build_report(tool, compare_report, model_results, refit_match, n_test_windows):
    return {"toolchain": tool["version"],
            "flags": {"common": footprint.COMMON_FLAGS, "link": footprint.LINK_FLAGS,
                      "targets": {k: v["flags"] for k, v in footprint.TARGETS.items()}},
            "target_labels": {k: v["label"] for k, v in footprint.TARGETS.items()},
            "n_test_windows": int(n_test_windows), "refit_matches_saved": refit_match, "models": model_results,
            "not_measured": ["stack use", "execution time", "feature extraction (FFT and statistics)", "anything on a real board"]}


def _b(n):
    return f"{n:,} B"


def write_outputs(out_dir, fp):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "footprint.json").write_text(json.dumps(fp, indent=2), encoding="utf-8")
    rows = ["| Model | Nodes | Thresholds | Cortex-M0+ flash | Cortex-M4F flash | Static RAM (M4F) | Same class as scikit-learn |",
            "|---|---|---|---|---|---|---|"]
    for m in fp["models"]:
        for v in m["variants"]:
            t0, t4, a = v["targets"]["cortex-m0plus"], v["targets"]["cortex-m4f"], v["agreement"]
            rows.append(f"| {m['short']} | {m['nodes']} | {v['variant']} | {_b(t0['flash_bytes'])} | {_b(t4['flash_bytes'])} | "
                        f"{_b(t4['ram_bytes'])} | {a['matches']} of {a['total']} |")
    (out_dir / "footprint.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def footprint_fig(fp, path):
    entries = [(f"{m['short']}\n({v['variant'].split()[0]})", v) for m in fp["models"] for v in m["variants"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(entries) + 1.8), sharey=True)
    for ax, key in zip(axes, ("cortex-m0plus", "cortex-m4f")):
        vals = [max(v["targets"][key]["flash_bytes"], 1) for _, v in entries]
        colors = ["#c0632b" if v["variant"].startswith("double") else "#3b6fb6" for _, v in entries]
        ys = list(range(len(entries)))[::-1]
        ax.barh(ys, vals, color=colors)
        for y, val in zip(ys, vals):
            ax.text(val * 1.08, y, f"{val:,} B", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(left=min(vals) / 2, right=max(vals) * 8)
        ax.set_title(footprint.TARGETS[key]["label"].split(":")[0], fontsize=10)
        ax.set_xlabel("flash used by the classifier (bytes, log scale)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(list(range(len(entries)))[::-1], [lbl for lbl, _ in entries], fontsize=8)
    fig.suptitle("Measured flash of the exported classifiers (orange: double thresholds, blue: float thresholds)")
    fig.tight_layout()
    _save(fig, path)


def render_block(fp):
    ms = {m["id"]: m for m in fp["models"]}
    n = fp["n_test_windows"]

    def flash(m, variant, tgt):
        v = next(x for x in m["variants"] if x["variant"].startswith(variant))
        return v, v["targets"][tgt]["flash_bytes"]

    lines = [START, "### Flash size on a microcontroller (measured with the ARM compiler)", "",
             f"- Each exported classifier was linked for two Cortex-M targets with `{fp['toolchain']}` at `-Os`. Flash is code plus constants plus "
             "initialised data plus any compiler helper routines (soft-float compares on the M0+), minus an identical build with an empty stub.",
             "- Static RAM counts static data only. **Not measured:** stack use, speed, and the feature extraction (FFT and statistics), which a "
             "device would also need. Nothing here ran on a board.",
             f"- \"Same class\" is how often the compiled C function, run on this PC, returns the class scikit-learn predicts, over the {n:,} test windows.", "",
             "| Model | Nodes | Thresholds | Cortex-M0+ flash | Cortex-M4F flash | Static RAM (M4F) | Same class as scikit-learn |",
             "|---|---|---|---|---|---|---|"]
    for m in fp["models"]:
        for v in m["variants"]:
            t0, t4, a = v["targets"]["cortex-m0plus"], v["targets"]["cortex-m4f"], v["agreement"]
            lines.append(f"| {m['short']} | {m['nodes']} | {v['variant']} | {_b(t0['flash_bytes'])} | {_b(t4['flash_bytes'])} | "
                         f"{_b(t4['ram_bytes'])} | {a['matches']} of {a['total']} |")
    te = ms["tree_extended_cv"]
    vd, fd = flash(te, "double", "cortex-m4f")
    vf, ff = flash(te, "float", "cortex-m4f")
    fo = ms["forest_extended_cv"]
    vo, fo_b = flash(fo, "float", "cortex-m4f")
    _, fo_0 = flash(fo, "float", "cortex-m0plus")
    name, cap = DEVICE_FLASH["cortex-m4f"]
    same = ("with the same predictions on all" if vf["agreement"]["matches"] == vf["agreement"]["total"] and vd["agreement"]["matches"] == vd["agreement"]["total"]
            else "and agreeing with scikit-learn on")
    lines += ["",
              f"For the extended tree on a Cortex-M4F, rounding the thresholds down to float32 takes {_b(ff)} against {_b(fd)} with doubles "
              f"({fd / ff:.1f} times smaller), {same} {vf['agreement']['total']:,} test windows. The random forest needs {_b(fo_b)} on the M4F "
              f"({fo_b / ff:.0f} times the tree) and {_b(fo_0)} on the M0+, and returns the same class as scikit-learn on "
              f"{vo['agreement']['matches']:,} of {vo['agreement']['total']:,} windows. For scale, {name} has {cap // 1024} KB of flash "
              f"(datasheet value), so the forest is {100 * fo_b / cap:.1f}% of it.", ""]
    bad = [k for k, ok in fp["refit_matches_saved"].items() if not ok]
    if bad:
        lines += [f"Note: the refitted models for {', '.join(bad)} did not reproduce the saved test accuracy, so their sizes describe a refit, not the exact saved model.", ""]
    lines += ["Why float thresholds are exact: for any float32 input x, `x <= t` and `x <= floor32(t)` are the same test, where floor32 rounds down to a float32. "
              "Rounding to the nearest float32 instead would flip a split when the midpoint of two adjacent floats rounds up; a test covers that case.", "",
              "![Measured flash sizes](results/figures/footprint.png)", END]
    return "\n".join(lines)


def update_readme(path, block):
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
    s = s.replace(OLD_SENTENCE, NEW_SENTENCE)
    if TABLE_ROW not in s:
        rows = list(re.finditer(r"^\| `src/tinyml_lab/[^`]+`.*$", s, re.M))
        if rows:
            s = s[:rows[-1].end()] + "\n" + TABLE_ROW + s[rows[-1].end():]
    if "scripts/measure_footprint.py" not in s:
        m = list(re.finditer(r"^python scripts/(?:run_cwru|compare_models|make_figures)\.py.*$", s, re.M))
        if m:
            s = s[:m[-1].end()] + "\n" + QUICK_LINE + s[m[-1].end():]
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
