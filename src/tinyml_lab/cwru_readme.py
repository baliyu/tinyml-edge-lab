"""Write the CWRU results into the README, generated from results/cwru.json (no typed-in numbers)."""
import re
from pathlib import Path

START = "<!-- cwru:start -->"
END = "<!-- cwru:end -->"
TABLE_ROW = "| `src/tinyml_lab/cwru*.py` | CWRU bearing pipeline: verified download, windowing, features and honest evaluation |"
QUICK_LINE = "python scripts/download_cwru.py && python scripts/run_cwru.py   # CWRU bearings (about 300 MB download)"


def _f(x):
    return f"{x:.3f}"


def render_block(report):
    p, s, d = report["protocols"], report["settings"], report["data"]
    lines = [START, "### Bearing faults (CWRU, run on a PC)", "",
             f"- Data: {d['files']} official recordings (normal, and drive-end inner-race, ball and outer-race faults at "
             f"{', '.join(str(x) for x in d['sizes_in'])} in, at motor loads {', '.join(str(x) for x in d['loads'])} hp), all at 48 kHz so every class "
             "was recorded the same way. Each file is checked against its SHA-256 value before use.",
             f"- {s['window_48k_samples']}-sample windows, five time-domain statistics and {s['n_bands']} spectral band fractions as features. "
             f"A decision tree (depth {s['tree_max_depth']}) and a random forest ({s['forest_trees']} trees), settings fixed in advance and not tuned.", "",
             "| Evaluation | Tree accuracy / macro-F1 | Forest accuracy / macro-F1 |", "|---|---|---|"]
    for pr in p.values():
        t, f = pr["models"]["tree"], pr["models"]["forest"]
        lines.append(f"| {pr['title']} | {_f(t['accuracy'])} / {_f(t['macro_f1'])} | {_f(f['accuracy'])} / {_f(f['macro_f1'])} |")
    fm = {k: v["models"]["forest"]["macro_f1"] for k, v in p.items()}
    sent = (f"For the forest, a random split of windows scored macro-F1 {_f(fm['random_windows'])}; holding out a whole motor load scored "
            f"{_f(fm['leave_one_load_out'])}; holding out a whole fault size (three fault types, no normal class) scored "
            f"{_f(fm['leave_one_size_out_fault_types'])}. After reducing the same recordings to 400 Hz, leave-one-load-out scored "
            f"{_f(fm['leave_one_load_out_400hz'])}.")
    def folds(key, name, fmt):
        pr = p[key]
        accs = pr["models"]["forest"]["fold_accuracy"]
        return ", ".join(f"{fmt(f[name])} {a:.3f}" for f, a in zip(pr["folds"], accs))

    chance = 1.0 / len(p["leave_one_size_out_fault_types"]["models"]["forest"]["labels"])
    lines += ["", sent, "",
              "Forest accuracy on each held-out fold (an average can hide a bad fold; a score below chance means the model was consistently wrong there):",
              f"- leave one motor load out: {folds('leave_one_load_out', 'test_load', lambda v: f'{v} hp')}",
              f"- leave one fault size out (chance is {chance:.3f}): {folds('leave_one_size_out_fault_types', 'test_size_in', lambda v: f'{v} in')}",
              "",
              "How to read it:",
              "- Windows cut from the same recording are close neighbours, so a random split can let the model recognise the recording instead of the fault. That row is shown to expose the risk, not as a result.",
              "- Every damaged bearing was recorded at all four loads, so leaving a load out is easier than meeting a new bearing. Leaving a fault size out is the harder test, because each size is a different damaged bearing.",
              "- The 400 Hz row decimates the same recordings with an anti-alias filter. It shows what a 400 Hz sensor could see of these signals; it says nothing about any real sensor or machine.",
              "- CWRU is a small laboratory benchmark with seeded faults. These scores do not predict field performance.", "",
              "![Protocols compared](results/figures/cwru_protocols.png)",
              "![Vibration signatures](results/figures/cwru_signatures.png)",
              "![Confusion matrix, forest, leave one load out](results/figures/cwru_confusion_forest.png)", END]
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
    if TABLE_ROW not in s:
        rows = list(re.finditer(r"^\| `src/tinyml_lab/[^`]+`.*$", s, re.M))
        if rows:
            s = s[:rows[-1].end()] + "\n" + TABLE_ROW + s[rows[-1].end():]
    if "scripts/run_cwru.py" not in s:
        m = list(re.finditer(r"^python scripts/(?:compare_models|make_figures)\.py.*$", s, re.M))
        if m:
            s = s[:m[-1].end()] + "\n" + QUICK_LINE + s[m[-1].end():]
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
