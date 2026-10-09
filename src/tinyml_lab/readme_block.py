"""Write the comparison results into the README, generated from results/compare.json (no typed-in numbers)."""
import re
from pathlib import Path

START = "<!-- compare:start -->"
END = "<!-- compare:end -->"
TABLE_ROW = "| `src/tinyml_lab/har_compare.py` | Subject-grouped model selection, richer features and a random-forest comparison |"
QUICK_LINE = "python scripts/compare_models.py                       # grouped CV, richer features, forest comparison"


def _f(x):
    return f"{x:.3f}"


def render_block(report):
    ms = {m["id"]: m for m in report["models"]}
    p = report["protocol"]
    n_tr, n_te = len(report["train_subjects"]), len(report["test_subjects"])
    lines = [START, "### Model selection and richer features", "", "How the models were chosen:",
             f"- Depth (and forest size) came from {p['cv_folds']}-fold cross-validation on the {n_tr} training volunteers only, "
             "with folds split by volunteer. The smallest model within one standard error of the best score was kept.",
             f"- The {n_te} test volunteers were used once per final model, after the choice was made.",
             "- Extended features add UCI's body acceleration, gyroscope and gravity signals to the total-acceleration features. "
             "On a device the body/gravity separation would need a filter, which is not implemented here.", "",
             "| Model | Features | Choice | Nodes | CV macro-F1 | Test accuracy | Test macro-F1 |", "|---|---|---|---|---|---|---|"]
    for m in report["models"]:
        cvs = "-" if m["cv_macro_f1_mean"] is None else f"{_f(m['cv_macro_f1_mean'])} (SE {_f(m['cv_macro_f1_se'])})"
        lines.append(f"| {m['label']} | {m['features']} | {m['choice']} | {m['nodes']} | {cvs} | "
                     f"{_f(m['test_accuracy'])} | {_f(m['test_macro_f1'])} |")
    tb, te, rf = ms["tree_baseline_cv"], ms["tree_extended_cv"], ms["forest_extended_cv"]
    weakest = min(te["per_class_recall"].items(), key=lambda kv: kv[1])
    lines += ["",
              f"Adding the extra features moved the compact tree's test macro-F1 from {_f(tb['test_macro_f1'])} to "
              f"{_f(te['test_macro_f1'])} ({te['nodes']} nodes against {tb['nodes']}). The random forest scored "
              f"{_f(rf['test_macro_f1'])} with {rf['nodes']} nodes in total. The extended tree's weakest class is "
              f"{weakest[0].lower().replace('_', ' ')} (recall {_f(weakest[1])}).",
              "",
              f"These are single runs with {n_te} test volunteers, so differences of a few points should not be over-read. "
              "The forest was not exported to C, so its size is shown in nodes only.", "",
              "![Choosing the depth](results/figures/cv_curves.png)",
              "![Size against test macro-F1](results/figures/model_comparison.png)",
              "![Confusion matrix of the extended tree](results/figures/confusion_matrix_extended_tree.png)", END]
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
        m = re.search(r"^\| `src/tinyml_lab/plots\.py`.*$", s, re.M)
        if m:
            s = s[:m.end()] + "\n" + TABLE_ROW + s[m.end():]
    if "scripts/compare_models.py" not in s:
        m = re.search(r"^python scripts/make_figures\.py.*$", s, re.M)
        if m:
            s = s[:m.end()] + "\n" + QUICK_LINE + s[m.end():]
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
