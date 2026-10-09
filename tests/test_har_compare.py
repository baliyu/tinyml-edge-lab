import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from tinyml_lab import har_compare
from tinyml_lab.har_compare import forest_nodes, run_compare
from tinyml_lab.plots import build_compare_figures
from tinyml_lab.readme_block import END, START, render_block, update_readme

SMALL = dict(depths=(2, 3, 4), rf_grid=((5, 3), (10, 4)), n_splits=4, fixed_depth=3, log=lambda *_: None)
PNG = b"\x89PNG\r\n\x1a\n"


def test_selection_only_ever_sees_training_volunteers(bigger_uci_root, tmp_path, monkeypatch):
    seen = []
    real = har_compare.group_cv_macro_f1

    def spy(est, X, y, groups, n_splits=5):
        seen.append((len(X), set(np.unique(groups))))
        return real(est, X, y, groups, n_splits)

    monkeypatch.setattr(har_compare, "group_cv_macro_f1", spy)
    report = run_compare(bigger_uci_root, out_dir=tmp_path / "r", **SMALL)
    n_train = report["n_train"]
    assert seen, "grouped cross-validation was never called"
    for n_rows, groups in seen:
        assert n_rows == n_train                         # never train + test together
        assert groups <= set(report["train_subjects"])   # no test volunteer inside the selection
        assert groups.isdisjoint(report["test_subjects"])


def test_pipeline_outputs_and_flags(bigger_uci_root, tmp_path):
    out = tmp_path / "r"
    report = run_compare(bigger_uci_root, out_dir=out, **SMALL)
    assert [m["id"] for m in report["models"]] == ["tree_fixed6", "tree_baseline_cv", "tree_extended_cv", "forest_extended_cv"]
    flags = {m["id"]: m["selected_by_cv"] for m in report["models"]}
    assert flags == {"tree_fixed6": False, "tree_baseline_cv": True, "tree_extended_cv": True, "forest_extended_cv": True}
    assert report["chosen"]["tree_extended"]["max_depth"] in SMALL["depths"]
    assert report["n_features"]["extended"] > report["n_features"]["baseline"]
    assert set(report["train_subjects"]).isdisjoint(report["test_subjects"])
    saved = json.loads((out / "compare.json").read_text())
    assert saved["versions"]["scikit_learn"]
    assert (out / "compare.md").read_text().startswith("| Model")
    assert "int har_tree_ext_predict(const float *x)" in (out / "har_tree_extended.c").read_text()
    for m in report["models"]:
        assert 0 <= m["test_accuracy"] <= 1 and m["nodes"] >= 1


def test_forest_nodes_adds_up_the_trees():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)); y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=7, max_depth=3, random_state=0).fit(X, y)
    assert forest_nodes(rf) == sum(t.tree_.node_count for t in rf.estimators_)
    assert forest_nodes(rf) >= 7


def test_comparison_figures_are_real_png_files(bigger_uci_root, tmp_path):
    out = tmp_path / "r"
    run_compare(bigger_uci_root, out_dir=out, **SMALL)
    names = build_compare_figures(out)
    assert names == ["cv_curves.png", "model_comparison.png", "confusion_matrix_extended_tree.png"]
    for n in names:
        data = (out / "figures" / n).read_bytes()
        assert data[:8] == PNG and len(data) > 3000


def test_readme_block_is_generated_from_results_and_is_idempotent(bigger_uci_root, tmp_path):
    report = run_compare(bigger_uci_root, out_dir=tmp_path / "r", **SMALL)
    block = render_block(report)
    for m in report["models"]:
        assert f"{m['test_macro_f1']:.3f}" in block      # numbers come from the results, not from typing
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\n| Part | What |\n|---|---|\n| `src/tinyml_lab/plots.py` | figs |\n\n```bash\n"
                      "python scripts/make_figures.py   # x\n```\n\n## Roadmap\n1. next\n")
    update_readme(readme, block)
    once = readme.read_text()
    update_readme(readme, block)
    twice = readme.read_text()
    assert once == twice
    assert once.count(START) == 1 and once.count(END) == 1
    assert once.count("har_compare.py") == 1 and once.count("scripts/compare_models.py") == 1
    assert once.index(START) < once.index("## Roadmap")
    assert once.startswith("# T\n")


def test_readme_update_refuses_when_the_layout_changed(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no roadmap here\n")
    with pytest.raises(SystemExit):
        update_readme(readme, "block")
