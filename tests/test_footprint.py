import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyml_lab import footprint, har_compare
from tinyml_lab.export_c import forest_to_c, tree_to_c
from tinyml_lab.footprint_report import (END, NEW_SENTENCE, OLD_SENTENCE, START, build_report, footprint_fig, render_block,
                                         update_readme, write_outputs)

TOOL = footprint.find_toolchain()
needs_arm = pytest.mark.skipif(TOOL is None, reason="no ARM compiler found (set ARM_GCC or ZEPHYR_SDK_INSTALL_DIR)")
PNG = b"\x89PNG\r\n\x1a\n"


def _data(n_features=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1500, n_features)).astype(np.float32)
    y = ((X[:, 0] + X[:, 3] + 0.5 * rng.normal(size=1500)) > 0).astype(int) + (X[:, 7] > 1).astype(int)
    return X, y


# ---------------------------------------------------------------- finding the compiler, reading its output
def test_parse_berkeley_size():
    text = "   text\t   data\t    bss\t    dec\t    hex\tfilename\n   1234\t     8\t    16\t   1258\t    4ea\tx.elf\n"
    assert footprint.parse_berkeley_size(text) == {"text": 1234, "data": 8, "bss": 16}
    with pytest.raises(ValueError):
        footprint.parse_berkeley_size("nothing useful")


def _fake_toolchain(tmp_path, name="arm-none-eabi"):
    d = tmp_path / "bin"
    d.mkdir()
    for tool in ("gcc", "size"):
        f = d / f"{name}-{tool}"
        f.write_text('#!/bin/sh\necho "fake gcc 1.2.3"\n')
        f.chmod(0o755)
    return d / f"{name}-gcc"


def test_find_toolchain_honours_ARM_GCC_and_the_sdk_folder_and_returns_none_otherwise(tmp_path, monkeypatch):
    gcc = _fake_toolchain(tmp_path)
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    found = footprint.find_toolchain({"ARM_GCC": str(gcc), "PATH": "/nonexistent"})
    assert found["gcc"] == str(gcc) and found["version"] == "fake gcc 1.2.3"
    sdk = tmp_path / "home" / "zephyr-sdk-9.9.9" / "gnu" / "arm-zephyr-eabi" / "bin"
    sdk.mkdir(parents=True)
    for tool in ("gcc", "size"):
        (sdk / f"arm-zephyr-eabi-{tool}").write_text('#!/bin/sh\necho "sdk gcc"\n')
        (sdk / f"arm-zephyr-eabi-{tool}").chmod(0o755)
    assert footprint.find_toolchain({"PATH": "/nonexistent"})["version"] == "sdk gcc"          # found by the usual folder
    odd = tmp_path / "somewhere" / "unusual" / "sdk" / "gnu" / "arm-zephyr-eabi" / "bin"      # a place the usual-folder search would not look
    odd.mkdir(parents=True)
    for tool in ("gcc", "size"):
        (odd / f"arm-zephyr-eabi-{tool}").write_text('#!/bin/sh\necho "odd sdk gcc"\n')
        (odd / f"arm-zephyr-eabi-{tool}").chmod(0o755)
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path / "empty")))
    assert footprint.find_toolchain({"PATH": "/nonexistent"}) is None
    assert footprint.find_toolchain({"ZEPHYR_SDK_INSTALL_DIR": str(tmp_path / "somewhere" / "unusual" / "sdk"), "PATH": "/nonexistent"})["version"] == "odd sdk gcc"
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path / "empty")))
    assert footprint.find_toolchain({"PATH": "/nonexistent"}) is None


# ---------------------------------------------------------------- measuring (needs the ARM compiler)
@needs_arm
def test_the_empty_stub_is_subtracted_so_an_empty_model_costs_nothing():
    src = "int f(const float *x)\n{\n    (void)x;\n    return 0;\n}\n"
    for target in footprint.TARGETS:
        r = footprint.measure(src, "f", 4, target, TOOL)
        assert 0 <= r["flash_bytes"] <= 8 and r["ram_bytes"] == 0
        assert r["linked_flash_bytes"] >= r["stub_flash_bytes"] > 0


@needs_arm
def test_float_thresholds_take_less_flash_than_doubles_and_a_forest_takes_more_than_a_tree():
    X, y = _data()
    tree = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)
    rows = {r["variant"]: r for r in footprint.measure_model(tree, X[:300], TOOL)}
    assert set(rows) == {"double thresholds", "float thresholds"}
    for target in footprint.TARGETS:
        assert rows["float thresholds"]["targets"][target]["flash_bytes"] < rows["double thresholds"]["targets"][target]["flash_bytes"]
    for r in rows.values():
        assert r["agreement"]["matches"] == r["agreement"]["total"] == 300       # both encodings are exact
    forest_row = footprint.measure_model(rf, X[:300], TOOL)[0]
    for target in footprint.TARGETS:
        assert forest_row["targets"][target]["flash_bytes"] > 3 * rows["float thresholds"]["targets"][target]["flash_bytes"]
    assert forest_row["agreement"]["rate"] >= 0.99


@needs_arm
def test_a_bigger_tree_needs_more_flash():
    X, y = _data()
    small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    big = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X, y)
    f = lambda m: footprint.measure(tree_to_c(m, "tree_predict", "float"), "tree_predict", X.shape[1], "cortex-m4f", TOOL)["flash_bytes"]
    assert f(big) > f(small)


def test_host_agreement_is_complete_for_exact_exports():
    X, y = _data()
    tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    a = footprint.host_agreement(tree_to_c(tree, "tree_predict", "float"), "tree_predict", tree, X[:400])
    assert a == {"matches": 400, "total": 400, "rate": 1.0}
    rf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(X, y)
    assert footprint.host_agreement(forest_to_c(rf, "forest_predict"), "forest_predict", rf, X[:400])["rate"] >= 0.99


# ---------------------------------------------------------------- rebuilding the saved models
def test_refit_reproduces_the_saved_models_exactly(bigger_uci_root, tmp_path):
    rep = har_compare.run_compare(bigger_uci_root, out_dir=tmp_path / "r", depths=(2, 3, 4), rf_grid=((5, 3), (10, 4)),
                                  n_splits=4, fixed_depth=3, log=lambda *_: None)
    sets_tr, _, ytr, _ = har_compare.build_features(bigger_uci_root, "train")
    sets_te, _, yte, _ = har_compare.build_features(bigger_uci_root, "test")
    models = footprint.refit_models(rep, sets_tr, ytr)
    assert set(models) == {"tree_fixed6", "tree_baseline_cv", "tree_extended_cv", "forest_extended_cv"}
    assert all(footprint.refit_matches_saved(rep, models, sets_te, yte).values())
    wrong = dict(rep, protocol=dict(rep["protocol"], fixed_depth=rep["protocol"]["fixed_depth"] + 1))
    bad = footprint.refit_models(wrong, sets_tr, ytr)
    assert not footprint.refit_matches_saved(rep, bad, sets_te, yte)["tree_fixed6"]


# ---------------------------------------------------------------- report writers (no compiler needed)
def _fake_report():
    def row(variant, m0, m4, ram=0, matches=100, total=100):
        return {"variant": variant, "c_source_bytes": 1000, "agreement": {"matches": matches, "total": total, "rate": matches / total},
                "targets": {"cortex-m0plus": {"flash_bytes": m0, "ram_bytes": ram}, "cortex-m4f": {"flash_bytes": m4, "ram_bytes": ram}}}

    def model(id_, short, nodes, variants):
        return {"id": id_, "label": short, "short": short, "features": "x", "n_features": 3, "nodes": nodes, "leaves": nodes // 2, "variants": variants}
    models = [model("tree_fixed6", "tree d6", 61, [row("double thresholds", 3000, 3100), row("float thresholds", 1500, 900)]),
              model("tree_baseline_cv", "tree CV", 17, [row("double thresholds", 1000, 1100), row("float thresholds", 500, 300)]),
              model("tree_extended_cv", "tree CV + extra", 33, [row("double thresholds", 2000, 2100), row("float thresholds", 900, 600)]),
              model("forest_extended_cv", "forest", 2186, [row("float thresholds", 61000, 65000, matches=99, total=100)])]
    tool = {"version": "fake-gcc 1.0"}
    return build_report(tool, {}, models, {m["id"]: True for m in models}, 100)


def test_report_files_figure_and_block(tmp_path):
    fp = _fake_report()
    write_outputs(tmp_path, fp)
    assert (tmp_path / "footprint.md").read_text().startswith("| Model")
    assert '"toolchain": "fake-gcc 1.0"' in (tmp_path / "footprint.json").read_text()
    footprint_fig(fp, tmp_path / "f.png")
    assert (tmp_path / "f.png").read_bytes()[:8] == PNG
    block = render_block(fp)
    for number in ("65,000 B", "61,000 B", "600 B", "2,100 B", "99 of 100"):
        assert number in block                      # numbers come from the results
    assert "3.5 times smaller" in block             # 2100 / 600
    assert "Not measured" in block and "Nothing here ran on a board" in block


def test_readme_update_is_idempotent_and_replaces_the_old_nodes_only_sentence(tmp_path):
    block = render_block(_fake_report())
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\n| Part | What |\n|---|---|\n| `src/tinyml_lab/plots.py` | figs |\n\n```bash\npython scripts/run_cwru.py   # x\n```\n\n"
                      f"Some text. {OLD_SENTENCE}\n\n## Roadmap\n1. next\n")
    update_readme(readme, block)
    once = readme.read_text()
    update_readme(readme, block)
    assert readme.read_text() == once
    assert once.count(START) == 1 and once.count(END) == 1
    assert OLD_SENTENCE not in once and NEW_SENTENCE in once
    assert once.count("footprint*.py") == 1 and once.count("scripts/measure_footprint.py") == 1
    assert once.index(START) < once.index("## Roadmap") and once.startswith("# T\n")


def test_a_refit_mismatch_is_disclosed_in_the_block():
    fp = _fake_report()
    fp["refit_matches_saved"]["tree_fixed6"] = False
    assert "did not reproduce the saved test accuracy" in render_block(fp)


@needs_arm
def test_a_six_class_forest_links_without_any_libc_call_on_a_core_without_an_fpu():
    """With six classes, zero-initialising the accumulator became a memset call that the libc-free harness cannot link."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1500, 20)).astype(np.float32)
    y = np.digitize(X[:, 0] + 0.3 * X[:, 1], [-1.2, -0.6, 0.0, 0.6, 1.2])         # six classes
    rf = RandomForestClassifier(n_estimators=8, max_depth=5, random_state=0).fit(X, y)
    assert len(rf.classes_) == 6
    src = forest_to_c(rf, "forest_predict")
    assert "float acc[FOREST_N_CLASSES] = {" not in src            # no zero-initialised array, hence no memset
    for target in footprint.TARGETS:
        assert footprint.measure(src, "forest_predict", 20, target, TOOL)["flash_bytes"] > 0
    assert footprint.host_agreement(src, "forest_predict", rf, X[:500])["rate"] >= 0.99
