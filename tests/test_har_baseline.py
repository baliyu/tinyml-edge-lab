import json

from tinyml_lab.har_baseline import run


def test_pipeline_end_to_end_on_synthetic_data(fake_uci_root, tmp_path):
    out = tmp_path / "results"
    r = run(fake_uci_root, depths=(2, 4), out_dir=out, chosen_depth=4)
    # synthetic classes differ in frequency and amplitude, so a small tree should separate them
    assert r["test_accuracy"] > 0.9
    assert set(r["train_subjects"]).isdisjoint(r["test_subjects"])
    saved = json.loads((out / "har_baseline.json").read_text())
    assert saved["test_accuracy"] == r["test_accuracy"]
    assert "int har_tree_predict(const float *x)" in (out / "har_tree.c").read_text()
    assert (out / "depth_sweep.md").read_text().startswith("| max depth")
