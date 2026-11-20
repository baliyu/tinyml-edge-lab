"""Tests for the model 2 PC run, on a tiny synthetic dataset (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import numpy as np
import pytest

from litert_quant.har_data import LABELS
from litert_quant.run_model2 import main


def _write(base, split, subjects, labels, rng):
    sig = base / split / "Inertial Signals"
    sig.mkdir(parents=True)
    n = len(labels)
    for axis in "xyz":
        np.savetxt(sig / f"total_acc_{axis}_{split}.txt", rng.normal(size=(n, 128)))
    np.savetxt(base / split / f"y_{split}.txt", labels, fmt="%d")
    np.savetxt(base / split / f"subject_{split}.txt", subjects, fmt="%d")


@pytest.fixture
def tiny_root(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "data" / "UCI HAR Dataset"
    _write(base, "train", [1] * 6 + [2] * 6, [1, 2, 3, 4, 5, 6] * 2, rng)
    _write(base, "test", [3] * 6, [1, 2, 3, 4, 5, 6], rng)
    return tmp_path / "data"


def test_outputs_and_fields(tiny_root, tmp_path):
    out = tmp_path / "out"
    main(root=tiny_root, epochs=1, out_root=out, baseline_path=None)
    assert (out / "models" / "har_cnn_float.tflite").is_file()
    assert (out / "models" / "har_cnn_int8.tflite").is_file()
    data = json.loads((out / "results" / "model2_pc.json").read_text())
    assert data["n_train"] == 12 and data["n_test"] == 6
    assert data["parameters"] == 3334
    assert data["train_subjects"] == [1, 2] and data["test_subjects"] == [3]
    assert data["tree_baseline"] is None
    assert data["accuracy_float_tflite"] == data["correct_float_tflite"] / 6
    assert set(data["per_class_recall_int8"]) == set(LABELS)
    preds = np.load(out / "results" / "model2_pc_predictions.npz")
    for key in ("y_test", "keras", "float", "int8"):
        assert len(preds[key]) == 6


def test_tree_baseline_is_copied(tiny_root, tmp_path):
    base = {"test_accuracy": 0.5, "test_macro_f1": 0.4, "nodes": 61, "leaves": 31,
            "chosen_depth": 6, "n_test": 2947}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(base))
    result = main(root=tiny_root, epochs=1, out_root=tmp_path / "out", baseline_path=path)
    assert result["tree_baseline"] == base


def test_two_runs_give_identical_files(tiny_root, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(root=tiny_root, epochs=1, out_root=a, baseline_path=None)
    main(root=tiny_root, epochs=1, out_root=b, baseline_path=None)
    for name in ("har_cnn_float.tflite", "har_cnn_int8.tflite"):
        assert (a / "models" / name).read_bytes() == (b / "models" / name).read_bytes()
    pa = np.load(a / "results" / "model2_pc_predictions.npz")
    pb = np.load(b / "results" / "model2_pc_predictions.npz")
    assert all(np.array_equal(pa[k], pb[k]) for k in ("y_test", "keras", "float", "int8"))
