"""Tests for the HAR firmware data generator, on tiny synthetic data (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import json
import re

import numpy as np
import pytest

from litert_quant.firmware_data import quantise_input
from litert_quant.har_data import load_har
from litert_quant.har_firmware_data import build, c_float_array, select_subset
from litert_quant.model2 import (
    build_cnn,
    input_quantization,
    predict_tflite,
    to_float_tflite,
    to_int8_tflite,
    to_model_input,
)

FLOAT_RE = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
INT_RE = r"-?\d+"


def _write(base, split, subjects, labels, rng):
    sig = base / split / "Inertial Signals"
    sig.mkdir(parents=True)
    n = len(labels)
    for axis in "xyz":
        np.savetxt(sig / f"total_acc_{axis}_{split}.txt", rng.normal(size=(n, 128)))
    np.savetxt(base / split / f"y_{split}.txt", labels, fmt="%d")
    np.savetxt(base / split / f"subject_{split}.txt", subjects, fmt="%d")


def _numbers(text, pattern):
    body = text.split("{", 1)[1].split("}", 1)[0]
    return re.findall(pattern, body)


def test_subset_is_stratified_sorted_and_seeded():
    rng = np.random.default_rng(0)
    y = rng.permutation(np.repeat(np.arange(6), 100))
    idx = select_subset(y, 10, 0)
    assert len(idx) == 60 and len(set(idx.tolist())) == 60
    assert np.array_equal(idx, np.sort(idx))
    assert np.bincount(y[idx], minlength=6).tolist() == [10] * 6
    assert np.array_equal(idx, select_subset(y, 10, 0))
    assert not np.array_equal(idx, select_subset(y, 10, 1))


def test_subset_too_small_raises():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 5)
    with pytest.raises(ValueError):
        select_subset(y, 10, 0)


def test_float_array_round_trips_exactly():
    x = (np.random.default_rng(2).normal(size=200) * 3).astype(np.float32)
    text = c_float_array("a", x)
    back = np.array([float(t) for t in _numbers(text, FLOAT_RE)], dtype=np.float32)
    assert np.array_equal(back, x)
    assert "a_len = 200" in text


def test_build_writes_consistent_headers(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    base = root / "UCI HAR Dataset"
    _write(base, "train", [1] * 6 + [2] * 6, [1, 2, 3, 4, 5, 6] * 2, rng)
    _write(base, "test", [3] * 18, [1, 2, 3, 4, 5, 6] * 3, rng)
    x_train, x_test, _, y_test, _, _ = load_har(root)
    xtr, xte = to_model_input(x_train), to_model_input(x_test)
    model = build_cnn()
    fblob = to_float_tflite(model)
    qblob = to_int8_tflite(model, xtr)
    models = tmp_path / "models"
    models.mkdir()
    (models / "har_cnn_float.tflite").write_bytes(fblob)
    (models / "har_cnn_int8.tflite").write_bytes(qblob)
    f_pred = predict_tflite(fblob, xte)
    q_pred = predict_tflite(qblob, xte)
    np.savez(tmp_path / "pred.npz", y_test=y_test, keras=f_pred, float=f_pred, int8=q_pred)

    out = tmp_path / "fw"
    report = build(models, out, tmp_path / "pred.npz", tmp_path / "subset.json", root=root, per_class=2, seed=0)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["model_float.h", "model_int8.h", "test_input_float.h", "test_input_int8.h", "test_meta.h"]
    idx = np.array(report["indices"])
    assert len(idx) == 12
    meta = (out / "test_meta.h").read_text()
    assert "#define TEST_COUNT 12" in meta and "#define TEST_SIZE 384" in meta
    scale, zero = input_quantization(qblob)
    assert "#define INPUT_ZERO_POINT %d" % zero in meta
    q = quantise_input(x_test[idx], scale, zero).ravel()
    got_q = np.array([int(t) for t in _numbers((out / "test_input_int8.h").read_text(), INT_RE)])
    assert np.array_equal(got_q, q.astype(int))
    got_f = np.array([float(t) for t in _numbers((out / "test_input_float.h").read_text(), FLOAT_RE)], dtype=np.float32)
    assert np.array_equal(got_f, x_test[idx].ravel())
    labels = [int(t) for t in _numbers(meta, INT_RE)]
    assert labels == y_test[idx].tolist()
    assert report["pc_correct"]["float"] == int(np.sum(f_pred[idx] == y_test[idx]))
    assert json.loads((tmp_path / "subset.json").read_text()) == report
