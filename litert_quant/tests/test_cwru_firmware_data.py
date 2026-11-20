"""Tests for the CWRU firmware data generator, on tiny synthetic data (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import hashlib
import json
import re

import numpy as np
import pytest
from scipy.io import savemat

from litert_quant.cwru_data import Recording, make_split
from litert_quant.cwru_firmware_data import build, select_subset
from litert_quant.firmware_data import quantise_input
from litert_quant.model1 import input_quantization
from litert_quant.model3 import preprocess
from litert_quant.run_model3 import main as run_main

FLOAT_RE = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
INT_RE = r"-?\d+"
RECS = [
    Recording(1, "normal", 0, 0), Recording(2, "normal", 0, 1), Recording(3, "normal", 0, 2),
    Recording(4, "normal", 0, 3), Recording(5, "IR", 7, 0), Recording(6, "B", 14, 2),
    Recording(7, "OR", 21, 3),
]


def _write(root):
    rng = np.random.default_rng(0)
    for r in RECS:
        amp = 0.06 if r.kind == "normal" else 0.5
        x = (rng.normal(size=4 * 2048) * amp).reshape(-1, 1)
        savemat(str(root / f"{r.num}.mat"), {f"X{r.num:03d}_DE_time": x})


def _numbers(text, pattern):
    body = text.split("{", 1)[1].split("}", 1)[0]
    return re.findall(pattern, body)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cwru_fw")
    root = tmp / "raw"
    root.mkdir()
    _write(root)
    out = tmp / "out"
    run_main(root=root, recordings=RECS, epochs=3, out_root=out)
    fw = tmp / "fw"
    paths = (out / "models", fw, out / "results" / "model3_pc_scores.npz", out / "results" / "model3_pc.json", tmp / "subset.json")
    rep = build(*paths, root=root, recordings=RECS, normal_n=8, per_group=5)
    return root, out, fw, rep, paths


def test_subset_is_stratified_sorted_and_seeded():
    y = np.array([0] * 100 + [1] * 90)
    kind = np.array(["normal"] * 100 + ["B"] * 30 + ["IR"] * 30 + ["OR"] * 30)
    size = np.array([0] * 100 + [14] * 30 + [7] * 30 + [21] * 30)
    idx = select_subset(y, kind, size, 10, 5, 0)
    assert len(idx) == 25 and len(set(idx.tolist())) == 25 and np.array_equal(idx, np.sort(idx))
    assert int(np.sum(y[idx] == 0)) == 10
    for k in ("B", "IR", "OR"):
        assert int(np.sum(kind[idx] == k)) == 5
    assert np.array_equal(idx, select_subset(y, kind, size, 10, 5, 0))
    assert not np.array_equal(idx, select_subset(y, kind, size, 10, 5, 1))


def test_subset_too_small_raises():
    y = np.array([0] * 5 + [1] * 6)
    kind = np.array(["normal"] * 5 + ["B"] * 6)
    size = np.array([0] * 5 + [7] * 6)
    with pytest.raises(ValueError):
        select_subset(y, kind, size, 10, 5, 0)
    with pytest.raises(ValueError):
        select_subset(y, kind, size, 5, 10, 0)


def test_build_subset_counts(built):
    _, _, _, rep, _ = built
    assert rep["n"] == 23 and rep["n_normal"] == 8 and rep["n_fault"] == 15
    assert len(rep["pc_float_scores"]) == 23 and len(rep["pc_int8_scores"]) == 23


def test_int8_and_float_headers_match_the_preprocessed_inputs(built):
    root, out, fw, rep, paths = built
    r = json.loads(paths[3].read_text())
    s = make_split(root, RECS)
    xs = preprocess(s.x_test[rep["indices"]], r["scaler_std"], r["scaler_clip"])
    scale, zero = input_quantization((paths[0] / "cwru_ae_int8.tflite").read_bytes())
    want_q = quantise_input(xs, scale, zero).reshape(-1)
    got_q = np.array([int(t) for t in _numbers((fw / "test_input_int8.h").read_text(), INT_RE)])
    assert np.array_equal(got_q, want_q)
    got_f = np.array([float(t) for t in _numbers((fw / "test_input_float.h").read_text(), FLOAT_RE)], dtype=np.float32)
    assert np.array_equal(got_f, xs.reshape(-1))


def test_meta_header_has_counts_thresholds_and_labels(built):
    _, _, fw, rep, paths = built
    text = (fw / "test_meta.h").read_text()
    assert "#define TEST_COUNT 23" in text and "#define TEST_SIZE 128" in text
    r = json.loads(paths[3].read_text())
    for name, key in (("FLOAT", "float"), ("INT8", "int8")):
        m = re.search(r"#define SCORE_THRESHOLD_%s (\S+)f\n" % name, text)
        assert m and abs(float(m.group(1)) - r["scores"][key]["operating_point"]["threshold"]) < 1e-6 * max(1.0, abs(float(m.group(1))))
    labels = re.search(r"test_labels[^{]*\{([^}]*)\}", text).group(1)
    nums = [int(t) for t in re.findall(INT_RE, labels)]
    assert len(nums) == 23 and sum(nums) == 15


def test_model_headers_hold_the_tflite_bytes_and_hashes_match(built):
    _, _, fw, rep, paths = built
    for kind in ("float", "int8"):
        blob = (paths[0] / ("cwru_ae_%s.tflite" % kind)).read_bytes()
        text = (fw / ("model_%s.h" % kind)).read_text()
        assert int(re.search(r"g_model_%s_len\D+(\d+)" % kind, text).group(1)) == len(blob)
    for name, digest in rep["header_sha256"].items():
        assert hashlib.sha256((fw / name).read_bytes()).hexdigest() == digest


def test_scores_from_a_different_test_set_raise(built, tmp_path):
    root, out, fw, rep, paths = built
    d = dict(np.load(paths[2]))
    d["y_test"] = d["y_test"][::-1].copy()
    bad = tmp_path / "bad.npz"
    np.savez(bad, **d)
    with pytest.raises(ValueError):
        build(paths[0], tmp_path / "fw2", bad, paths[3], tmp_path / "r.json", root=root, recordings=RECS, normal_n=8, per_group=5)
