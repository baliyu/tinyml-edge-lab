import numpy as np
import pytest
import tensorflow as tf

from litert_quant.dense_diagnosis import (
    load_inputs,
    main,
    parse_logits,
    per_class_matches,
    recompute,
    requantise,
    run_dense,
)
from litert_quant.model1 import build_cnn, to_int8_tflite


def test_parse_logits_reads_l_lines():
    text = "START,int8,2\nS,0,1,1,5\nL,0,1,2,3,4,5,6,7,8,9,10\nL,1,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10\nDONE\n"
    got = parse_logits(text)
    assert got.shape == (2, 10)
    assert got[1][0] == -1


def test_parse_logits_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_logits("L,0,1,2,3\n")
    with pytest.raises(ValueError):
        parse_logits("S,0,1,1,5\n")


def test_load_inputs_reads_header():
    values = list(range(-128, 128)) * 90
    values = values[: 360 * 64]
    header = "const int8_t test_input_int8[23040] = {" + ",".join(str(v) for v in values) + "};"
    x = load_inputs(header)
    assert x.shape == (360, 8, 8, 1)
    assert int(x[0, 0, 0, 0]) == -128


def test_per_class_matches_counts_columns():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 0], [3, 0], [0, 6]])
    assert per_class_matches(a, b) == [2, 1]


def test_requantise_rounds_half_away_and_clips():
    got = requantise(np.array([5, -5, 1000, -1000]), 0.5, 0)
    assert got.tolist() == [3, -3, 127, -128]


def test_recompute_matches_interpreter_on_a_small_model():
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x_rep = rng.random((20, 8, 8, 1)).astype(np.float32)
    blob = to_int8_tflite(build_cnn(), x_rep)
    x = rng.integers(-128, 128, size=(20, 8, 8, 1)).astype(np.int8)
    parts = run_dense(blob, x)
    own, first = recompute(parts)
    assert len(parts["sw"]) == 1
    assert (own == parts["pc"]).mean() > 0.9
    assert (first == own).all()


def test_main_runs_end_to_end(tmp_path):
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(1)
    x_rep = rng.random((20, 8, 8, 1)).astype(np.float32)
    blob = to_int8_tflite(build_cnn(), x_rep)
    x = rng.integers(-128, 128, size=(360, 8, 8, 1)).astype(np.int8)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "firmware" / "digits_bench").mkdir(parents=True)
    for name in ("digits_cnn_int8_perchannel_dense.tflite", "digits_cnn_int8.tflite"):
        (tmp_path / "models" / name).write_bytes(blob)
    header = "const int8_t test_input_int8[23040] = {" + ",".join(str(int(v)) for v in x.flatten()) + "};"
    (tmp_path / "firmware" / "digits_bench" / "test_data.h").write_text(header)
    pc = run_dense(blob, x)["pc"]
    log = "\n".join("L,%d,%s" % (i, ",".join(str(int(v)) for v in pc[i])) for i in range(360)) + "\n"
    (tmp_path / "results" / "board_int8_logits_run1.txt").write_text(log)
    (tmp_path / "results" / "board_int8_fixed_run1.txt").write_text(log)
    assert main(root=tmp_path) == 0
    assert (tmp_path / "results" / "dense_scale_diagnosis.json").exists()
    assert (tmp_path / "results" / "figures" / "dense_scale_diagnosis.png").exists()
