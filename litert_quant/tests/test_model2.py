"""Tests for model 2 (HAR 1D-CNN), on random data; no dataset needed.

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from litert_quant.model2 import (
    build_cnn,
    representative_set,
    to_float_tflite,
    to_int8_tflite,
    to_model_input,
    train,
)

ALLOWED_OPS = {"CONV_2D", "RESHAPE", "FULLY_CONNECTED"}


def _x(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 128, 1, 3)).astype(np.float32)


def _interp(blob):
    it = Interpreter(
        model_content=blob,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    return it


def test_parameter_count():
    assert build_cnn().count_params() == 3334


def test_output_shape():
    out = build_cnn()(np.zeros((2, 128, 1, 3), dtype=np.float32))
    assert tuple(out.shape) == (2, 6)


def test_to_model_input_adds_the_width_axis():
    x = np.arange(2 * 128 * 3, dtype=np.float32).reshape(2, 128, 3)
    y = to_model_input(x)
    assert y.shape == (2, 128, 1, 3)
    assert np.array_equal(y[:, :, 0, :], x)


def test_representative_set_is_a_deterministic_subset():
    x = _x(50)
    a = representative_set(x, n=20, seed=0)
    b = representative_set(x, n=20, seed=0)
    assert a.shape == (20, 128, 1, 3)
    assert np.array_equal(a, b)
    assert all(any(np.array_equal(r, row) for row in x) for r in a)
    assert representative_set(x, n=500).shape[0] == 50


def test_exported_ops_are_only_conv_reshape_fc():
    x = _x(40)
    model = build_cnn()
    for blob in (to_float_tflite(model), to_int8_tflite(model, x)):
        ops = {o["op_name"] for o in _interp(blob)._get_ops_details()}
        assert ops <= ALLOWED_OPS, ops


def test_dense_weights_are_per_tensor_in_int8_model():
    it = _interp(to_int8_tflite(build_cnn(), _x(40)))
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"]
    assert len(fc) == 1
    weights = int(fc[0]["inputs"][1])
    details = {t["index"]: t for t in it.get_tensor_details()}
    assert len(details[weights]["quantization_parameters"]["scales"]) == 1


def test_int8_input_and_output_are_int8():
    it = _interp(to_int8_tflite(build_cnn(), _x(40)))
    assert it.get_input_details()[0]["dtype"] == np.int8
    assert it.get_output_details()[0]["dtype"] == np.int8


def test_training_with_the_same_seed_is_reproducible():
    x = _x(64)
    y = np.random.default_rng(1).integers(0, 6, 64)
    w1 = train(x, y, epochs=1, seed=0).get_weights()
    w2 = train(x, y, epochs=1, seed=0).get_weights()
    assert all(np.array_equal(a, b) for a, b in zip(w1, w2))


def test_export_bytes_do_not_depend_on_earlier_models():
    x = _x(40)
    model = build_cnn()
    first = (to_float_tflite(model), to_int8_tflite(model, x))
    build_cnn()
    build_cnn()
    second = (to_float_tflite(model), to_int8_tflite(model, x))
    assert first[0] == second[0]
    assert first[1] == second[1]
