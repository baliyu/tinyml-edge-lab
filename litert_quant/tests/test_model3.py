"""Tests for model 3 (CWRU autoencoder), on synthetic sine windows; no dataset needed.

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from litert_quant.model3 import (
    build_autoencoder,
    fit_scaler,
    mse_scores,
    predict_scores,
    preprocess,
    representative_set,
    to_float_tflite,
    to_int8_tflite,
    train,
)

ALLOWED_OPS = {"FULLY_CONNECTED", "RESHAPE"}


def _sines(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 128.0
    amp = rng.uniform(0.5, 1.0, size=(n, 1))
    freq = rng.choice([2, 3], size=(n, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    x = amp * np.sin(2 * np.pi * freq * t + phase) + 0.01 * rng.normal(size=(n, 128))
    return x.astype(np.float32)


def _interp(blob):
    it = Interpreter(
        model_content=blob,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    return it


@pytest.fixture(scope="module")
def trained():
    x = _sines(256, 0)
    model = train(x, epochs=100, seed=0)
    return x, model


def test_parameter_count():
    assert build_autoencoder().count_params() == 4520


def test_output_shape():
    out = build_autoencoder()(np.zeros((2, 128), dtype=np.float32))
    assert tuple(out.shape) == (2, 128)


def test_fit_scaler_and_preprocess():
    x = np.random.default_rng(0).normal(size=(50, 128)).astype(np.float32) * 0.1
    std, clip = fit_scaler(x)
    assert abs(std - float(np.std(x))) < 1e-6
    assert clip == float(int(clip)) and clip >= np.max(np.abs(x)) / std
    p = preprocess(x, std, clip)
    assert p.dtype == np.float32 and abs(float(p.std()) - 1.0) < 1e-3
    big = preprocess(np.full((1, 128), 1e3, dtype=np.float32), std, clip)
    assert float(big.max()) == clip


def test_representative_set_is_a_deterministic_subset():
    x = _sines(50, 1)
    a = representative_set(x, n=20, seed=0)
    assert a.shape == (20, 128) and np.array_equal(a, representative_set(x, n=20, seed=0))


def test_exported_ops_are_only_fully_connected():
    x = _sines(40, 2)
    model = build_autoencoder()
    for blob in (to_float_tflite(model), to_int8_tflite(model, x)):
        ops = {o["op_name"] for o in _interp(blob)._get_ops_details()}
        assert ops <= ALLOWED_OPS, ops


def test_dense_weights_are_per_tensor_in_int8_model():
    it = _interp(to_int8_tflite(build_autoencoder(), _sines(40, 3)))
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"]
    assert len(fc) == 4
    details = {t["index"]: t for t in it.get_tensor_details()}
    for op in fc:
        assert len(details[int(op["inputs"][1])]["quantization_parameters"]["scales"]) == 1


def test_int8_input_and_output_are_int8():
    it = _interp(to_int8_tflite(build_autoencoder(), _sines(40, 4)))
    assert it.get_input_details()[0]["dtype"] == np.int8
    assert it.get_output_details()[0]["dtype"] == np.int8


def test_training_with_the_same_seed_is_reproducible():
    x = _sines(64, 5)
    w1 = train(x, epochs=1, seed=0).get_weights()
    w2 = train(x, epochs=1, seed=0).get_weights()
    assert all(np.array_equal(a, b) for a, b in zip(w1, w2))


def test_training_lowers_the_error(trained):
    x, model = trained
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)
    untrained = build_autoencoder()
    before = mse_scores(x, untrained.predict(x, verbose=0)).mean()
    after = mse_scores(x, model.predict(x, verbose=0)).mean()
    assert after < 0.6 * before


def test_float_tflite_scores_match_keras(trained):
    x, model = trained
    xt = _sines(30, 6)
    keras_scores = mse_scores(xt, model.predict(xt, verbose=0))
    blob = to_float_tflite(model)
    np.testing.assert_allclose(predict_scores(blob, xt), keras_scores, rtol=1e-3, atol=1e-6)


def test_int8_scores_track_float_and_separate_anomalies(trained):
    x, model = trained
    xn = _sines(60, 7)
    xa = np.random.default_rng(8).uniform(-1, 1, size=(30, 128)).astype(np.float32)
    f_blob = to_float_tflite(model)
    q_blob = to_int8_tflite(model, representative_set(x, 200, 0))
    fn, fa = predict_scores(f_blob, xn), predict_scores(f_blob, xa)
    qn, qa = predict_scores(q_blob, xn), predict_scores(q_blob, xa)
    assert np.median(fa) > 3 * np.median(fn)
    assert np.median(qa) > 3 * np.median(qn)
    assert np.corrcoef(np.concatenate([fn, fa]), np.concatenate([qn, qa]))[0, 1] > 0.9


def test_export_bytes_do_not_depend_on_earlier_models():
    x = _sines(40, 9)
    model = build_autoencoder()
    first = (to_float_tflite(model), to_int8_tflite(model, x))
    build_autoencoder()
    build_autoencoder()
    second = (to_float_tflite(model), to_int8_tflite(model, x))
    assert first[0] == second[0] and first[1] == second[1]
