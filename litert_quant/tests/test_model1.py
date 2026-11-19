import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter

from litert_quant.data import load_split
from litert_quant.model1 import (
    accuracy,
    input_quantization,
    predict_tflite,
    to_float_tflite,
    to_int8_tflite,
    train,
)


@pytest.fixture(scope="module")
def trained():
    x_train, x_test, y_train, y_test = load_split()
    model = train(x_train, y_train, epochs=5)
    float_blob = to_float_tflite(model)
    int8_blob = to_int8_tflite(model, x_train[:100])
    return model, float_blob, int8_blob, x_test, y_test


def test_float_tflite_matches_keras(trained):
    model, float_blob, _, x_test, _ = trained
    keras_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    lite_pred = predict_tflite(float_blob, x_test)
    assert np.mean(keras_pred == lite_pred) >= 0.99


def test_int8_model_has_int8_input_and_output(trained):
    _, _, int8_blob, _, _ = trained
    interp = Interpreter(model_content=int8_blob)
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.int8
    assert interp.get_output_details()[0]["dtype"] == np.int8
    scale, zero = input_quantization(int8_blob)
    assert scale > 0
    assert -128 <= zero <= 127


def test_int8_runs_and_is_above_chance(trained):
    _, _, int8_blob, x_test, y_test = trained
    assert accuracy(predict_tflite(int8_blob, x_test), y_test) > 0.3


def test_int8_file_is_smaller_than_float(trained):
    _, float_blob, int8_blob, _, _ = trained
    assert len(int8_blob) < len(float_blob)


def test_accuracy_function():
    assert accuracy([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5
