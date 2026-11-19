import json

import pytest

from litert_quant.data import fingerprint, load_split
from litert_quant.run_model1 import main


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("model1")
    result = main(epochs=2, out_root=out)
    return out, result


def test_files_and_json_are_written(run):
    out, result = run
    float_file = out / "models" / "digits_cnn_float.tflite"
    int8_file = out / "models" / "digits_cnn_int8.tflite"
    assert float_file.stat().st_size == result["float_bytes"]
    assert int8_file.stat().st_size == result["int8_bytes"]
    assert (out / "results" / "model1_pc_predictions.npz").exists()
    saved = json.loads((out / "results" / "model1_pc.json").read_text())
    assert saved == result


def test_result_matches_the_fixed_split(run):
    _, result = run
    _, x_test, _, y_test = load_split()
    assert result["test_fingerprint"] == fingerprint(x_test, y_test)
    assert result["n_test"] == 360
    assert 0.0 <= result["accuracy_int8_tflite"] <= 1.0
    assert result["int8_bytes"] < result["float_bytes"]
