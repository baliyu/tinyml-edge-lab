"""Tests for the CWRU net flash calculation, on small synthetic compiler output.

AI-assisted: drafted with Claude, checked by the author.
"""
import json

from litert_quant.cwru_flash import measure


def _write(tmp_path, kind, inp, model, total, inp_size=1000, model_size=50):
    (tmp_path / ("model3_flash_symbols_%s.txt" % kind)).write_text(
        "1 316 r test_labels\n2 %d r %s\n3 %d r %s\n4 8 b other\n" % (inp_size, inp, model_size, model)
    )
    (tmp_path / ("compile_cwru_%s.txt" % kind)).write_text("Sketch uses %d bytes (1%%) of x.\n" % total)


def test_measure_subtracts_test_arrays_and_splits_model(tmp_path):
    _write(tmp_path, "float", "test_input_float", "g_model_float", 300000, 4000, 200)
    _write(tmp_path, "int8", "test_input_int8", "g_model_int8", 180000, 1000, 80)
    out = measure(tmp_path)
    assert out["float"]["test_data_bytes"] == 4316 and out["float"]["net_firmware_bytes"] == 300000 - 4316
    assert out["int8"]["test_data_bytes"] == 1316 and out["int8"]["model_bytes"] == 80
    assert out["float"]["runtime_and_code_bytes"] == 300000 - 4316 - 200
    assert out["net_difference_float_minus_int8"] == (300000 - 4316) - (180000 - 1316)
    assert out["model_difference_float_minus_int8"] == 120
    assert out["runtime_and_code_difference_float_minus_int8"] == out["net_difference_float_minus_int8"] - 120


def test_measure_writes_json(tmp_path):
    _write(tmp_path, "float", "test_input_float", "g_model_float", 300000)
    _write(tmp_path, "int8", "test_input_int8", "g_model_int8", 180000)
    out = measure(tmp_path)
    assert json.loads((tmp_path / "model3_flash.json").read_text()) == out
