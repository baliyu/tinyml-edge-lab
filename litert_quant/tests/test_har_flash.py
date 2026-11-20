"""Tests for the net flash calculation, on small synthetic compiler output.

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import pytest

from litert_quant.har_flash import measure, net_flash, sketch_bytes, symbol_sizes

NM = """0000100000 0000000360 r test_labels
0000100360 0000552960 r test_input_float
0000700000 0000016356 r g_model_float
0000800000 0000000008 b something_else
"""


def test_symbol_sizes_reads_decimal_nm_lines():
    s = symbol_sizes(NM)
    assert s["test_labels"] == 360 and s["test_input_float"] == 552960 and s["g_model_float"] == 16356


def test_sketch_bytes_parses_the_log_and_rejects_others():
    assert sketch_bytes("x\nSketch uses 710432 bytes (72%) of program storage space.\n") == 710432
    with pytest.raises(ValueError):
        sketch_bytes("nothing here")


def test_net_flash_subtracts_test_arrays_and_splits_model():
    r = net_flash("float", 710432, symbol_sizes(NM))
    assert r["test_data_bytes"] == 553320 and r["net_firmware_bytes"] == 157112
    assert r["model_bytes"] == 16356 and r["runtime_and_code_bytes"] == 157112 - 16356


def test_measure_writes_json(tmp_path):
    for kind, inp, model, total in (("float", "test_input_float", "g_model_float", 700000), ("int8", "test_input_int8", "g_model_int8", 300000)):
        (tmp_path / ("model2_flash_symbols_%s.txt" % kind)).write_text(
            "1 360 r test_labels\n2 1000 r %s\n3 50 r %s\n" % (inp, model)
        )
        (tmp_path / ("compile_har_%s.txt" % kind)).write_text("Sketch uses %d bytes (1%%) of x.\n" % total)
    out = measure(tmp_path)
    assert out["float"]["net_firmware_bytes"] == 700000 - 1360 and out["int8"]["model_bytes"] == 50
    assert out["net_difference_float_minus_int8"] == (700000 - 1360) - (300000 - 1360)
    assert json.loads((tmp_path / "model2_flash.json").read_text()) == out
