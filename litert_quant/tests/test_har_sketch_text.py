"""Text checks on the HAR sketch (it cannot be compiled inside pytest).

AI-assisted: drafted with Claude, checked by the author.
"""
from pathlib import Path

INO = Path(__file__).resolve().parents[1] / "firmware" / "har_bench" / "har_bench.ino"


def _text():
    return INO.read_text()


def test_variants_switch_on_flags_and_unset_is_an_error():
    t = _text()
    assert "#if defined(USE_INT8)" in t and "#elif defined(USE_FLOAT)" in t
    assert '#error "Define USE_INT8 or USE_FLOAT"' in t
    assert t.index('#include "test_input_int8.h"') < t.index("#elif defined(USE_FLOAT)")
    assert t.index('#include "test_input_float.h"') > t.index("#elif defined(USE_FLOAT)")


def test_six_outputs_and_no_digits_leftovers():
    t = _text()
    assert "k < 6" in t and "k < 10" not in t
    assert "test_pixels" not in t and "16.0f" not in t and "test_data.h" not in t
    assert '#include "test_meta.h"' in t


def test_protocol_ops_and_no_em_dash():
    t = _text()
    for tag in ('"START,"', '"S,"', '"L,"', '"DONE,"', "ERROR,invoke", "ERROR,allocate"):
        assert tag in t
    assert "MicroMutableOpResolver<3>" in t
    for op in ("AddConv2D", "AddReshape", "AddFullyConnected"):
        assert t.count(op + "()") == 1
    assert "kArenaSize = 32 * 1024" in t
    assert "\u2014" not in t
