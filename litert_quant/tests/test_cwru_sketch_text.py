"""Text checks on the CWRU sketch (it cannot be compiled inside pytest).

AI-assisted: drafted with Claude, checked by the author.
"""
from pathlib import Path

INO = Path(__file__).resolve().parents[1] / "firmware" / "cwru_bench" / "cwru_bench.ino"


def _text():
    return INO.read_text()


def test_variants_switch_on_flags_and_unset_is_an_error():
    t = _text()
    assert "#if defined(USE_INT8)" in t and "#elif defined(USE_FLOAT)" in t
    assert '#error "Define USE_INT8 or USE_FLOAT"' in t
    assert t.index('#include "test_input_int8.h"') < t.index("#elif defined(USE_FLOAT)")
    assert t.index('#include "test_input_float.h"') > t.index("#elif defined(USE_FLOAT)")
    assert '#include "test_meta.h"' in t


def test_score_uses_header_input_and_each_variant_has_its_own_threshold():
    t = _text()
    assert "score_window(int base)" in t
    assert "SCORE_THRESHOLD_INT8" in t and "SCORE_THRESHOLD_FLOAT" in t
    assert "input->data.f[j] -" not in t and "input->data.int8[j] -" not in t
    assert "Serial.print(score, 6)" in t
    assert "k < 6" not in t and "test_pixels" not in t


def test_protocol_ops_and_no_em_dash():
    t = _text()
    for tag in ('"START,"', '"S,"', '"DONE,"', "ERROR,invoke", "ERROR,allocate", "ERROR,schema"):
        assert tag in t
    assert "MicroMutableOpResolver<1>" in t
    assert t.count("AddFullyConnected()") == 1
    for op in ("AddConv2D", "AddReshape", "AddQuantize", "AddDequantize"):
        assert op not in t
    assert "kArenaSize = 8 * 1024" in t
    assert "\u2014" not in t
