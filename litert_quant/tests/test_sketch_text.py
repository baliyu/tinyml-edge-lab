from litert_quant.firmware_data import ROOT

SKETCH = ROOT / "firmware" / "digits_bench" / "digits_bench.ino"


def test_sketch_includes_headers_and_variants():
    text = SKETCH.read_text()
    for token in [
        '#include "test_data.h"',
        '#include "model_int8.h"',
        '#include "model_float.h"',
        "USE_INT8",
        "USE_FLOAT",
    ]:
        assert token in text, token


def test_sketch_prints_the_documented_lines():
    text = SKETCH.read_text()
    for token in ['"START,"', '"S,"', '"DONE,"', '"ERROR,']:
        assert token in text, token


def test_sketch_is_marked_ai_assisted():
    assert "AI-assisted" in SKETCH.read_text()
