import numpy as np
from ai_edge_litert.interpreter import Interpreter

from litert_quant.data import load_split
from litert_quant.firmware_data import (
    MODEL_FILES,
    ROOT,
    c_array,
    make_headers,
    quantise_input,
)
from litert_quant.model1 import input_quantization


def test_c_array_format():
    text = c_array("uint8_t", "a", [1, 2, 3], per_line=2)
    assert "const uint8_t a[] = {" in text
    assert "const unsigned int a_len = 3;" in text
    assert "1, 2," in text
    assert "3," in text


def test_aligned_array_has_alignas():
    assert c_array("unsigned char", "m", [0], align=16).startswith("alignas(16) ")


def test_int8_inputs_reproduce_the_pc_predictions():
    blob = (ROOT / "models" / MODEL_FILES["int8"]).read_bytes()
    scale, zero = input_quantization(blob)
    _, x_test, _, _ = load_split()
    q = quantise_input(x_test, scale, zero)
    interp = Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for i in range(len(q)):
        interp.set_tensor(inp["index"], q[i : i + 1])
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out["index"])[0])))
    saved = np.load(ROOT / "results" / "model1_pc_predictions.npz")["int8"]
    assert np.array_equal(np.array(preds), saved)


def test_committed_headers_are_up_to_date(tmp_path):
    names = make_headers(ROOT / "models", tmp_path)
    for name in names:
        fresh = (tmp_path / name).read_text()
        committed = (ROOT / "firmware" / "digits_bench" / name).read_text()
        assert fresh == committed, name
