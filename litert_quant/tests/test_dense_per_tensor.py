"""The exported int8 model must have per-tensor dense weights.

AI-assisted: drafted with Claude, checked by the author.
Measured on the Nano 33 BLE: with per-channel dense weights the board applied
one output scale to all outputs and gave wrong results. The check loads the
model without the default XNNPACK delegate, which would show a rewritten graph.
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from litert_quant.model1 import build_cnn, to_int8_tflite


def test_dense_weights_are_per_tensor_in_int8_model():
    rng = np.random.default_rng(0)
    x_rep = rng.random((20, 8, 8, 1)).astype(np.float32)
    blob = to_int8_tflite(build_cnn(), x_rep)
    it = Interpreter(
        model_content=blob,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"]
    assert len(fc) == 1
    weights = int(fc[0]["inputs"][1])
    details = {t["index"]: t for t in it.get_tensor_details()}
    assert len(details[weights]["quantization_parameters"]["scales"]) == 1
