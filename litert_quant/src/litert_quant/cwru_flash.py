"""Net firmware flash of the two CWRU builds, no hardware needed.

AI-assisted: drafted with Claude, checked by the author.
Same method as har_flash: the sketch carries the test subset as const arrays, so the
arduino-cli total mostly measures test data. The net figure subtracts the test arrays
(inputs and labels), read from demangled nm symbol sizes. The remainder is split into
the model and the rest (TFLM runtime, kernels, Arduino core and sketch code).
"""
import json
from pathlib import Path

from litert_quant.har_flash import net_flash, sketch_bytes, symbol_sizes

ROOT = Path(__file__).resolve().parents[2]


def measure(results_dir=ROOT / "results"):
    results_dir = Path(results_dir)
    out = {"note": "net = sketch bytes minus the test arrays; sketch bytes come from the arduino-cli log"}
    for kind in ("float", "int8"):
        sizes = symbol_sizes((results_dir / ("model3_flash_symbols_%s.txt" % kind)).read_text())
        total = sketch_bytes((results_dir / ("compile_cwru_%s.txt" % kind)).read_text())
        out[kind] = net_flash(kind, total, sizes)
    out["net_difference_float_minus_int8"] = out["float"]["net_firmware_bytes"] - out["int8"]["net_firmware_bytes"]
    out["model_difference_float_minus_int8"] = out["float"]["model_bytes"] - out["int8"]["model_bytes"]
    out["runtime_and_code_difference_float_minus_int8"] = (
        out["float"]["runtime_and_code_bytes"] - out["int8"]["runtime_and_code_bytes"]
    )
    (results_dir / "model3_flash.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
