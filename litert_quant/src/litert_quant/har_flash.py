"""Net firmware flash from compiler output, no hardware needed.

AI-assisted: drafted with Claude, checked by the author.
The sketch carries a test set as const arrays, so the arduino-cli totals mostly
measure test data. The net figure subtracts the test arrays (inputs and labels),
read from demangled nm symbol sizes (nm -C -S -t d). The remainder is split into
the model and the rest (TFLM runtime, kernels, Arduino core and sketch code).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_SYMBOLS = {"float": ("test_input_float", "test_labels"), "int8": ("test_input_int8", "test_labels")}
MODEL_SYMBOL = {"float": "g_model_float", "int8": "g_model_int8"}


def symbol_sizes(text):
    """Map symbol name to size from lines of nm -C -S -t d: address size type name."""
    sizes = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            sizes[parts[3]] = int(parts[1])
    return sizes


def sketch_bytes(log_text):
    m = re.search(r"Sketch uses (\d+) bytes", log_text)
    if not m:
        raise ValueError("no 'Sketch uses' line in the compile log")
    return int(m.group(1))


def net_flash(kind, total, sizes):
    data = sum(sizes[n] for n in TEST_SYMBOLS[kind])
    model = sizes[MODEL_SYMBOL[kind]]
    net = total - data
    return {
        "sketch_bytes": total,
        "test_data_bytes": data,
        "net_firmware_bytes": net,
        "model_bytes": model,
        "runtime_and_code_bytes": net - model,
    }


def measure(results_dir=ROOT / "results"):
    results_dir = Path(results_dir)
    out = {"note": "net = sketch bytes minus the test arrays; sketch bytes come from the arduino-cli log"}
    for kind in ("float", "int8"):
        sizes = symbol_sizes((results_dir / ("model2_flash_symbols_%s.txt" % kind)).read_text())
        total = sketch_bytes((results_dir / ("compile_har_%s.txt" % kind)).read_text())
        out[kind] = net_flash(kind, total, sizes)
    out["net_difference_float_minus_int8"] = out["float"]["net_firmware_bytes"] - out["int8"]["net_firmware_bytes"]
    (results_dir / "model2_flash.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
