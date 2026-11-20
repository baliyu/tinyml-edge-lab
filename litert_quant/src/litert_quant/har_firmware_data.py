"""Generate the C headers that the HAR sketch includes.

AI-assisted: drafted with Claude, checked by the author.
The board cannot hold all 2,947 test windows (1.1 MB as int8), so the sketch
carries a fixed stratified subset: PER_CLASS windows of each class, chosen with
a seeded generator. The subset indices and the PC predictions on them are
written to results/model2_subset.json, so board and PC are compared on
identical windows. The int8 inputs are the exact ones the PC used, and the
float inputs are stored with 9 significant digits, which round-trips float32.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from litert_quant.firmware_data import HEADER, c_array, quantise_input
from litert_quant.har_data import DEFAULT_ROOT, LABELS, fingerprint, load_har
from litert_quant.model1 import input_quantization

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = {"float": "har_cnn_float.tflite", "int8": "har_cnn_int8.tflite"}
PER_CLASS = 60
SEED = 0
TEST_SIZE = 128 * 3
PRAGMA = "#pragma once\n#include <stdint.h>\n\n"


def select_subset(y, per_class=PER_CLASS, seed=SEED):
    """Sorted indices of per_class randomly chosen windows of every class."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    picks = []
    for c in range(len(LABELS)):
        idx = np.flatnonzero(y == c)
        if len(idx) < per_class:
            raise ValueError("class %d has only %d windows" % (c, len(idx)))
        picks.append(rng.choice(idx, size=per_class, replace=False))
    return np.sort(np.concatenate(picks))


def c_float_array(name, values, per_line=8):
    """C source for a const float array; 9 significant digits round-trip float32."""
    values = np.asarray(values, dtype=np.float32).ravel()
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("  " + ", ".join("%.9g" % float(v) for v in chunk) + ",")
    return "const float %s[] = {\n%s\n};\nconst unsigned int %s_len = %d;\n" % (
        name,
        "\n".join(lines),
        name,
        len(values),
    )


def build(models_dir, out_dir, pred_npz, report_json, root=DEFAULT_ROOT, per_class=PER_CLASS, seed=SEED):
    """Write the headers to out_dir and the subset report to report_json."""
    models_dir, out_dir = Path(models_dir), Path(out_dir)
    _, x_test, _, y_test, _, _ = load_har(root)
    preds = np.load(pred_npz)
    if not np.array_equal(preds["y_test"], y_test):
        raise ValueError("the predictions belong to a different test set")
    idx = select_subset(y_test, per_class, seed)
    xs, ys = x_test[idx], y_test[idx]
    n = len(idx)

    int8_blob = (models_dir / MODEL_FILES["int8"]).read_bytes()
    scale, zero = input_quantization(int8_blob)
    q = quantise_input(xs, scale, zero).reshape(n, TEST_SIZE)

    files = {}
    for kind, fname in MODEL_FILES.items():
        blob = (models_dir / fname).read_bytes()
        files["model_%s.h" % kind] = HEADER + PRAGMA + c_array("unsigned char", "g_model_%s" % kind, blob, align=16)
    meta = HEADER + PRAGMA
    meta += "// SHA-256 of the subset float inputs and labels: %s\n" % fingerprint(xs, ys)
    meta += "#define TEST_COUNT %d\n#define TEST_SIZE %d\n" % (n, TEST_SIZE)
    meta += "#define INPUT_ZERO_POINT %d\n\n" % zero
    meta += c_array("uint8_t", "test_labels", ys.astype(np.uint8).tolist(), per_line=32)
    files["test_meta.h"] = meta
    files["test_input_int8.h"] = HEADER + PRAGMA + c_array("int8_t", "test_input_int8", q.ravel().tolist(), per_line=32)
    files["test_input_float.h"] = HEADER + PRAGMA + c_float_array("test_input_float", xs.reshape(n, TEST_SIZE))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (out_dir / name).write_text(body)

    report = {
        "per_class": per_class,
        "seed": seed,
        "n": n,
        "indices": idx.tolist(),
        "subset_fingerprint": fingerprint(xs, ys),
        "input_scale": scale,
        "input_zero_point": zero,
        "pc_correct": {k: int(np.sum(preds[k][idx] == ys)) for k in ("keras", "float", "int8")},
        "pc_disagree_float_int8": int(np.sum(preds["float"][idx] != preds["int8"][idx])),
        "header_bytes": {k: len(v.encode()) for k, v in files.items()},
        "header_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
    }
    report_json = Path(report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    return report
