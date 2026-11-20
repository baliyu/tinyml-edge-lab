"""Generate the C headers that the CWRU sketch includes.

AI-assisted: drafted with Claude, checked by the author.
The board cannot hold all the test windows, so the sketch carries a fixed subset:
NORMAL_N windows of the 3 hp normal recording and PER_GROUP windows of every fault
kind and size, chosen with a seeded generator. The subset indices and the PC scores on
them are written to results/model3_subset.json, so board and PC are compared on
identical windows. Inputs are the preprocessed windows the PC models saw. The int8
inputs are the exact ones the PC used, and the float inputs are stored with 9
significant digits, which round-trips float32. The score thresholds are each model's
99th percentile of its own training scores, from results/model3_pc.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from litert_quant.cwru_data import DEFAULT_ROOT, WINDOW, catalogue, fingerprint, make_split
from litert_quant.firmware_data import HEADER, c_array, quantise_input
from litert_quant.har_firmware_data import PRAGMA, c_float_array
from litert_quant.model1 import input_quantization
from litert_quant.model3 import preprocess

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = {"float": "cwru_ae_float.tflite", "int8": "cwru_ae_int8.tflite"}
NORMAL_N = 100
PER_GROUP = 24
SEED = 0
TEST_SIZE = WINDOW


def select_subset(y, kind, size, normal_n=NORMAL_N, per_group=PER_GROUP, seed=SEED):
    """Sorted indices: normal_n normal windows and per_group windows of each fault kind and size."""
    rng = np.random.default_rng(seed)
    y, kind, size = np.asarray(y), np.asarray(kind), np.asarray(size)
    normal = np.flatnonzero(y == 0)
    if len(normal) < normal_n:
        raise ValueError("only %d normal windows" % len(normal))
    picks = [rng.choice(normal, size=normal_n, replace=False)]
    fault = y == 1
    groups = sorted({(str(k), int(s)) for k, s in zip(kind[fault], size[fault])})
    for k, s in groups:
        idx = np.flatnonzero(fault & (kind == k) & (size == s))
        if len(idx) < per_group:
            raise ValueError("group %s %d has only %d windows" % (k, s, len(idx)))
        picks.append(rng.choice(idx, size=per_group, replace=False))
    return np.sort(np.concatenate(picks))


def build(models_dir, out_dir, scores_npz, result_json, report_json, root=DEFAULT_ROOT,
          recordings=None, normal_n=NORMAL_N, per_group=PER_GROUP, seed=SEED):
    """Write the headers to out_dir and the subset report to report_json."""
    models_dir, out_dir = Path(models_dir), Path(out_dir)
    recs = catalogue() if recordings is None else list(recordings)
    s = make_split(root, recs, WINDOW)
    d = np.load(scores_npz)
    r = json.loads(Path(result_json).read_text())
    if not (np.array_equal(d["y_test"], s.y_test) and np.array_equal(d["rec_test"], s.rec_test)):
        raise ValueError("the scores belong to a different test set")
    idx = select_subset(s.y_test, s.kind_test, d["size_test"], normal_n, per_group, seed)
    xs = preprocess(s.x_test[idx], r["scaler_std"], r["scaler_clip"])
    ys = s.y_test[idx]
    n = len(idx)

    int8_blob = (models_dir / MODEL_FILES["int8"]).read_bytes()
    scale, zero = input_quantization(int8_blob)
    q = quantise_input(xs, scale, zero).reshape(n, TEST_SIZE)
    thr_f = r["scores"]["float"]["operating_point"]["threshold"]
    thr_q = r["scores"]["int8"]["operating_point"]["threshold"]

    files = {}
    for kind, fname in MODEL_FILES.items():
        blob = (models_dir / fname).read_bytes()
        files["model_%s.h" % kind] = HEADER + PRAGMA + c_array("unsigned char", "g_model_%s" % kind, blob, align=16)
    meta = HEADER + PRAGMA
    meta += "// SHA-256 of the subset float inputs and labels: %s\n" % fingerprint(xs, ys)
    meta += "#define TEST_COUNT %d\n#define TEST_SIZE %d\n" % (n, TEST_SIZE)
    meta += "#define INPUT_ZERO_POINT %d\n" % zero
    meta += "#define SCORE_THRESHOLD_FLOAT %.9gf\n#define SCORE_THRESHOLD_INT8 %.9gf\n\n" % (thr_f, thr_q)
    meta += c_array("uint8_t", "test_labels", ys.astype(np.uint8).tolist(), per_line=32)
    files["test_meta.h"] = meta
    files["test_input_int8.h"] = HEADER + PRAGMA + c_array("int8_t", "test_input_int8", q.ravel().tolist(), per_line=32)
    files["test_input_float.h"] = HEADER + PRAGMA + c_float_array("test_input_float", xs.reshape(n, TEST_SIZE))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (out_dir / name).write_text(body)

    pf, pq = d["float"][idx], d["int8"][idx]
    report = {
        "normal_n": normal_n,
        "per_group": per_group,
        "seed": seed,
        "n": n,
        "n_normal": int(np.sum(ys == 0)),
        "n_fault": int(np.sum(ys == 1)),
        "indices": idx.tolist(),
        "subset_fingerprint": fingerprint(xs, ys),
        "input_scale": scale,
        "input_zero_point": zero,
        "threshold_float": thr_f,
        "threshold_int8": thr_q,
        "pc_float_scores": pf.tolist(),
        "pc_int8_scores": pq.tolist(),
        "pc_correct": {
            "float": int(np.sum((pf > thr_f).astype(int) == ys)),
            "int8": int(np.sum((pq > thr_q).astype(int) == ys)),
        },
        "pc_disagree_float_int8": int(np.sum((pf > thr_f) != (pq > thr_q))),
        "header_bytes": {k: len(v.encode()) for k, v in files.items()},
        "header_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
    }
    report_json = Path(report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    return report
