"""Parse board captures of the HAR benchmark and compare them with the PC run.

AI-assisted: drafted with Claude, checked by the author.
A capture holds START, one S line per window (index, label, prediction, cycles),
optional L lines (int8 outputs) and a DONE line. The core clock of the
nRF52840 is taken as 64 MHz for the millisecond figure; cycles are the measurement.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLOCK_HZ = 64_000_000


class CaptureError(ValueError):
    pass


def parse_capture(path):
    model = count = done = None
    rows, logits = [], {}
    for raw in Path(path).read_text().splitlines():
        parts = raw.strip().split(",")
        tag = parts[0]
        if tag == "START":
            model, count = parts[1], int(parts[2])
        elif tag == "S":
            i, label, pred, cyc = (int(p) for p in parts[1:5])
            if i != len(rows):
                raise CaptureError("window %d out of order" % i)
            rows.append((i, label, pred, cyc))
        elif tag == "L":
            logits[int(parts[1])] = [int(p) for p in parts[2:]]
        elif tag == "DONE":
            done = {
                "model": parts[1],
                "arena_used": int(parts[2]),
                "count": int(parts[3]),
                "correct": int(parts[4]),
                "model_bytes": int(parts[5]),
            }
        elif tag.startswith("ERROR"):
            raise CaptureError(raw.strip())
    if model is None or done is None:
        raise CaptureError("missing START or DONE")
    if not (len(rows) == count == done["count"]) or done["model"] != model:
        raise CaptureError("START, S lines and DONE disagree")
    arr = np.array(rows, dtype=np.int64).reshape(-1, 4)
    return {
        "model": model,
        "labels": arr[:, 1],
        "preds": arr[:, 2],
        "cycles": arr[:, 3],
        "logits": logits,
        "done": done,
    }


def summarise(cap, pc_pred, y_true):
    preds, cyc = cap["preds"], cap["cycles"]
    if not np.array_equal(cap["labels"], y_true):
        raise CaptureError("board labels differ from the subset labels")
    correct = int(np.sum(preds == y_true))
    if correct != cap["done"]["correct"]:
        raise CaptureError("DONE correct count does not match the S lines")
    diff = np.flatnonzero(preds != np.asarray(pc_pred))
    med = float(np.median(cyc))
    return {
        "model": cap["model"],
        "n": int(len(preds)),
        "correct": correct,
        "accuracy": correct / len(preds),
        "arena_used": cap["done"]["arena_used"],
        "model_bytes": cap["done"]["model_bytes"],
        "board_pc_disagree": int(len(diff)),
        "board_pc_disagree_windows": diff.tolist(),
        "cycles": {
            "min": int(cyc.min()),
            "median": med,
            "mean": float(cyc.mean()),
            "p95": float(np.percentile(cyc, 95)),
            "max": int(cyc.max()),
        },
        "ms_median_at_64mhz": med / CLOCK_HZ * 1000.0,
    }


def analyse(results_dir=ROOT / "results"):
    results_dir = Path(results_dir)
    idx = np.array(json.loads((results_dir / "model2_subset.json").read_text())["indices"])
    pc = np.load(results_dir / "model2_pc_predictions.npz")
    y = pc["y_test"][idx]
    out = {"clock_assumption": "64 MHz core clock for the ms figure; cycles are measured"}
    for kind in ("float", "int8"):
        cap = parse_capture(results_dir / ("device_har_%s.txt" % kind))
        out[kind] = summarise(cap, pc[kind][idx], y)
    out["median_cycle_ratio_float_over_int8"] = out["float"]["cycles"]["median"] / out["int8"]["cycles"]["median"]
    (results_dir / "model2_device.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
