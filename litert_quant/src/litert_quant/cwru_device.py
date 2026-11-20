"""Parse board captures of the CWRU autoencoder benchmark and compare them with the PC run.

AI-assisted: drafted with Claude, checked by the author.
A capture holds START, one S line per window (index, label, anomaly score, cycles) and a
DONE line (model, arena used, count, correct decisions, model bytes). The board applies
its own threshold. Here the printed scores are thresholded again with the same PC value,
and both counts are reported, because 6 decimal places could flip a score that sits
right on the threshold. The core clock is taken as 64 MHz for the millisecond figure;
cycles are the measurement.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
CLOCK_HZ = 64_000_000


class CaptureError(ValueError):
    pass


def parse_capture(path):
    model = count = done = None
    rows = []
    for raw in Path(path).read_text().splitlines():
        parts = raw.strip().split(",")
        tag = parts[0]
        if tag == "START":
            model, count = parts[1], int(parts[2])
        elif tag == "S":
            i, label, cyc = int(parts[1]), int(parts[2]), int(parts[4])
            if i != len(rows):
                raise CaptureError("window %d out of order" % i)
            rows.append((i, label, float(parts[3]), cyc))
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
    return {
        "model": model,
        "labels": np.array([r[1] for r in rows], dtype=np.int64),
        "scores": np.array([r[2] for r in rows], dtype=np.float64),
        "cycles": np.array([r[3] for r in rows], dtype=np.int64),
        "done": done,
    }


def auc(y, score):
    """Area under the ROC curve from ranks (ties get average ranks)."""
    y = np.asarray(y)
    r = rankdata(score)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarise(cap, pc_scores, y_true, threshold):
    if not np.array_equal(cap["labels"], np.asarray(y_true)):
        raise CaptureError("board labels differ from the subset labels")
    s = cap["scores"]
    pc = np.asarray(pc_scores, dtype=np.float64)
    y = np.asarray(y_true)
    cyc = cap["cycles"]
    flagged = (s > threshold).astype(int)
    pc_flag = (pc > threshold).astype(int)
    diff = np.flatnonzero(flagged != pc_flag)
    rel = np.abs(s - pc) / np.maximum(np.abs(pc), 1e-12)
    correct = int(np.sum(flagged == y))
    med = float(np.median(cyc))
    return {
        "model": cap["model"],
        "n": int(len(s)),
        "n_normal": int(np.sum(y == 0)),
        "n_fault": int(np.sum(y == 1)),
        "threshold": float(threshold),
        "correct_board_done": cap["done"]["correct"],
        "correct_from_printed_scores": correct,
        "done_matches_printed": cap["done"]["correct"] == correct,
        "accuracy_board_done": cap["done"]["correct"] / len(s),
        "tpr": float(np.mean(flagged[y == 1])),
        "fpr": float(np.mean(flagged[y == 0])),
        "auc_board": auc(y, s),
        "auc_pc": auc(y, pc),
        "arena_used": cap["done"]["arena_used"],
        "model_bytes": cap["done"]["model_bytes"],
        "board_pc_decision_disagree": int(len(diff)),
        "board_pc_decision_disagree_windows": diff.tolist(),
        "score_max_abs_diff": float(np.max(np.abs(s - pc))),
        "score_median_rel_diff": float(np.median(rel)),
        "score_max_rel_diff": float(np.max(rel)),
        "score_spearman_board_pc": float(spearmanr(s, pc).statistic),
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
    sub = json.loads((results_dir / "model3_subset.json").read_text())
    idx = np.array(sub["indices"])
    y = np.load(results_dir / "model3_pc_scores.npz")["y_test"][idx]
    out = {"clock_assumption": "64 MHz core clock for the ms figure; cycles are measured"}
    caps = {}
    for kind in ("float", "int8"):
        caps[kind] = parse_capture(results_dir / ("device_cwru_%s.txt" % kind))
        out[kind] = summarise(caps[kind], sub["pc_%s_scores" % kind], y, sub["threshold_%s" % kind])
    f, q = caps["float"]["scores"], caps["int8"]["scores"]
    ff = (f > sub["threshold_float"]).astype(int)
    qq = (q > sub["threshold_int8"]).astype(int)
    out["board_float_vs_int8"] = {
        "score_spearman": float(spearmanr(f, q).statistic),
        "score_median_rel_diff": float(np.median(np.abs(f - q) / np.maximum(np.abs(f), 1e-12))),
        "decision_disagree": int(np.sum(ff != qq)),
    }
    out["median_cycle_ratio_float_over_int8"] = out["float"]["cycles"]["median"] / out["int8"]["cycles"]["median"]
    (results_dir / "model3_device.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
