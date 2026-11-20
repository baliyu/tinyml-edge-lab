"""Tests for the CWRU board capture parser, on tiny synthetic captures.

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import numpy as np
import pytest

from litert_quant.cwru_device import CaptureError, analyse, auc, parse_capture, summarise


def _capture(path, model, labels, scores, cycles, correct=None, count=None):
    lines = ["START,%s,%d" % (model, len(labels) if count is None else count)]
    for i, (lab, sc, cyc) in enumerate(zip(labels, scores, cycles)):
        lines.append("S,%d,%d,%.6f,%d" % (i, lab, sc, cyc))
    lines.append("DONE,%s,100,%d,%d,50" % (model, len(labels), 3 if correct is None else correct))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parse_reads_scores_cycles_and_done(tmp_path):
    cap = parse_capture(_capture(tmp_path / "c.txt", "float", [0, 1], [0.25, 2.5], [10, 20], correct=2))
    assert cap["model"] == "float"
    assert cap["scores"].tolist() == [0.25, 2.5] and cap["cycles"].tolist() == [10, 20]
    assert cap["labels"].tolist() == [0, 1]
    assert cap["done"]["correct"] == 2 and cap["done"]["arena_used"] == 100


def test_out_of_order_error_and_count_mismatch_raise(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("START,float,2\nS,1,0,0.1,5\nS,0,0,0.1,5\nDONE,float,1,2,2,3\n")
    with pytest.raises(CaptureError):
        parse_capture(p)
    p.write_text("START,float,1\nERROR,invoke\n")
    with pytest.raises(CaptureError):
        parse_capture(p)
    with pytest.raises(CaptureError):
        parse_capture(_capture(tmp_path / "b.txt", "float", [0, 1], [0.1, 2.0], [1, 2], count=3))


def test_auc_hand_values():
    assert auc([0, 0, 1, 1], [0.1, 0.9, 2.0, 3.0]) == 1.0
    assert auc([0, 0, 1, 1], [3.0, 2.0, 1.0, 0.5]) == 0.0
    assert abs(auc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) - 0.75) < 1e-12


def test_summarise_decisions_differences_and_cycles(tmp_path):
    cap = parse_capture(_capture(tmp_path / "c.txt", "float", [0, 0, 1, 1], [0.1, 0.9, 2.0, 3.0], [10, 20, 30, 40], correct=3))
    s = summarise(cap, [0.1, 0.2, 2.0, 3.0], np.array([0, 0, 1, 1]), 0.5)
    assert s["correct_from_printed_scores"] == 3 and s["done_matches_printed"] is True
    assert s["board_pc_decision_disagree"] == 1 and s["board_pc_decision_disagree_windows"] == [1]
    assert s["cycles"]["median"] == 25.0 and s["cycles"]["min"] == 10 and s["cycles"]["max"] == 40
    assert s["tpr"] == 1.0 and s["fpr"] == 0.5 and s["auc_board"] == 1.0
    assert s["score_max_abs_diff"] == pytest.approx(0.7)
    with pytest.raises(CaptureError):
        summarise(cap, [0.1, 0.2, 2.0, 3.0], np.array([0, 0, 1, 0]), 0.5)


def test_done_count_mismatch_is_reported_not_raised(tmp_path):
    cap = parse_capture(_capture(tmp_path / "c.txt", "int8", [0, 1], [0.1, 2.0], [5, 6], correct=1))
    s = summarise(cap, [0.1, 2.0], np.array([0, 1]), 0.5)
    assert s["correct_board_done"] == 1 and s["correct_from_printed_scores"] == 2
    assert s["done_matches_printed"] is False


def test_analyse_writes_json_from_synthetic_files(tmp_path):
    idx = [1, 3, 5, 7]
    y_all = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    np.savez(tmp_path / "model3_pc_scores.npz", y_test=y_all)
    labels = y_all[idx].tolist()
    pc_f = [0.1, 2.0, 3.0, 4.0]
    pc_q = [0.12, 1.9, 3.1, 4.2]
    (tmp_path / "model3_subset.json").write_text(json.dumps({
        "indices": idx, "pc_float_scores": pc_f, "pc_int8_scores": pc_q,
        "threshold_float": 0.5, "threshold_int8": 0.5,
    }))
    _capture(tmp_path / "device_cwru_float.txt", "float", labels, pc_f, [100, 110, 120, 130], correct=4)
    _capture(tmp_path / "device_cwru_int8.txt", "int8", labels, pc_q, [50, 60, 70, 80], correct=4)
    out = analyse(tmp_path)
    assert (tmp_path / "model3_device.json").exists()
    assert out["float"]["board_pc_decision_disagree"] == 0 and out["int8"]["correct_board_done"] == 4
    assert out["median_cycle_ratio_float_over_int8"] == pytest.approx(115.0 / 65.0)
    assert out["board_float_vs_int8"]["decision_disagree"] == 0
    assert json.loads((tmp_path / "model3_device.json").read_text())["float"]["n"] == 4
