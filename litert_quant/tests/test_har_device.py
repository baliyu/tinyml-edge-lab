"""Tests for the HAR board capture parser, on tiny synthetic captures.

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
import pytest

from litert_quant.har_device import CaptureError, parse_capture, summarise


def _capture(tmp_path, model, labels, preds, cycles, logits=False):
    lines = ["START,%s,%d" % (model, len(labels))]
    for i, (lab, pred, cyc) in enumerate(zip(labels, preds, cycles)):
        lines.append("S,%d,%d,%d,%d" % (i, lab, pred, cyc))
        if logits:
            lines.append("L,%d,1,2,3,4,5,6" % i)
    correct = sum(int(a == b) for a, b in zip(labels, preds))
    lines.append("DONE,%s,100,%d,%d,50" % (model, len(labels), correct))
    path = tmp_path / "c.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parse_reads_rows_logits_and_done(tmp_path):
    cap = parse_capture(_capture(tmp_path, "int8", [0, 1, 2], [0, 1, 1], [10, 20, 30], logits=True))
    assert cap["model"] == "int8"
    assert cap["preds"].tolist() == [0, 1, 1] and cap["cycles"].tolist() == [10, 20, 30]
    assert cap["logits"][1] == [1, 2, 3, 4, 5, 6]
    assert cap["done"]["correct"] == 2 and cap["done"]["arena_used"] == 100


def test_out_of_order_window_raises(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("START,float,2\nS,1,0,0,5\nS,0,0,0,5\nDONE,float,1,2,2,3\n")
    with pytest.raises(CaptureError):
        parse_capture(path)


def test_error_line_raises(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("START,float,1\nERROR,invoke\n")
    with pytest.raises(CaptureError):
        parse_capture(path)


def test_summarise_counts_disagreement_and_cycles(tmp_path):
    cap = parse_capture(_capture(tmp_path, "float", [0, 1, 2, 3], [0, 1, 2, 0], [10, 20, 30, 40]))
    s = summarise(cap, [0, 1, 2, 3], np.array([0, 1, 2, 3]))
    assert s["correct"] == 3 and s["board_pc_disagree"] == 1 and s["board_pc_disagree_windows"] == [3]
    assert s["cycles"]["median"] == 25.0 and s["cycles"]["min"] == 10 and s["cycles"]["max"] == 40
    with pytest.raises(CaptureError):
        summarise(cap, [0, 1, 2, 3], np.array([0, 1, 2, 4]))
