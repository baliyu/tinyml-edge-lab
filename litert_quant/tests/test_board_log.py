import json

import numpy as np
import pytest

from litert_quant.board_log import (
    analyse_log,
    cli,
    compare,
    parse_log,
    summarise_cycles,
)


def make_log(labels, preds, cycles=None, model="int8", arena=3000, model_bytes=5640):
    n = len(labels)
    cycles = cycles or [1000 + i for i in range(n)]
    lines = ["START,%s,%d" % (model, n)]
    for i in range(n):
        lines.append("S,%d,%d,%d,%d" % (i, labels[i], preds[i], cycles[i]))
    correct = sum(1 for a, b in zip(labels, preds) if a == b)
    lines.append("DONE,%s,%d,%d,%d,%d" % (model, arena, n, correct, model_bytes))
    return "\n".join(lines) + "\n"


def test_parse_valid_log():
    parsed = parse_log(make_log([1, 2, 3], [1, 2, 0]))
    assert parsed["model"] == "int8"
    assert parsed["count"] == 3
    assert parsed["correct"] == 2
    assert parsed["arena_used_bytes"] == 3000
    assert parsed["model_bytes"] == 5640
    assert parsed["samples"][2] == (2, 3, 0, 1002)


def test_error_line_raises():
    with pytest.raises(ValueError):
        parse_log("START,int8,1\nERROR,invoke\n")


def test_incomplete_log_raises():
    with pytest.raises(ValueError):
        parse_log("START,int8,2\nS,0,1,1,100\n")


def test_wrong_sample_count_raises():
    text = make_log([1, 2, 3], [1, 2, 3]).replace("S,2,3,3,1002\n", "")
    with pytest.raises(ValueError):
        parse_log(text)


def test_board_correct_count_must_match_lines():
    text = make_log([1, 2, 3], [1, 2, 3]).replace(
        "DONE,int8,3000,3,3,5640", "DONE,int8,3000,3,2,5640"
    )
    with pytest.raises(ValueError):
        parse_log(text)


def test_cycle_summary():
    parsed = parse_log(make_log([0] * 5, [0] * 5, [10, 20, 30, 40, 50]))
    cyc = summarise_cycles(parsed)
    assert cyc["min"] == 10
    assert cyc["median"] == 30
    assert cyc["mean"] == 30.0
    assert cyc["max"] == 50
    assert cyc["first"] == 10


def test_compare_counts_agreement():
    parsed = parse_log(make_log([1, 2, 3], [1, 2, 0]))
    result = compare(parsed, [1, 2, 3], [1, 2, 3])
    assert result["agree_with_pc"] == 2
    assert result["disagree_with_pc"] == 1
    assert abs(result["board_accuracy"] - 2 / 3) < 1e-12


def test_compare_rejects_different_labels():
    parsed = parse_log(make_log([1, 2, 3], [1, 2, 0]))
    with pytest.raises(ValueError):
        compare(parsed, [1, 2, 3], [1, 2, 9])


def write_npz(path):
    labels = np.array([1, 2, 3])
    np.savez(path, y_test=labels, keras=labels, float=labels, int8=np.array([1, 2, 0]))


def test_analyse_log_writes_json(tmp_path):
    log = tmp_path / "board_test.txt"
    log.write_text(make_log([1, 2, 3], [1, 2, 0]))
    npz = tmp_path / "pc.npz"
    write_npz(npz)
    out = tmp_path / "out.json"
    result = analyse_log(log, npz, out)
    assert result["agree_with_pc"] == 3
    assert result["comparison_to"] == "int8"
    assert len(result["log_sha256"]) == 64
    assert json.loads(out.read_text()) == result


def test_cli_writes_next_to_the_log_name(tmp_path):
    (tmp_path / "results").mkdir()
    write_npz(tmp_path / "results" / "model1_pc_predictions.npz")
    log = tmp_path / "board_test.txt"
    log.write_text(make_log([1, 2, 3], [1, 2, 0]))
    assert cli(["analyse", str(log)], root=tmp_path) == 0
    assert (tmp_path / "results" / "board_test.json").exists()
