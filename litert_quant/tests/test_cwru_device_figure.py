"""Tests for the CWRU device figure, on synthetic numbers (not real results).

AI-assisted: drafted with Claude, checked by the author.
"""
import json

from litert_quant.cwru_device_figure import make_device_figure


def _dev(correct, cycles, arena):
    return {
        "n": 100, "correct_board_done": correct, "arena_used": arena, "board_pc_decision_disagree": 0,
        "cycles": {"median": float(cycles)}, "ms_median_at_64mhz": cycles / 64000.0,
    }


def _fl(model, rest):
    return {"model_bytes": model, "runtime_and_code_bytes": rest, "net_firmware_bytes": model + rest}


def test_figure_is_written_from_synthetic_inputs(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"float": _dev(97, 60000, 1500), "int8": _dev(96, 36000, 1100)}))
    (tmp_path / "f.json").write_text(json.dumps({"float": _fl(20000, 128000), "int8": _fl(8000, 128000)}))
    out = make_device_figure(tmp_path / "d.json", tmp_path / "f.json", tmp_path / "out" / "fig.png")
    data = out.read_bytes()
    assert data[:4] == b"\x89PNG" and len(data) > 5000


def test_perfect_accuracy_does_not_break_the_interval(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"float": _dev(100, 60000, 1500), "int8": _dev(100, 36000, 1100)}))
    (tmp_path / "f.json").write_text(json.dumps({"float": _fl(20000, 128000), "int8": _fl(8000, 128000)}))
    out = make_device_figure(tmp_path / "d.json", tmp_path / "f.json", tmp_path / "fig2.png")
    assert out.exists() and out.stat().st_size > 5000
