"""Tests for the HAR device figure, on synthetic numbers (not real results).

AI-assisted: drafted with Claude, checked by the author.
"""
import json

from litert_quant.har_device_figure import make_device_figure, wilson


def test_wilson_matches_a_hand_calculation():
    lo, hi = wilson(315, 360)
    assert abs(lo - 0.8368) < 1e-3 and abs(hi - 0.9053) < 1e-3


def test_wilson_edges_stay_inside_zero_and_one():
    lo, hi = wilson(10, 10)
    assert 0.0 < lo < 1.0 and abs(hi - 1.0) < 1e-9
    lo, hi = wilson(0, 10)
    assert abs(lo) < 1e-9 and 0.0 < hi < 1.0


def test_figure_is_written_from_synthetic_inputs(tmp_path):
    def dev(correct, cycles, arena):
        return {
            "n": 100, "correct": correct, "arena_used": arena, "board_pc_disagree": 0,
            "cycles": {"median": float(cycles)}, "ms_median_at_64mhz": cycles / 64000.0,
        }
    def fl(model, rest):
        return {"model_bytes": model, "runtime_and_code_bytes": rest, "net_firmware_bytes": model + rest}
    (tmp_path / "d.json").write_text(json.dumps({"float": dev(80, 300000, 2000), "int8": dev(79, 100000, 900)}))
    (tmp_path / "f.json").write_text(json.dumps({"float": fl(9000, 100000), "int8": fl(4000, 100000)}))
    out = make_device_figure(tmp_path / "d.json", tmp_path / "f.json", tmp_path / "out" / "fig.png")
    data = out.read_bytes()
    assert data[:4] == b"\x89PNG" and len(data) > 5000
