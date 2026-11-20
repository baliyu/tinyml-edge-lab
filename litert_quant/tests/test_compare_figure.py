"""Tests for the CNN against tree comparison figure, on synthetic numbers.

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import pytest

from litert_quant.compare_figure import make_comparison_figure, tree_footprint


def _footprint(path):
    def v(name, flash):
        return {"variant": name, "targets": {"cortex-m4f": {"flash_bytes": flash, "ram_bytes": 0}}}
    data = {"models": [
        {"id": "other", "variants": [v("float thresholds", 1)]},
        {"id": "tree_fixed6", "variants": [v("float thresholds", 552), v("double thresholds", 1876)]},
    ]}
    path.write_text(json.dumps(data))
    return path


def test_tree_footprint_picks_the_named_model_and_variants(tmp_path):
    fp = tree_footprint(_footprint(tmp_path / "f.json"))
    assert fp == {
        "float thresholds": {"flash_bytes": 552, "ram_bytes": 0},
        "double thresholds": {"flash_bytes": 1876, "ram_bytes": 0},
    }


def test_tree_footprint_unknown_model_raises(tmp_path):
    with pytest.raises(KeyError):
        tree_footprint(_footprint(tmp_path / "f.json"), model_id="nope")


def test_figure_is_written_from_synthetic_inputs(tmp_path):
    pc = {"n_test": 100, "correct_float_tflite": 85, "correct_int8_tflite": 86,
          "tree_baseline": {"test_accuracy": 0.8, "n_test": 100}}
    dev = {k: {"arena_used": 1234, "cycles": {"median": 1000000.0}, "ms_median_at_64mhz": 15.6} for k in ("float", "int8")}
    fl = {k: {"model_bytes": 5000, "net_firmware_bytes": 150000} for k in ("float", "int8")}
    for name, obj in (("pc.json", pc), ("dev.json", dev), ("fl.json", fl)):
        (tmp_path / name).write_text(json.dumps(obj))
    out = make_comparison_figure(tmp_path / "pc.json", tmp_path / "dev.json", tmp_path / "fl.json",
                                 _footprint(tmp_path / "f.json"), tmp_path / "o" / "fig.png")
    data = out.read_bytes()
    assert data[:4] == b"\x89PNG" and len(data) > 5000
