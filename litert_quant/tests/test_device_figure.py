import json

import pytest

from litert_quant.compile_log import parse_compile_log
from litert_quant.device_figure import main, make_device_figure, wilson

COMPILE = (
    "Sketch uses 174280 bytes (17%) of program storage space. Maximum is 983040 bytes.\n"
    "Global variables use 62744 bytes (23%) of dynamic memory, leaving 199400 bytes "
    "for local variables. Maximum is 262144 bytes.\n"
)


def fake_result(model, median, correct, arena, model_bytes):
    return {
        "model": model,
        "count": 360,
        "arena_used_bytes": arena,
        "model_bytes": model_bytes,
        "cycles": {"min": median - 2, "median": median, "mean": median - 1.0, "max": median, "first": median},
        "agree_with_pc": 360,
        "disagree_with_pc": 0,
        "board_accuracy": correct / 360,
        "comparison_to": model,
        "log_file": "x.txt",
        "log_sha256": "0" * 64,
    }


def write_inputs(folder):
    (folder / "board_float_run1.json").write_text(json.dumps(fake_result("float", 288943, 337, 2304, 10120)))
    (folder / "board_int8_fixed_run1.json").write_text(json.dumps(fake_result("int8", 105067, 338, 1764, 5424)))
    (folder / "compile_float.txt").write_text(COMPILE)
    (folder / "compile_int8.txt").write_text(COMPILE.replace("174280", "169648"))


def test_parse_compile_log():
    got = parse_compile_log(COMPILE)
    assert got == {"flash_bytes": 174280, "flash_max": 983040, "static_ram_bytes": 62744}


def test_parse_compile_log_rejects_empty():
    with pytest.raises(ValueError):
        parse_compile_log("nothing here")


def test_wilson_perfect_score():
    lo, hi = wilson(360, 360)
    assert hi == 1.0
    assert 0.98 < lo < 1.0


def test_make_device_figure_writes_png(tmp_path):
    write_inputs(tmp_path)
    out = tmp_path / "fig.png"
    make_device_figure(
        tmp_path / "board_float_run1.json",
        tmp_path / "board_int8_fixed_run1.json",
        tmp_path / "compile_float.txt",
        tmp_path / "compile_int8.txt",
        out,
    )
    data = out.read_bytes()
    assert data[:4] == b"\x89PNG"
    assert len(data) > 1000


def test_main_writes_figure_under_results(tmp_path):
    (tmp_path / "results").mkdir()
    write_inputs(tmp_path / "results")
    assert main(root=tmp_path) == 0
    assert (tmp_path / "results" / "figures" / "model1_device.png").exists()
