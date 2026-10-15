import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "record_board_run.py"
spec = importlib.util.spec_from_file_location("record_board_run", TOOL)
rbr = importlib.util.module_from_spec(spec)
sys.modules["record_board_run"] = rbr
spec.loader.exec_module(rbr)

HEADER = "pc_time,window,t_ticks,gesture,compute_us"
README = "# Demo\n\nintro\n\n## Not done yet\n- something\n"


def log(path, n, us=3800, start_ticks=1_000_000, gesture="rest", first=1, dup=0):
    rows = [f"2026-10-15T12:00:{i % 60:02d}.000,{first + i},{(start_ticks + 3_840_000 * i) % (1 << 32)},{gesture},{us + (i % 3) * 50}" for i in range(n)]
    rows = rows[:1] + [rows[0]] * dup + rows[1:]                    # a flood of repeats after the first line, as an old monitor logged it
    Path(path).write_text("\n".join([HEADER, *rows]) + "\n")
    return str(path)


def test_repeated_rows_are_dropped_and_the_rest_is_read(tmp_path):
    rows = rbr.read_log(log(tmp_path / "a.csv", 20, dup=300))
    assert len(rows) == 20 and rows[0]["window"] == 1 and rows[-1]["window"] == 20


def test_unreadable_rows_are_skipped(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(HEADER + "\nx,1,2,rest,3\nbad,row\n2026,5,100,tilt,3000\n")
    assert [r["window"] for r in rbr.read_log(p)] == [1, 5]


def test_the_summary_has_the_timing_the_interval_and_the_counts(tmp_path):
    s = rbr.summarise(rbr.read_log(log(tmp_path / "a.csv", 30, us=3800)), "FFT", "a.csv")
    assert s["windows"] == 30 and s["mean_ms"] == pytest.approx(3.85, abs=0.06) and s["max_ms"] == pytest.approx(3.9) and s["min_ms"] == pytest.approx(3.8)
    assert s["median_interval_s"] == pytest.approx(1.28) and "sample_period_ms" not in s and s["windows_not_consecutive"] == 0
    assert s["gesture_counts"] == {"rest": 30} and s["duration_s"] == pytest.approx(29 * 1.28)


def test_the_32_bit_counter_wrapping_does_not_disturb_the_interval(tmp_path):
    s = rbr.summarise(rbr.read_log(log(tmp_path / "a.csv", 20, start_ticks=(1 << 32) - 10_000_000)), "x", "a")
    assert s["median_interval_s"] == pytest.approx(1.28) and s["duration_s"] == pytest.approx(19 * 1.28)


def test_missing_windows_are_noticed(tmp_path):
    p = tmp_path / "a.csv"
    log(p, 20)
    lines = p.read_text().splitlines()
    del lines[5:7]
    p.write_text("\n".join(lines) + "\n")
    assert rbr.summarise(rbr.read_log(p), "x", "a")["windows_not_consecutive"] == 1


def test_a_run_that_is_too_short_is_refused(tmp_path):
    with pytest.raises(ValueError, match="only 5 windows"):
        rbr.summarise(rbr.read_log(log(tmp_path / "a.csv", 5)), "x", "a.csv")


def test_the_section_is_written_from_the_numbers_and_replaced_not_duplicated(tmp_path, capsys):
    readme, out = tmp_path / "README.md", tmp_path / "res"
    readme.write_text(README)
    fft, direct = log(tmp_path / "fft.csv", 69, us=3750), log(tmp_path / "old.csv", 46, us=16000, dup=200)      # means: 3.8 ms and 16.0 ms
    args = ["--run", f"FFT features={fft}", "--run", f"direct-transform features (first build)={direct}", "--out", str(out), "--readme", str(readme)]
    assert rbr.main(args) == 0
    s = readme.read_text()
    assert "| FFT features | 69 | 3.8 ms | 3.9 ms |" in s and "| direct-transform features (first build) | 46 | 16.0 ms | 16.1 ms |" in s
    assert s.index("## Measured on the board") < s.index("## Not done yet") and "19 % of one sample period" in s       # 3.8 ms of a 19.6 ms sample period
    assert "window comes every 1.28 s" in s and "divided by" not in s and "Interval between windows" not in s
    assert rbr.main(args) == 0
    again = readme.read_text()
    assert again == s and again.count(rbr.START) == 1
    rep = json.loads((out / "board_run.json").read_text())
    assert [r["windows"] for r in rep["runs"]] == [69, 46] and "nothing about accuracy" in rep["note"].replace("says ", "")


def test_bad_arguments_are_refused(tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert rbr.main(["--run", "no equals sign", "--readme", str(readme)]) == 2
    assert rbr.main(["--run", "x=" + str(tmp_path / "missing.csv"), "--readme", str(readme)]) == 2
    short = log(tmp_path / "s.csv", 4)
    assert rbr.main(["--run", f"x={short}", "--readme", str(readme), "--out", str(tmp_path / "o")]) == 1
    assert "only 4 windows" in capsys.readouterr().err
    plain = tmp_path / "plain.md"
    plain.write_text("# no marker here\n")
    ok = log(tmp_path / "ok.csv", 20)
    with pytest.raises(SystemExit):
        rbr.main(["--run", f"x={ok}", "--readme", str(plain), "--out", str(tmp_path / "o")])
