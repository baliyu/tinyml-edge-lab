import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "edge_gesture_monitor.py"
spec = importlib.util.spec_from_file_location("edge_gesture_monitor", TOOL)
egm = importlib.util.module_from_spec(spec)
sys.modules["edge_gesture_monitor"] = egm
spec.loader.exec_module(egm)

CONFIG = "# odr_hz=50 tick_hz=3000000 timer=hfrc3mhz window=128 step=64 features=40 classes=rest,shake_lr,shake_ud,circle,tilt"
HELLO = "# sparkfun_edge gesture demo v1, whoami 0x33"


def feed(lines):
    m = egm.Monitor()
    for ln in lines:
        m.feed(ln + "\r\n")
    return m


def windows(n, gesture="rest", start=1, us=3100):
    return [f"G,{start + i},{1000000 + 3750000 * i},{gesture},{us}" for i in range(n)]


def test_parse_a_window_line_and_the_start_up_lines():
    assert egm.parse_line("G,12,123456789,shake_lr,4100\r\n") == ("window", {"window": 12, "t_ticks": 123456789, "gesture": "shake_lr", "compute_us": 4100})
    assert egm.parse_line(HELLO) == ("hello", {"version": 1, "whoami": 0x33})
    kind, cfg = egm.parse_line(CONFIG)
    assert kind == "config" and cfg["tick_hz"] == 3000000 and cfg["classes"] == ["rest", "shake_lr", "shake_ud", "circle", "tilt"] and cfg["features"] == 40


def test_parse_a_statistics_line_and_an_error_line():
    kind, st = egm.parse_line("# stats windows=20 overruns=0 i2c_err=0 restarts=1 ignored=2 max_compute_us=3180")
    assert kind == "stats" and st == {"windows": 20, "overruns": 0, "i2c_err": 0, "restarts": 1, "ignored": 2, "max_compute_us": 3180}
    assert egm.parse_line("# error: accelerometer init failed, code 1, whoami 0x00 (expected 0x33)")[0] == "error"


def test_garbage_and_truncated_lines_are_not_windows():
    for bad in ("", "G,12,1234", "G,12,123456789,shake lr,4100", "G,x,1,rest,2", "hello", "12345,1,2,3", "# stats windows=abc"):
        assert egm.parse_line(bad) is None or egm.parse_line(bad)[0] == "comment"


def test_a_clean_run_has_no_problems_and_the_counts_add_up():
    m = feed([HELLO, CONFIG, *windows(5, "rest"), *windows(3, "shake_lr", start=6), "# stats windows=8 overruns=0 i2c_err=0 restarts=0 ignored=0 max_compute_us=3180"])
    s = m.summary()
    assert s["windows"] == 8 and s["counts"] == {"rest": 5, "shake_lr": 3} and s["problems"] == [] and s["max_compute_us"] == 3100


def test_lost_lines_are_noticed_from_the_gap_in_the_window_numbers():
    m = feed([CONFIG, *windows(3), *windows(2, start=7)])
    assert m.skipped_windows == 3 and any("lost on the way" in p for p in m.problems())


def test_unreadable_lines_are_counted_but_comments_are_not():
    m = feed([CONFIG, "G,1,2", "@@@@", "# some comment", *windows(2, start=1)])
    assert m.bad_lines == 2


def test_overruns_slow_windows_restarts_and_errors_are_called_out():
    m = feed([CONFIG, *windows(3, us=16000), "# stats windows=3 overruns=4 i2c_err=2 restarts=1 ignored=1 max_compute_us=16000", "# error: accelerometer init failed, code 1, whoami 0x00 (expected 0x33)"])
    text = " | ".join(m.problems())
    for needle in ("reported an error", "took 16.0 ms", "overwrote 4 samples", "2 accelerometer reads failed", "restarted the window 1", "repeated a timestamp"):
        assert needle in text


def test_a_missing_settings_line_or_a_different_timer_is_pointed_out():
    assert any("settings line was never seen" in p for p in feed(windows(2)).problems())
    other = "# odr_hz=50 tick_hz=32768 timer=xtal32k window=128 step=64 features=40 classes=rest,shake_lr"
    assert any("32768 Hz" in p for p in feed([other, *windows(2)]).problems())


def test_replaying_a_saved_file_prints_a_summary_and_writes_the_log(tmp_path, capsys):
    saved = tmp_path / "out.txt"
    saved.write_text("\n".join([HELLO, CONFIG, *windows(4, "tilt"), "# stats windows=4 overruns=0 i2c_err=0 restarts=0 ignored=0 max_compute_us=3100"]) + "\n")
    log = tmp_path / "live" / "run.csv"
    assert egm.main(["--replay", str(saved), "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "window    4  tilt" in out and "4 windows." in out and "tilt" in out and "Nothing looked wrong." in out
    rows = log.read_text().splitlines()
    assert rows[0] == "pc_time,window,t_ticks,gesture,compute_us" and len(rows) == 5 and rows[1].split(",")[1:] == ["1", "1000000", "tilt", "3100"]


def test_the_tool_needs_exactly_one_source_and_a_real_file(tmp_path, capsys):
    assert egm.main([]) == 2 and egm.main(["--port", "COM4", "--replay", "x.txt"]) == 2
    assert egm.main(["--replay", str(tmp_path / "missing.txt")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_reading_from_a_port_stops_after_the_time_is_up():
    class FakePort:
        def __init__(self): self.lines = [b"G,1,2,rest,3\r\n", b"", b"G,2,3,rest,3\r\n"]
        def readline(self): return self.lines.pop(0) if self.lines else b""
    t = iter(range(0, 100))
    got = list(egm.lines_from_port(FakePort(), 5, clock=lambda: next(t)))
    assert len(got) >= 1 and got[0].startswith("G,1")


# ---------------------------------------------------------------- the stale buffer Windows hands over when it opens the port
def test_a_flood_of_repeated_lines_at_the_start_is_dropped_and_not_blamed_on_the_board():
    line15 = windows(1, start=15)[0]
    m = feed([CONFIG, *windows(14), *[line15] * 500, windows(1, start=18)[0], *windows(2, start=19)])
    assert m.stale_lines == 499 and len(m.windows) == 14 + 1 + 1 + 2
    assert m.skipped_windows == 0 and not any("lost on the way" in p for p in m.problems())
    assert m.summary()["stale_lines"] == 499


def test_lines_lost_after_the_flood_has_ended_are_still_noticed():
    line15 = windows(1, start=15)[0]
    m = feed([CONFIG, *windows(14), *[line15] * 50, windows(1, start=18)[0], windows(1, start=25)[0]])
    assert m.skipped_windows == 6


def test_a_restarted_board_is_not_mistaken_for_stale_data():
    m = feed([HELLO, CONFIG, *windows(3), HELLO, CONFIG, *windows(3)])
    assert m.stale_lines == 0 and len(m.windows) == 6


def test_the_run_that_showed_the_problem_is_summarised_correctly(tmp_path, capsys):
    saved = tmp_path / "out.txt"
    stale = windows(1, start=15, us=16000)[0]
    real = [f"G,{w},{1000000 + 3750000 * w},{'rest' if w % 2 else 'tilt'},{16000 + w % 3 * 100}" for w in range(18, 63)]
    saved.write_text("\n".join([*[stale] * 300, *real]) + "\n")
    assert egm.main(["--replay", str(saved), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "ignored 299 repeated lines and 0 cut-off lines" in out and "46 windows." in out
    assert "took 16.2 ms" in out and "Things to look at" in out
    assert out.count("window ") == 0 or "  window " not in out                   # quiet: no line per window


def test_quiet_mode_prints_dots_and_a_progress_line_not_one_line_per_window(tmp_path, capsys):
    saved = tmp_path / "out.txt"
    saved.write_text("\n".join([CONFIG, *windows(12, "rest")]) + "\n")
    assert egm.main(["--replay", str(saved), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert out.count(".") >= 12 and ".......... 10 windows: rest 10" in out and "  window " not in out and "12 windows." in out


def test_statistics_lines_are_shown_as_they_arrive(tmp_path, capsys):
    saved = tmp_path / "out.txt"
    saved.write_text("\n".join([CONFIG, *windows(3), "# stats windows=3 overruns=2 i2c_err=0 restarts=0 ignored=0 max_compute_us=3100"]) + "\n")
    assert egm.main(["--replay", str(saved)]) == 0
    out = capsys.readouterr().out
    assert "stats: windows=3 overruns=2 i2c_err=0 restarts=0 ignored=0 max_compute_us=3100" in out and "overwrote 2 samples" in out


def test_cut_off_lines_inside_and_just_before_a_stale_flood_are_debris_not_corruption():
    line5 = windows(1, start=5)[0]
    m = feed([CONFIG, *windows(4), "G,5,100", "@@", line5, line5, "ge,5,1000000,rest", "G,5,12", line5, line5, windows(1, start=9)[0]])
    assert m.bad_lines == 0 and m.stale_fragments == 4 and m.stale_lines == 3
    assert not any("unreadable" in p for p in m.problems())


def test_unreadable_lines_after_the_flood_has_ended_are_still_counted():
    line5 = windows(1, start=5)[0]
    m = feed([CONFIG, *windows(4), line5, line5, line5, windows(1, start=9)[0], "@@@@", "G,10,12", *windows(1, start=11)])
    assert m.bad_lines == 2 and m.stale_fragments == 0 and any("2 unreadable" in p for p in m.problems())
