"""The microphone tools (tools/mic_capture.py, mic_eval.py, mic_plot.py): the parts that decide what data is trusted and how it is scored.
Built with AI assistance (Claude). All data here is synthetic and made on the spot; these tests prove the code paths, not model quality."""
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mic_capture as mc  # noqa: E402
import mic_eval as me  # noqa: E402
import mic_plot as mp  # noqa: E402


def line(w, rate=7887, **kw):
    d = dict(mean=8440, rms=145, peak=700, zcr=760, hf=109, clip=0)
    d.update(kw)
    return f"{w},{d['mean']},{d['rms']},{d['peak']},{d['zcr']},{d['hf']},{d['clip']},{rate}"


class FakeSerial:
    def __init__(self, lines):
        self.q = [ln + "\r\n" for ln in lines]

    def readline(self):
        return self.q.pop(0).encode() if self.q else b""


def capture(lines):
    t = itertools.count()
    return mc.capture(FakeSerial(lines), 5.0, clock=lambda: next(t) * 0.001)


HEAD = ["# mic_logger v1 fs_target=8000 window=2000 block=100 pin=29 adc=SE1 ref=2p0 avg=1 bits=14", "# w,mean,rms,peak,zcr,hf,clip,rate_hz"]


# ---------------------------------------------------------------- mic_capture
def test_parse_line_accepts_a_good_line_and_rejects_garbage_and_impossible_values():
    assert mc.parse_line(line(3))[0] == "row"
    assert mc.parse_line("748,108,0,7888") is None                    # the tail of a line, as Windows hands it over
    assert mc.parse_line(line(3, rms=99999)) is None                  # more than a 14-bit count cannot be real
    assert mc.parse_line(line(3, clip=5000)) is None                  # more clipped samples than a window holds
    assert mc.parse_line(HEAD[0])[0] == "meta"


def test_unreadable_lines_before_the_first_good_window_are_stale_and_later_ones_are_problems():
    r = capture(HEAD[1:] + ["748,108,0,7888"] * 30 + [line(w) for w in range(1, 11)] + ["garbage"] * 3 + [line(11)])
    assert r["stale_start"] == 30 and r["bad_lines"] == 3 and len(r["rows"]) == 11


def test_repeated_window_numbers_are_ignored_and_a_restart_is_counted():
    r = capture(HEAD + [line(1), line(2), line(3), line(3), line(3)] + [line(w) for w in range(4, 9)] + [line(1), line(2)])
    assert r["duplicates"] == 2 and r["restarts"] == 1


def test_start_up_slow_and_after_gap_windows_are_marked_invalid():
    rows = [mc.parse_line(line(w))[1] for w in list(range(1, 11)) + [12]]
    rows[5] = mc.parse_line(line(6, rate=7300))[1]
    info = mc.mark_valid(rows)
    why = {r["w"]: r["why"] for r in rows}
    assert why[1] == why[2] == "start-up" and why[6] == "rate" and why[12] == "after-gap" and info["gaps"] == 1
    assert why[3] == why[4] == ""


def test_a_saved_file_is_never_overwritten(tmp_path):
    r = capture(HEAD + [line(w) for w in range(1, 8)])
    now = datetime(2026, 10, 28, 12, 0, 0)
    paths = {mc.save_capture(r, label="tap", subject="s1", session="s9", note="", port="COM4", baud=115200,
                             out_dir=tmp_path, now=now)["path"].name for _ in range(3)}
    assert len(paths) == 3 and all((tmp_path / p).exists() for p in paths)


def test_an_empty_capture_says_so(tmp_path):
    assert not mc.save_capture(capture([]), label="quiet", subject="s1", session="s1", note="", port="COM4", baud=115200,
                               out_dir=tmp_path)["ok"]


# ---------------------------------------------------------------- mic_eval and mic_plot
def make_sessions(directory, levels, n=90, seed=0):
    rng = np.random.default_rng(seed)
    classes = {"quiet": ([88, 380, 1000, 100], [4, 40, 40, 2]), "speech": ([300, 1300, 450, 150], [150, 500, 200, 50]),
               "knock": ([800, 7500, 580, 330], [250, 1200, 80, 80])}
    for sess, k in levels.items():
        for label, (mu, sd) in classes.items():
            out = [f"# label={label}", f"# session={sess}", "w,mean,rms,peak,zcr,hf,clip,rate_hz,valid,why"]
            for w in range(1, n + 1):
                v = [max(1, rng.normal(m * k, s * k)) for m, s in zip(mu, sd)]
                out.append(f"{w},8440,{int(v[0])},{int(v[1])},{int(v[2])},{int(v[3])},0,7885,{0 if w <= 2 else 1},")
            (Path(directory) / f"{sess}_{label}_20261028_120000.csv").write_text("\n".join(out) + "\n")


LEVELS = {"s1": 1.6, "s2": 1.6, "s3": 1.6, "b": 0.95, "c": 1.0, "d": 1.25, "e": 1.2}


def test_a_session_cannot_be_in_both_train_and_test(tmp_path):
    make_sessions(tmp_path, LEVELS)
    with pytest.raises(SystemExit):
        me.main(["--dir", str(tmp_path), "--train", "s1,b", "--test", "b"])


def test_calibration_windows_are_dropped_unless_asked_not_to(tmp_path):
    make_sessions(tmp_path, {"b": 1.0})
    kept, _, _ = me.build(tmp_path, ["b"], False, drop_cal=False)
    dropped, _, _ = me.build(tmp_path, ["b"], False)
    assert len(kept) - len(dropped) == me.CAL_WINDOWS


def test_calibrated_features_are_relative_to_the_sessions_own_quiet_level(tmp_path):
    make_sessions(tmp_path, {"s1": 1.6, "b": 0.95})
    X, y, _ = me.build(tmp_path, ["s1", "b"], True)
    quiet_rms = X[y == "quiet", me.FEATURES.index("rms")]
    assert abs(np.median(quiet_rms) - 1.0) < 0.05                      # both sessions' quiet is about 1 whatever its level


def test_the_compare_run_prints_a_pass_or_fail(tmp_path, capsys):
    make_sessions(tmp_path, LEVELS)
    assert me.main(["--dir", str(tmp_path), "--train", "s1,s2,s3,b,c", "--test", "d,e", "--compare"]) == 0
    out = capsys.readouterr().out
    assert "UNCALIBRATED" in out and "CALIBRATED" in out and ("OVERALL: PASS" in out or "OVERALL: FAIL" in out)


def test_the_figure_tool_writes_both_figures(tmp_path):
    data, out = tmp_path / "mic", tmp_path / "fig"
    data.mkdir()
    make_sessions(data, LEVELS)
    assert mp.main(["--dir", str(data), "--out", str(out)]) == 0
    assert (out / "mic_quiet_levels.png").stat().st_size > 5000 and (out / "mic_heldout_f1.png").stat().st_size > 5000


def test_the_figure_tool_reports_missing_recordings(tmp_path):
    assert mp.main(["--dir", str(tmp_path), "--out", str(tmp_path / "fig")]) == 1


def test_power_states_are_parsed():
    assert mp.parse_power("d=battery,e=plugged_in") == {"d": "battery", "e": "plugged in"}
