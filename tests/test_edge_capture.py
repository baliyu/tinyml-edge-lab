import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "edge_capture.py"
spec = importlib.util.spec_from_file_location("edge_capture", TOOL)
ec = importlib.util.module_from_spec(spec)
sys.modules["edge_capture"] = ec
spec.loader.exec_module(ec)

TICK_HZ, ODR = 32768, 50


def ticks_for(n, hz=TICK_HZ, odr=ODR, start=1000, skip=()):
    """Timestamps of a perfect stream, optionally with some samples missing."""
    return [(start + round(i * hz / odr)) % ec.WRAP for i in range(n) if i not in skip]


# ---------------------------------------------------------------- parsing
def test_parse_a_normal_and_a_negative_sample():
    assert ec.parse_line("12345678,-541,507,544\r\n") == ("sample", 12345678, -541, 507, 544)
    assert ec.parse_line("0,0,0,-1") == ("sample", 0, 0, 0, -1)


@pytest.mark.parametrize("bad", ["", "   ", "t_ticks,x_mg,y_mg,z_mg", "12345,10,20",                 # header, truncated
                                 "Acc [mg] 202.0 x, -382.mg] 120.0 x, -356.0 y",                      # the garbled line from the first run
                                 "123,45,67,89,10", "abc,1,2,3", "4294967296,1,2,3", "100,9000,1,1"])  # too many fields, junk, out of range
def test_garbage_is_rejected(bad):
    assert ec.parse_line(bad) is None


def test_meta_stats_and_comment_lines():
    assert ec.parse_line("# odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k") == \
        ("meta", {"odr_hz": 50, "tick_hz": 32768, "timer": "xtal32k"})
    kind, st = ec.parse_line("# stats samples=500 overruns=2 i2c_err=1 rate_hz=49.987 tick_hz=32768 timer=xtal32k")
    assert kind == "stats" and st == {"samples": 500, "overruns": 2, "i2c_err": 1, "rate_hz": 49.987, "tick_hz": 32768, "timer": "xtal32k"}
    assert ec.parse_line("# sparkfun_edge lis2dh12 logger v1, whoami 0x33")[0] == "comment"


# ---------------------------------------------------------------- timestamps
def test_unwrap_follows_the_32_bit_counter_across_its_wrap():
    near = ec.WRAP - 1000
    un = ec.unwrap_ticks([near, near + 600, (near + 1200) % ec.WRAP, (near + 1800) % ec.WRAP])
    assert un == [0, 600, 1200, 1800]


def test_a_perfect_50hz_stream_measures_50hz_with_no_gaps():
    s = ec.summarise(ec.unwrap_ticks(ticks_for(1000)), TICK_HZ, ODR)
    assert s["n"] == 1000 and s["gaps"] == 0 and s["missing_est"] == 0
    assert s["rate_hz"] == pytest.approx(50.0, rel=1e-3)
    assert ec.verdict(s, ODR, [], 0) == []


def test_a_dropped_sample_is_found_and_counted():
    s = ec.summarise(ec.unwrap_ticks(ticks_for(1000, skip={300, 700})), TICK_HZ, ODR)
    assert s["gaps"] == 2 and s["missing_est"] == 2
    assert any("gap" in p for p in ec.verdict(s, ODR, [], 0))


def test_a_wrong_rate_is_flagged_even_when_no_sample_is_missing():
    slow = ec.summarise(ec.unwrap_ticks(ticks_for(1000, odr=47)), TICK_HZ, ODR)          # 47 Hz where 50 was configured (6% off)
    assert abs(slow["rate_hz"] - 47) < 0.1
    assert any("differs from the configured" in p and "5%" in p for p in ec.verdict(slow, ODR, [], 0))


def test_new_overruns_and_i2c_errors_and_corruption_are_reported():
    s = ec.summarise(ec.unwrap_ticks(ticks_for(1000)), TICK_HZ, ODR)
    stats = [{"overruns": 0, "i2c_err": 0}, {"overruns": 3, "i2c_err": 2}]
    problems = ec.verdict(s, ODR, stats, bad_lines=50)
    assert any("3 new overrun" in p for p in problems) and any("2 new I2C" in p for p in problems) and any("unreadable" in p for p in problems)


def test_a_startup_overrun_that_does_not_grow_is_not_a_problem():
    """Your first run showed 'overruns=1' once. The counter is cumulative since power-up, so only growth during a capture counts."""
    s = ec.summarise(ec.unwrap_ticks(ticks_for(1000)), TICK_HZ, ODR)
    assert ec.verdict(s, ODR, [{"overruns": 1, "i2c_err": 0}, {"overruns": 1, "i2c_err": 0}, {"overruns": 1, "i2c_err": 0}], 0) == []
    assert ec.verdict(s, ODR, [{"overruns": 1, "i2c_err": 0}], 0) == []          # a single line cannot show a change
    assert ec.change_during_capture([{"overruns": 1}], "overruns") is None
    assert ec.change_during_capture([{"overruns": 1}, {"overruns": 4}], "overruns") == 3


def test_a_sensor_running_2_4_percent_fast_is_information_not_a_failure():
    """Your first run: 51.220 Hz measured for a configured 50 Hz."""
    fast = ec.summarise(ec.unwrap_ticks(ticks_for(1000, odr=51.22)), TICK_HZ, ODR)
    assert fast["rate_hz"] == pytest.approx(51.22, abs=0.02)
    assert ec.verdict(fast, ODR, [], 0) == []                                     # not a problem...
    assert "51.2" in ec.rate_note(fast, ODR) and "resample" in ec.rate_note(fast, ODR)   # ...but it is reported
    exact = ec.summarise(ec.unwrap_ticks(ticks_for(1000)), TICK_HZ, ODR)
    assert ec.rate_note(exact, ODR) == ""


def test_resampling_puts_a_51hz_recording_on_an_exact_50hz_grid():
    n, rate = 600, 51.22
    t = [i / rate for i in range(n)]
    vals = [(1000.0 * tt, -500.0 * tt, 1042.0) for tt in t]                        # a straight ramp, so interpolation is exact
    grid, out = ec.resample_uniform(t, vals, 50)
    assert grid[0] == 0.0 and all(abs((b - a) - 0.02) < 1e-9 for a, b in zip(grid, grid[1:]))
    assert len(grid) == int(t[-1] * 50) + 1
    for g, v in zip(grid, out):
        assert v[0] == pytest.approx(1000.0 * g, abs=1e-6) and v[1] == pytest.approx(-500.0 * g, abs=1e-6) and v[2] == pytest.approx(1042.0)


def test_resampling_needs_two_points():
    assert ec.resample_uniform([0.0], [(1, 2, 3)], 50) == ([], [])


def test_too_few_samples_is_not_ok():
    assert ec.verdict(ec.summarise([0], TICK_HZ, ODR), ODR, [], 0) == ["fewer than two samples were captured"]


# ---------------------------------------------------------------- labels and the whole capture
def test_labels_become_safe_file_names():
    assert ec.safe_label("Walking Upstairs!") == "walking_upstairs"
    assert ec.safe_label("../../etc") == "etc"
    with pytest.raises(ValueError):
        ec.safe_label("???")


class FakeSerial:
    """Plays back lines like the firmware prints them, with one chunk of garbage and a statistics line."""
    def __init__(self, lines):
        self.lines, self.i = lines, 0
        self.t = 0.0

    def readline(self):
        self.t += 0.02                      # time always moves on, as with a real port that times out
        if self.i >= len(self.lines):
            return b""
        line = self.lines[self.i]; self.i += 1
        return (line + "\r\n").encode()


def firmware_lines(n, skip=()):
    out = ["# sparkfun_edge lis2dh12 logger v1, whoami 0x33",
           f"# odr_hz={ODR} fs_g=2 mode=hr12 units=mg tick_hz={TICK_HZ} timer=xtal32k", "t_ticks,x_mg,y_mg,z_mg"]
    for i, t in enumerate(ticks_for(n, skip=skip)):
        out.append(f"{t},{i % 50},{-i % 70},{1000 + i % 9}")
        if i == 199:
            out.append("Acc [mg] 202.0 x, -382.mg] 120.0 x")                     # garbage
            out.append(f"# stats samples=500 overruns=0 i2c_err=0 rate_hz=50.000 tick_hz={TICK_HZ} timer=xtal32k")
    return out


def run_capture(lines, tmp_path, monkeypatch, seconds=3600):
    ser = FakeSerial(lines)
    clock = lambda: ser.t
    result = ec.capture(ser, seconds=seconds, clock=clock)
    ticks = ec.unwrap_ticks([s[0] for s in result["samples"]])
    info = result["meta"]
    summary = ec.summarise(ticks, info["tick_hz"], info["odr_hz"])
    problems = ec.verdict(summary, info["odr_hz"], result["stats"], result["bad_lines"])
    out = tmp_path / "c.csv"
    ec.write_csv(out, result, "walking", "s1", "waist", "COM4", 115200, info["odr_hz"], info["tick_hz"], info["timer"], summary, problems)
    return result, summary, problems, out.read_text().splitlines()


def test_end_to_end_capture_file_and_counts(tmp_path, monkeypatch):
    result, summary, problems, text = run_capture(firmware_lines(1000), tmp_path, monkeypatch, seconds=60)
    assert len(result["samples"]) == 1000 and result["bad_lines"] == 1 and len(result["stats"]) == 1
    assert problems == []
    assert "# label=walking" in text and "# verdict=OK" in text and "# rate_note=none" in text and "t_s,x_mg,y_mg,z_mg" in text
    body = text[text.index("t_s,x_mg,y_mg,z_mg") + 1:]
    assert len(body) == 1000 and body[0].startswith("0.000000,")
    t = [float(r.split(",")[0]) for r in body]
    assert all(b > a for a, b in zip(t, t[1:])) and t[-1] == pytest.approx(999 / 50, rel=1e-3)


def test_a_dropped_sample_shows_up_in_the_saved_file(tmp_path, monkeypatch):
    _, summary, problems, text = run_capture(firmware_lines(1000, skip={400}), tmp_path, monkeypatch, seconds=60)
    assert summary["gaps"] == 1 and problems and any("# verdict=PROBLEMS" in line for line in text)


def test_the_header_line_is_not_counted_as_an_error():
    ser = FakeSerial(["t_ticks,x_mg,y_mg,z_mg"] + firmware_lines(10)[2:])
    result = ec.capture(ser, seconds=60, clock=lambda: ser.t)
    assert result["bad_lines"] == 0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- stale data handed over when the port is opened
def _live_lines(n, start_tick, hz=51.22, tick_hz=32768):
    return [f"{(start_tick + round(i * tick_hz / hz)) % ec.WRAP},{50 + i % 3},{-80 + i % 2},{1042 + i % 4}" for i in range(n)]


def _stats(tick_hz=32768):
    return f"# stats samples=500 overruns=0 i2c_err=0 rate_hz=51.220 tick_hz={tick_hz} timer=xtal32k"


def _records(ticks, tick_hz=32768):
    return [(t, 0, 0, 0) for t in ticks]


def test_the_exact_failure_from_the_first_check_recording_is_recovered(tmp_path):
    """227 copies of one early line, then the live stream 27,672 s of uptime later (996 samples, 'duration 27686.9 s')."""
    stale = ["3456,50,-80,1042"] * 227
    lines = stale + _live_lines(500, 27672 * 32768) + [_stats()] + _live_lines(269, 27672 * 32768 + round(500 * 32768 / 51.22))
    ser = FakeSerial(lines)
    result = ec.capture(ser, seconds=60, clock=lambda: ser.t)
    assert len(result["samples"]) == 996
    r = ec.save_capture(result, label="check", subject="me", session="", note="", port="COM4", baud=115200, out_dir=tmp_path,
                        now=datetime(2026, 10, 11, 17, 16, 27))
    assert r["ok"] and r["stale_dropped"] == 227 and r["problems"] == []
    assert r["summary"]["n"] == 769 and r["summary"]["duration_s"] == pytest.approx(15.0, abs=0.1)
    assert r["summary"]["rate_hz"] == pytest.approx(51.22, abs=0.05) and r["summary"]["gaps"] == 0
    text = r["path"].read_text()
    assert "# stale_start_dropped=227" in text and "# verdict=OK" in text
    lines_ = text.splitlines()
    body = lines_[lines_.index("t_s,x_mg,y_mg,z_mg") + 1:]
    assert len(body) == 769 and body[0].startswith("0.000000,") and float(body[-1].split(",")[0]) == pytest.approx(15.0, abs=0.1)
    assert r["n_resampled"] == pytest.approx(750, abs=2)


def test_a_clean_stream_loses_nothing():
    samples = _records(ec.unwrap_ticks([1000 + round(i * 32768 / 50) for i in range(600)]))
    kept, dropped = ec.drop_stale_start(samples, 32768, 50)
    assert dropped == 0 and kept == samples


def test_a_stale_replay_with_normal_spacing_is_found_by_its_jump():
    old = [5000 + round(i * 32768 / 50) for i in range(100)]                     # old data that looks like a healthy stream
    live = [900_000_000 + round(i * 32768 / 50) for i in range(600)]
    kept, dropped = ec.drop_stale_start(_records(old + live), 32768, 50)
    assert dropped == 100 and len(kept) == 600 and kept[0][0] == 900_000_000


def test_a_counter_wrap_in_the_live_stream_is_not_mistaken_for_stale_data():
    live = [(ec.WRAP - 5000 + round(i * 32768 / 50)) % ec.WRAP for i in range(400)]     # wraps part-way through
    kept, dropped = ec.drop_stale_start(_records(["x"] and live), 32768, 50)
    assert dropped == 0 and len(kept) == 400
    stale = [3456] * 60
    kept, dropped = ec.drop_stale_start(_records(stale + live), 32768, 50)
    assert dropped == 60 and len(kept) == 400


def test_a_gap_later_in_the_recording_is_still_reported(tmp_path):
    ticks = [1000 + round(i * 32768 / 50) for i in range(700)]
    ticks = ticks[:500] + [t + 3000 for t in ticks[500:]]                                  # a 90 ms hiccup after sample 500
    kept, dropped = ec.drop_stale_start(_records(ticks), 32768, 50)
    assert dropped == 0 and len(kept) == 700
    s = ec.summarise(ec.unwrap_ticks([k[0] for k in kept]), 32768, 50)
    assert s["gaps"] == 1 and any("gap" in p for p in ec.verdict(s, 50, [], 0))


def test_a_hiccup_in_the_first_second_costs_only_the_samples_before_it_and_is_counted():
    ticks = [1000 + round(i * 32768 / 50) for i in range(700)]
    ticks = ticks[:10] + [t + 3000 for t in ticks[10:]]
    kept, dropped = ec.drop_stale_start(_records(ticks), 32768, 50)
    assert dropped == 10 and len(kept) == 690


def test_a_stream_that_never_settles_is_left_for_the_verdict_to_judge():
    rng = np.random.default_rng(0)
    ticks = list(np.cumsum(rng.choice([0, 5, 20000, 650], size=300)))
    samples = _records([int(t) for t in ticks])
    kept, dropped = ec.drop_stale_start(samples, 32768, 50)
    assert dropped == 0 and kept == samples


def test_very_short_captures_are_left_alone():
    samples = _records([100, 100, 100, 99999999, 99999999])
    assert ec.drop_stale_start(samples, 32768, 50) == (samples, 0)
    assert ec.drop_stale_start([], 32768, 50) == ([], 0)


def test_the_session_recorder_reports_ignored_stale_samples(tmp_path):
    rs = _load("record_session")

    class Stream(FakeSerial):
        def __init__(self):
            super().__init__([])

        def reset_input_buffer(self):
            self.lines = (["3456,50,-80,1042"] * 227 + ["# odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k"]
                          + _live_lines(800, 27672 * 32768))
            self.i, self.t = 0, 0.0

    ser, log = Stream(), []
    answers = iter(["", ""])
    kept = rs.run_session(ser, session="s1", subject="me", labels=["rest"], seconds=20, note="", port="COM4", baud=115200,
                          out_dir=tmp_path, out=log.append, sleep_fn=lambda s: None, clock=lambda: ser.t,
                          input_fn=lambda prompt: next(answers))
    assert kept["rest"]["problems"] == [] and kept["rest"]["stale_dropped"] == 227
    assert any("ignored 227 stale samples" in line for line in log) and any("over 15.6 s" in line or "over 1" in line for line in log)
