"""Tests for the firmware's real main.c (firmware/sparkfun_edge/gesture_demo/main.c), run on the PC against a fake board.
The fake accelerometer replays SYNTHETIC or constructed samples; the fake timer ticks 1000 us per read. These tests check main.c's own
logic: its start-up text, one line per window, the timing figure, the error counters and the start-up failure path. They say nothing
about the chip's I2C, timer or UART."""
import re
import shutil

import numpy as np
import pytest

from tinyml_lab import edge_synth
from tinyml_lab import gesture_demo as gd
from tinyml_lab.edge_features_c import find_gcc

pytestmark = pytest.mark.skipif(find_gcc() is None, reason="no C compiler")

HFRC = 3_000_000
WRAP = 1 << 32


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic") / "recordings"
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=20, seed=0)
    clf, X, y, labels, ds = gd.train_deployable(d, "tree")
    md = tmp_path_factory.mktemp("model")
    gd.write_model(clf, labels, "raw", "tree", md, ds)
    return md, labels


@pytest.fixture(scope="module")
def sim(model, tmp_path_factory):
    return gd.build_host_sim(model[0], tmp_path_factory.mktemp("sim"))


@pytest.fixture(scope="module")
def pipe(model, tmp_path_factory):
    return gd.CPipeline(model[0], "float", tmp_path_factory.mktemp("pipe"))


def stream(seconds, seed=0, start=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * 51.2)
    rel = np.cumsum(np.rint((1 / 51.2) * (1 + rng.uniform(-0.08, 0.08, n)) * HFRC).astype(np.int64))
    rel -= rel[0]
    t = rel / HFRC
    xyz = np.column_stack([300 * np.sin(2 * np.pi * 1.3 * t), 200 * np.sin(2 * np.pi * 0.7 * t + 1) + rng.normal(0, 4, n), 1000 + 100 * np.sin(2 * np.pi * 3.1 * t)])
    return ((rel + start) % WRAP).astype(np.uint64), np.rint(xyz).astype(np.int64)


def g_lines(lines):
    return [ln.split(",") for ln in lines if ln.startswith("G,")]


def test_the_start_up_text_describes_the_build_and_the_columns(sim, model):
    ticks, xyz = stream(5)
    lines = gd.run_host_sim(sim, ticks, xyz)
    assert lines[0] == "# sparkfun_edge gesture demo v1, whoami 0x33"
    assert lines[1] == f"# odr_hz=50 tick_hz=3000000 timer=hfrc3mhz window=128 step=64 features=40 classes={','.join(model[1])}"
    assert lines[2] == "# columns: G,window,t_ticks,gesture,compute_us"


def test_the_sensor_and_timer_are_set_up_the_way_the_recordings_were_made(sim):
    ticks, xyz = stream(5)
    _, log = gd.run_host_sim(sim, ticks, xyz, log=True)
    setup = [ln for ln in log if not ln.startswith("push,")]
    assert setup == ["stimer,3", "stimer,4", "block_data_update,1", "data_rate,50Hz", "full_scale,2g", "operating_mode,hr12"]     # clear+freeze, then the 3 MHz oscillator


def test_every_sample_reaches_the_chain_exactly_as_the_sensor_gave_it(sim):
    ticks, xyz = stream(10, seed=7)
    flags = np.zeros(len(xyz), dtype=int)
    flags[40] = 2                                                     # this sample's status read fails: it must not reach the chain
    _, log = gd.run_host_sim(sim, ticks, xyz, flags, log=True)
    pushed = [tuple(int(v) for v in ln.split(",")[1:]) for ln in log if ln.startswith("push,")]
    expected = [(int(t), *(int(v) for v in row)) for i, (t, row) in enumerate(zip(ticks, xyz)) if i != 40]
    assert pushed == expected


def test_it_prints_one_line_per_window_with_the_gesture_the_chain_gives(sim, pipe, model):
    ticks, xyz = stream(40, seed=1)
    lines = gd.run_host_sim(sim, ticks, xyz)
    g = g_lines(lines)
    ref = pipe.run(ticks, xyz, HFRC)
    assert len(g) == ref["windows"] > 10
    assert [int(r[1]) for r in g] == list(range(1, len(g) + 1))
    assert [r[3] for r in g] == [model[1][c] for c in ref["classes"]]
    stamps = [int(r[2]) for r in g]
    assert stamps == sorted(stamps) and all(s < WRAP for s in stamps)


def test_the_time_it_reports_is_the_time_around_the_chain_call(sim):
    ticks, xyz = stream(30, seed=2)
    assert {r[4] for r in g_lines(gd.run_host_sim(sim, ticks, xyz))} == {"1000"}        # the fake timer moves 3000 ticks (1000 us) per read


def test_the_statistics_line_counts_what_went_wrong(sim):
    ticks, xyz = stream(40, seed=3)
    ticks, xyz = ticks.astype(np.int64), xyz.copy()
    flags = np.zeros(len(xyz), dtype=int)
    flags[100], flags[101], flags[250] = 1, 1, 1                      # three samples the sensor says it overwrote
    flags[150] = 2                                                    # one failed status read: that sample is lost
    ticks = np.insert(ticks, 200, ticks[199]).astype(np.uint64)       # a repeated timestamp
    xyz, flags = np.insert(xyz, 200, xyz[199], axis=0), np.insert(flags, 200, 0)
    keep = np.ones(len(ticks), dtype=bool)
    keep[500:560] = False                                             # a hole of about 1.2 s
    lines = gd.run_host_sim(sim, ticks[keep], xyz[keep], flags[keep])
    stats = [ln for ln in lines if ln.startswith("# stats")]
    assert stats, "no statistics line: fewer than 20 windows?"
    assert stats[0] == "# stats windows=20 overruns=3 i2c_err=1 restarts=1 ignored=1 max_compute_us=1000"


def test_a_quiet_clean_run_reports_no_problems(sim):
    ticks, xyz = stream(45, seed=4)
    stats = [ln for ln in gd.run_host_sim(sim, ticks, xyz) if ln.startswith("# stats")]
    assert stats and re.fullmatch(r"# stats windows=20 overruns=0 i2c_err=0 restarts=0 ignored=0 max_compute_us=1000", stats[0])


def test_the_32_bit_counter_wrapping_does_not_disturb_the_output(sim):
    ticks0, xyz = stream(30, seed=5, start=0)
    ticks1, _ = stream(30, seed=5, start=WRAP - 8 * HFRC)
    assert ticks1.min() < ticks1[0]
    a, b = g_lines(gd.run_host_sim(sim, ticks0, xyz)), g_lines(gd.run_host_sim(sim, ticks1, xyz))
    assert len(a) == len(b) > 5 and [r[3] for r in a] == [r[3] for r in b]


def test_a_missing_accelerometer_is_reported_and_nothing_else_is(sim):
    ticks, xyz = stream(5)
    lines = gd.run_host_sim(sim, ticks, xyz, whoami="0x00")
    assert lines and all(ln.startswith("# error: accelerometer init failed") for ln in lines)
    assert "whoami 0x00" in lines[0] and not g_lines(lines)


def test_the_simulated_firmware_builds_cleanly_with_the_committed_model_too(tmp_path):
    if not (gd.DEMO_DIR / "gesture_model.c").is_file():
        pytest.skip("no committed model")
    exe = gd.build_host_sim(gd.DEMO_DIR, tmp_path)
    ticks, xyz = stream(10, seed=6)
    assert g_lines(gd.run_host_sim(exe, ticks, xyz))


def test_a_stray_model_header_next_to_main_cannot_shadow_the_model_under_test(model, tmp_path, monkeypatch):
    fake = tmp_path / "demo"
    shutil.copytree(gd.DEMO_DIR, fake, ignore=shutil.ignore_patterns("__pycache__"))
    (fake / "gesture_model.h").write_text("#error stray header used\n")
    monkeypatch.setattr(gd, "DEMO_DIR", fake)
    exe = gd.build_host_sim(model[0], tmp_path / "w")
    assert exe.is_file()
