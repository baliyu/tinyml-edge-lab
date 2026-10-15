"""Tests for the on-device gesture chain (firmware/sparkfun_edge/gesture_demo): resample to 50 Hz, window, features, model.
All signals are SYNTHETIC or constructed (NOT real data): they test that the C chain does what the Python chain does and what
its own contract says, not how well gestures are recognised."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tinyml_lab import edge_gestures as eg
from tinyml_lab import edge_synth
from tinyml_lab import gesture_demo as gd
from tinyml_lab.edge_features_c import find_gcc

pytestmark = pytest.mark.skipif(find_gcc() is None, reason="no C compiler")

ROOT = Path(__file__).resolve().parents[1]
WRAP = 1 << 32
HFRC, XTAL = 3_000_000, 32_768
X_MEAN = eg.feature_names_for("raw").index("x_mean")


@pytest.fixture(scope="module")
def syn_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic") / "recordings"
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=20, seed=0)
    return d


@pytest.fixture(scope="module")
def tree(syn_dir, tmp_path_factory):
    clf, X, y, labels, ds = gd.train_deployable(syn_dir, "tree")
    md = tmp_path_factory.mktemp("tree_model")
    gd.write_model(clf, labels, "raw", "tree", md, ds)
    return clf, labels, md


@pytest.fixture(scope="module")
def c64(tree, tmp_path_factory):
    return gd.CPipeline(tree[2], "double", tmp_path_factory.mktemp("c64"))


@pytest.fixture(scope="module")
def c32(tree, tmp_path_factory):
    return gd.CPipeline(tree[2], "float", tmp_path_factory.mktemp("c32"))


def stream(seconds, tick_hz, rate=51.2, jitter=0.10, seed=0, start=0):
    """Irregular sample times (the sensor's rate, plus or minus jitter) as integer ticks, and integer milli-g from a mix of sines."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    dt = (1.0 / rate) * (1 + rng.uniform(-jitter, jitter, n))
    rel = np.cumsum(np.rint(dt * tick_hz).astype(np.int64))
    rel -= rel[0]
    t = rel / tick_hz
    xyz = np.column_stack([300 * np.sin(2 * np.pi * 1.3 * t) + 20 * np.sin(2 * np.pi * 7 * t),
                           200 * np.sin(2 * np.pi * 0.7 * t + 1) + rng.normal(0, 4, n),
                           1000 + 100 * np.sin(2 * np.pi * 3.1 * t)])
    return ((rel + start) % WRAP).astype(np.uint64), t, np.rint(xyz).astype(np.int64)


def run_and_reference(pipe, clf, ticks, t, xyz, tick_hz, skip=0):
    return pipe.run(ticks, xyz, tick_hz, skip), gd.python_chain(t, xyz, clf, "raw", skip)


def ramp_stream(seconds, tick_hz, start=0, seed=0):
    """Timestamps on whole milliseconds (19 to 21 ms apart) and x = the time in ms: exactly 1000 mg/s, so linear interpolation is
    exact and the grid sample k is exactly 20*k mg, whatever the jitter. y and z are constant."""
    rng = np.random.default_rng(seed)
    ms = np.cumsum(rng.integers(19, 22, int(seconds * 52)))
    ms -= ms[0]
    ticks = ((ms * (tick_hz // 1000) + start) % WRAP).astype(np.uint64)
    return ticks, np.column_stack([ms, np.zeros_like(ms), np.full_like(ms, 1000)]).astype(np.int64)


# ------------------------------------------------------------------------------------------------ against the Python chain
@pytest.mark.parametrize("tick_hz", [HFRC, XTAL])
def test_built_as_double_it_matches_the_python_chain_to_machine_precision(c64, tree, tick_hz):
    ticks, t, xyz = stream(40, tick_hz, seed=1)
    c, py = run_and_reference(c64, tree[0], ticks, t, xyz, tick_hz)
    r = gd.compare_chains(c, py, "raw")
    assert r["windows"] >= 20 and r["class_mismatches"] == 0
    assert r["max_error_vs_feature_size"] < 1e-9 and r["windows_with_a_different_dominant_bin"] == 0


@pytest.mark.parametrize("tick_hz", [HFRC, XTAL])
def test_built_as_float_it_matches_the_python_chain_closely(c32, tree, tick_hz):
    ticks, t, xyz = stream(40, tick_hz, seed=2)
    c, py = run_and_reference(c32, tree[0], ticks, t, xyz, tick_hz)
    r = gd.compare_chains(c, py, "raw")
    assert r["class_mismatches"] == 0 and r["max_error_vs_feature_size"] < 1e-4 and r["windows_with_a_different_dominant_bin"] == 0


def test_recorded_files_go_through_the_chain_like_the_python_loader(syn_dir, tree, tmp_path):
    r = gd.check_recordings(syn_dir, tree[0], tree[1], "raw", tree[2], "double", workdir=tmp_path)
    assert r["files"] == 15 and r["windows"] > 100 and r["class_mismatches"] == 0 and r["worst_feature_error"] < 1e-9


def test_the_chain_notices_when_the_models_answer_would_differ(c32, tree):
    clf, labels, md = tree
    ticks, t, xyz = stream(40, HFRC, seed=3)
    c, py = run_and_reference(c32, clf, ticks, t, xyz, HFRC)
    py = dict(py, classes=(py["classes"] + 1) % 5)
    assert gd.compare_chains(c, py, "raw")["class_mismatches"] == len(py["classes"])


# ------------------------------------------------------------------------------------------------ the contract, with known answers
@pytest.mark.parametrize("tick_hz", [HFRC, 32_000])
@pytest.mark.parametrize("start", [0, WRAP - 7 * HFRC // 3])
def test_a_linear_ramp_lands_exactly_on_the_grid_and_windows_hop_by_64(c64, tick_hz, start):
    ticks, xyz = ramp_stream(30, tick_hz, start=start)
    out = c64.run(ticks, xyz, tick_hz)
    k = np.arange(len(out["features"]))
    # window k holds grid samples 64k .. 64k+127, whose x values are 20*index: the mean is 20 * (64k + 63.5)
    assert len(k) >= 20
    assert np.allclose(out["features"][:, X_MEAN], 20 * (64 * k + 63.5), atol=1e-6)


def test_the_float_build_keeps_the_ramp_exact_enough(c32):
    ticks, xyz = ramp_stream(30, HFRC, seed=4)
    out = c32.run(ticks, xyz, HFRC)
    k = np.arange(len(out["features"]))
    assert np.allclose(out["features"][:, X_MEAN], 20 * (64 * k + 63.5), atol=0.5)


@pytest.mark.parametrize("tick_hz", [HFRC, XTAL])
def test_the_32_bit_counter_wrapping_changes_nothing(c64, tick_hz):
    ticks0, t, xyz = stream(40, tick_hz, seed=5, start=0)
    near_wrap = int(WRAP - 11 * tick_hz)                      # the counter wraps about 11 s into the recording
    ticks1, _, _ = stream(40, tick_hz, seed=5, start=near_wrap)
    assert ticks1.min() < ticks1[0]                           # it really did wrap
    a, b = c64.run(ticks0, xyz, tick_hz), c64.run(ticks1, xyz, tick_hz)
    assert a["windows"] == b["windows"] > 10 and b["restarts"] == 0
    assert np.array_equal(a["features"], b["features"]) and np.array_equal(a["classes"], b["classes"])


def test_a_long_recording_does_not_drift_against_the_grid(c64, tree):
    ticks, t, xyz = stream(600, HFRC, seed=6)                 # ten minutes
    out = c64.run(ticks, xyz, HFRC)
    n_grid = int(t[-1] * 50) + 1
    assert out["windows"] == (n_grid - 128) // 64 + 1
    py = gd.python_chain(t, xyz, tree[0], "raw")
    r = gd.compare_chains(out, py, "raw")
    assert r["class_mismatches"] == 0 and r["max_error_vs_feature_size"] < 1e-9


def test_skip_drops_the_start_up_samples(c64, tree):
    ticks, t, xyz = stream(30, HFRC, seed=7)
    for skip in (0, 37, 100):
        c, py = run_and_reference(c64, tree[0], ticks, t, xyz, HFRC, skip)
        r = gd.compare_chains(c, py, "raw")
        assert r["windows"] > 10 and r["class_mismatches"] == 0 and r["max_error_vs_feature_size"] < 1e-9


def test_a_short_gap_is_interpolated_over_like_the_python_chain(c64, tree):
    ticks, t, xyz = stream(30, HFRC, seed=8)
    keep = np.ones(len(t), dtype=bool)
    keep[300:306] = False                                     # six samples: about 0.12 s, inside the 0.24 s limit
    c = c64.run(ticks[keep], xyz[keep], HFRC)
    py = gd.python_chain(t[keep], xyz[keep], tree[0], "raw")
    assert c["restarts"] == 0
    assert gd.compare_chains(c, py, "raw")["class_mismatches"] == 0
    assert gd.compare_chains(c, py, "raw")["max_error_vs_feature_size"] < 1e-9


def test_a_long_gap_restarts_the_window_and_nothing_is_made_up(c64, tree):
    ticks, t, xyz = stream(40, HFRC, seed=9)
    cut = 800
    keep = np.ones(len(t), dtype=bool)
    keep[cut:cut + 60] = False                                # about 1.2 s with no samples
    c = c64.run(ticks[keep], xyz[keep], HFRC)
    first, second = slice(0, cut), slice(cut + 60, None)
    p1 = gd.python_chain(t[first], xyz[first], tree[0], "raw")
    p2 = gd.python_chain(t[second] - t[second][0], xyz[second], tree[0], "raw")      # the chain starts afresh on the first sample after the hole
    assert c["restarts"] == 1
    assert len(c["classes"]) == len(p1["classes"]) + len(p2["classes"])
    X = np.vstack([p1["features"], p2["features"]])
    assert gd.compare_chains(c, {"classes": np.concatenate([p1["classes"], p2["classes"]]), "features": X}, "raw")["max_error_vs_feature_size"] < 1e-9


def test_skip_applies_at_start_up_only_not_again_after_a_gap(c64, tree):
    ticks, t, xyz = stream(40, HFRC, seed=13)
    keep = np.ones(len(t), dtype=bool)
    keep[800:860] = False
    c = c64.run(ticks[keep], xyz[keep], HFRC, skip=50)
    p1 = gd.python_chain(t[:800], xyz[:800], tree[0], "raw", skip=50)
    p2 = gd.python_chain(t[860:] - t[860], xyz[860:], tree[0], "raw", skip=0)
    assert c["restarts"] == 1 and len(c["classes"]) == len(p1["classes"]) + len(p2["classes"])
    expected = np.vstack([p1["features"], p2["features"]])
    assert np.max(np.abs(c["features"] - expected) / np.maximum(np.abs(expected).max(axis=0), 1e-12)) < 1e-9


def test_time_running_backwards_is_treated_as_a_gap(c64):
    ticks, t, xyz = stream(20, HFRC, seed=10)
    ticks = ticks.copy()
    ticks[400:] = ticks[400:] - np.uint64(HFRC)               # the counter jumps back one second
    out = c64.run(ticks, xyz, HFRC)
    assert out["restarts"] == 1


def test_a_gap_whose_size_wraps_a_32_bit_product_is_still_a_gap(c64):
    ticks, t, xyz = stream(20, HFRC, seed=14)
    ticks = ticks.astype(np.int64)
    ticks[400:] += 85_899_346 - (ticks[400] - ticks[399])      # 28.6 s: the hole times 50 is 2**32 + 4, which a 32-bit product would read as 4
    out = c64.run(ticks.astype(np.uint64) % np.uint64(WRAP), xyz, HFRC)
    assert out["restarts"] == 1


def test_a_repeated_timestamp_is_ignored(c64):
    ticks, t, xyz = stream(20, HFRC, seed=11)
    dup_t, dup_x = np.insert(ticks, 201, ticks[200]), np.insert(xyz, 201, xyz[200] + 5, axis=0)      # a second sample with the same stamp, after the first
    a, b = c64.run(ticks, xyz, HFRC), c64.run(dup_t, dup_x, HFRC)
    assert b["ignored"] == 1 and a["ignored"] == 0
    assert np.array_equal(a["features"], b["features"])


def test_a_stuck_clock_never_fills_a_window(c64):
    out = c64.run(np.full(500, 1234, dtype=np.uint64), np.zeros((500, 3), dtype=np.int64), HFRC)
    assert out["windows"] == 0 and out["ignored"] == 499


def test_the_chain_refuses_bad_settings(c64, tmp_path):
    import ctypes
    state = ctypes.create_string_buffer(int(c64.lib.gp_state_size()))
    assert c64.lib.gp_init(state, 0, 0) == -1
    assert c64.lib.gp_init(state, 100_000_001, 0) == -1
    assert c64.lib.gp_init(None, 1000, 0) == -1
    with pytest.raises(FileNotFoundError):
        gd.CPipeline(tmp_path)


# ------------------------------------------------------------------------------------------------ the model that goes in it
def test_the_forest_model_in_the_chain_answers_like_scikit_learn(syn_dir, tmp_path):
    clf, X, y, labels, ds = gd.train_deployable(syn_dir, "forest")
    gd.write_model(clf, labels, "raw", "forest", tmp_path / "m", ds)
    r = gd.check_recordings(syn_dir, clf, labels, "raw", tmp_path / "m", "float", workdir=tmp_path / "w")
    assert r["files"] == 15 and r["windows"] > 100 and r["class_mismatches"] == 0


def test_a_model_on_motion_features_goes_through_the_motion_feature_code(syn_dir, tmp_path):
    clf, X, y, labels, ds = gd.train_deployable(syn_dir, "tree", "motion")
    gd.write_model(clf, labels, "motion", "tree", tmp_path / "m", ds)
    h = (tmp_path / "m" / "gesture_model.h").read_text()
    assert "GESTURE_USES_MOTION_FEATURES 1" in h and "GESTURE_N_FEATURES 37" in h
    r = gd.check_recordings(syn_dir, clf, labels, "motion", tmp_path / "m", "double", workdir=tmp_path / "w")
    assert r["class_mismatches"] == 0 and r["worst_feature_error"] < 1e-9


def test_the_exported_files_describe_the_model(tree):
    clf, labels, md = tree
    meta = json.loads((md / "gesture_model.json").read_text())
    assert meta["classes"] == labels and meta["feature_set"] == "raw" and len(meta["feature_names"]) == 40
    assert meta["window_samples"] == 128 and meta["step_samples"] == 64 and "no held-out score" in meta["note"]
    c, h = (md / "gesture_model.c").read_text(), (md / "gesture_model.h").read_text()
    assert "GESTURE_N_CLASSES 5" in h and "do not edit by hand" in c and '"shake_lr"' in c


def test_a_stray_model_header_next_to_the_chain_cannot_shadow_the_model_under_test(syn_dir, tmp_path, monkeypatch):
    # the committed model lives next to gesture_pipeline.h; a build of some other model must not pick up its header instead
    fake_demo = tmp_path / "demo"
    fake_demo.mkdir()
    for name in ("gesture_pipeline.c", "gesture_pipeline.h"):
        shutil.copy(gd.DEMO_DIR / name, fake_demo / name)
    (fake_demo / "gesture_model.h").write_text("#define GESTURE_N_FEATURES 3\n#define GESTURE_USES_MOTION_FEATURES 0\n#error stray header used\n")
    monkeypatch.setattr(gd, "DEMO_DIR", fake_demo)
    clf, X, y, labels, ds = gd.train_deployable(syn_dir, "tree", "motion")
    gd.write_model(clf, labels, "motion", "tree", tmp_path / "m", ds)
    pipe = gd.CPipeline(tmp_path / "m", "double", tmp_path / "w")
    assert pipe.n_features == 37
    if shutil.which("arm-none-eabi-gcc"):
        assert gd.arm_footprint(tmp_path / "m", tmp_path / "arm")["flash_bytes"]["total"] > 0


def test_the_model_writer_rejects_an_unknown_kind(tree):
    with pytest.raises(ValueError):
        gd.model_sources(tree[0], tree[1], "raw", "svm")


def test_the_committed_model_if_there_is_one_builds_and_is_consistent(tmp_path):
    md = gd.DEMO_DIR
    if not (md / "gesture_model.c").is_file():
        pytest.skip("no committed model yet (scripts/export_gesture_model.py writes it)")
    pipe = gd.CPipeline(md, "float", tmp_path)
    meta = json.loads((md / "gesture_model.json").read_text())
    assert pipe.n_features == len(meta["feature_names"]) and meta["note"].startswith("trained on every session")
    ticks, t, xyz = stream(20, HFRC, seed=12)
    assert pipe.run(ticks, xyz, HFRC)["windows"] > 5


# ------------------------------------------------------------------------------------------------ what it costs on the board
@pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="no ARM compiler")
def test_the_board_build_is_small_uses_little_ram_and_needs_no_double_arithmetic(tree, tmp_path):
    f = gd.arm_footprint(tree[2], tmp_path)
    assert f["flash_bytes"]["pipeline"] < 1000 and f["flash_bytes"]["pipeline"] + f["flash_bytes"]["features"] < 2500
    assert f["state_ram_bytes"] < 2000 and f["deepest_stack_estimate_bytes"] < 4000
    assert set(f["undefined_symbols"]) <= {"ef_raw_features", "gesture_model_predict", "memcpy"}      # no soft-double helper, no libm
    assert f["stack_bytes"]["gp_push"] < 100 and f["stack_bytes"]["gp_emit"] < 300


# ------------------------------------------------------------------------------------------------ the scripts
def test_the_check_script_runs_and_writes_its_report(syn_dir, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_gesture_pipeline.py"), "--data", str(syn_dir), "--model", "tree", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "SYNTHETIC DATA, NOT REAL RECORDINGS" in r.stdout
    rep = json.loads((tmp_path / "gesture_pipeline_check.json").read_text())
    assert rep["data"]["synthetic"] is True and rep["builds"]["float"]["class_mismatches"] == 0 and "no" not in rep["note"][:2]


def test_the_export_script_refuses_synthetic_data_for_the_committed_model(syn_dir, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_gesture_model.py"), "--data", str(syn_dir)], capture_output=True, text=True)
    assert r.returncode == 1 and "SYNTHETIC" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_gesture_model.py"), "--data", str(syn_dir), "--allow-synthetic",
                        "--out", str(gd.DEMO_DIR)], capture_output=True, text=True)
    assert r.returncode == 1                                       # not even with the flag, if the target is the committed model
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_gesture_model.py"), "--data", str(syn_dir), "--allow-synthetic", "--model", "tree",
                        "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "gesture_model.c").is_file() and "SYNTHETIC dry run" in r.stdout
