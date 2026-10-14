"""Tests for the desk-gesture pipeline. All recordings used here are SYNTHETIC (NOT real data) or made by the real
capture tool from simulated firmware lines; they test the code, not whether gestures can be recognised."""
import ctypes
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tinyml_lab import edge_gesture_eval as ev
from tinyml_lab import edge_gesture_report as rp
from tinyml_lab import edge_gestures as eg
from tinyml_lab import edge_synth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_edge_capture as tec  # noqa: E402  (FakeSerial and firmware_lines from the capture tool's tests)


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ec = _load_tool("edge_capture")
rs = _load_tool("record_session")


@pytest.fixture(scope="module")
def syn_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_gestures")
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=24, seed=0)
    return d


@pytest.fixture(scope="module")
def syn_report(syn_dir, tmp_path_factory):
    return ev.run_gestures(syn_dir, out_dir=tmp_path_factory.mktemp("res"), log=lambda *_: None)


# ---------------------------------------------------------------- reading capture files
def _real_tool_file(tmp_path, label="shake_lr", session="s2", note="board flat in my palm", crlf=False):
    """A file written by the REAL capture tool from simulated firmware lines."""
    ser = tec.FakeSerial(tec.firmware_lines(800))
    result = ec.capture(ser, seconds=60, clock=lambda: ser.t)
    r = ec.save_capture(result, label=label, subject="me", session=session, note=note, port="COM4", baud=115200,
                        out_dir=tmp_path, now=datetime(2026, 10, 12, 9, 30, 0))
    assert r["ok"]
    if crlf:     # what Windows Python writes
        r["path"].write_bytes(r["path"].read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    return r["path"]


@pytest.mark.parametrize("crlf", [False, True])
def test_reads_a_file_written_by_the_real_capture_tool(tmp_path, crlf):
    cap = eg.read_capture(_real_tool_file(tmp_path, crlf=crlf))
    assert (cap.label, cap.session, cap.subject) == ("shake_lr", "s2", "me")
    assert cap.note == "board flat in my palm"          # free text with spaces survives
    assert cap.odr_hz == 50 and cap.tick_hz == 32768 and cap.timer == "xtal32k"
    assert cap.verdict == "OK" and not cap.synthetic
    assert cap.measured_rate_hz == pytest.approx(50.0, abs=0.1)
    assert cap.xyz_mg.shape == (800, 3) and cap.t_s[0] == 0.0 and np.all(np.diff(cap.t_s) > 0)


def test_the_capture_file_name_carries_the_session(tmp_path):
    assert _real_tool_file(tmp_path).name == "s2_shake_lr_20261012_093000.csv"


@pytest.mark.parametrize("body, message", [
    ("# label=x\nt_s,x_mg,y_mg\n0,1,2\n", "expected the header"),
    ("# label=x\nt_s,x_mg,y_mg,z_mg\n0.0,1,2\n0.02,1,2,3\n", ":3: expected 4 comma-separated values"),
    ("# label=x\nt_s,x_mg,y_mg,z_mg\n0.0,1,2,3\n0.02,a,2,3\n", ":4: not numbers"),
    ("# label=x\nt_s,x_mg,y_mg,z_mg\n0.0,1,2,3\n0.0,1,2,3\n", "not strictly increasing"),
    ("# label=x\nt_s,x_mg,y_mg,z_mg\n0.0,1,2,3\n", "fewer than two samples"),
    ("# label=x\n0.0,1,2,3\n0.1,1,2,3\n", "expected the header"),
    ("# label=x\n", "no 't_s,x_mg,y_mg,z_mg' header"),
])
def test_malformed_files_are_rejected_with_a_reason(tmp_path, body, message):
    f = tmp_path / "bad.csv"
    f.write_text(body)
    with pytest.raises(ValueError, match=re.escape(message)):
        eg.read_capture(f)


# ---------------------------------------------------------------- resampling, trimming, windows
def test_resampling_lands_on_an_exact_grid_and_follows_the_signal():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.018, 0.022, 700))            # irregular, about 50 Hz
    t -= t[0]
    xyz = np.column_stack([3 * t, -2 * t + 5, np.full_like(t, 1000.0)])      # straight lines: interpolation is exact
    out = eg.resample_to_grid(t, xyz, 50.0)
    grid = np.arange(len(out)) / 50.0
    assert len(out) == int(t[-1] * 50) + 1 and grid[-1] <= t[-1]
    assert np.allclose(out[:, 0], 3 * grid) and np.allclose(out[:, 1], -2 * grid + 5) and np.allclose(out[:, 2], 1000.0)


def test_resampling_a_51hz_recording_gives_50_samples_per_second():
    n = 2000
    t = np.arange(n) / 51.22
    out = eg.resample_to_grid(t, np.zeros((n, 3)), 50.0)
    assert abs(len(out) - (t[-1] * 50 + 1)) < 1.01


def test_numpy_resampler_agrees_with_the_capture_tools_own():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.017, 0.023, 500)); t -= t[0]
    vals = rng.normal(0, 300, (500, 3))
    grid, tool_out = ec.resample_uniform(list(t), [tuple(v) for v in vals], 50)
    mine = eg.resample_to_grid(t, vals, 50.0)
    assert len(grid) == len(mine) and np.allclose(np.array(tool_out), mine)


def test_trimming_drops_the_same_time_at_both_ends():
    x = np.arange(1000)[:, None] * np.ones((1, 3))
    y = eg.trim_edges(x, 50.0, 2.0)
    assert len(y) == 800 and y[0, 0] == 100 and y[-1, 0] == 899
    assert len(eg.trim_edges(x, 50.0, 0.0)) == 1000


def test_windows_have_the_right_count_and_content():
    x = np.arange(1000)[:, None] * np.ones((1, 3))
    w = eg.make_windows(x, 128, 64)
    assert w.shape == ((1000 - 128) // 64 + 1, 128, 3)
    assert np.array_equal(w[0][:, 0], np.arange(0, 128)) and np.array_equal(w[1][:, 0], np.arange(64, 192))
    assert w[-1][-1, 0] <= 999
    assert eg.make_windows(x[:100], 128, 64).shape == (0, 128, 3)


# ---------------------------------------------------------------- loading a folder
def test_a_synthetic_folder_loads_with_the_expected_shape(syn_dir):
    ds = eg.load_dataset(syn_dir)
    assert ds.synthetic and ds.windows.shape[1:] == (128, 3) and len(ds.files) == 15
    assert sorted(set(ds.y.tolist())) == sorted(eg.GESTURES) and sorted(set(ds.session.tolist())) == ["s1", "s2", "s3"]
    per_file = {f["file"]: f["windows"] for f in ds.files}
    assert sum(per_file.values()) == len(ds.y)
    for name, n in per_file.items():                                   # every window belongs to exactly one file
        assert int(np.sum(ds.file == name)) == n


def test_the_capture_tools_resampled_copy_is_not_loaded_twice(tmp_path):
    f = _real_tool_file(tmp_path)
    assert (tmp_path / (f.stem + "_50hz.csv")).exists()
    ds = eg.load_dataset(tmp_path, trim_s=0.5)
    assert [x["file"] for x in ds.files] == [f.name]
    assert ds.skipped == []                      # the copy is not even reported as a skipped file


def test_unusable_files_are_skipped_with_a_reason(tmp_path):
    good = _real_tool_file(tmp_path)
    nosess = tmp_path / "nosession.csv"
    nosess.write_text(good.read_text().replace("# session=s2", "# session="))
    flagged = tmp_path / "flagged.csv"
    flagged.write_text(good.read_text().replace("# verdict=OK", "# verdict=PROBLEMS: 3 gap(s) in the timestamps"))
    short = tmp_path / "short.csv"
    short.write_text("# label=x\n# session=s1\n# verdict=OK\nt_s,x_mg,y_mg,z_mg\n" + "".join(f"{i/50:.3f},1,2,3\n" for i in range(150)))
    broken = tmp_path / "broken.csv"
    broken.write_text("# label=x\nt_s,x_mg,y_mg,z_mg\n0,1\n")
    ds = eg.load_dataset(tmp_path, trim_s=0.5)
    reasons = dict(ds.skipped)
    assert "no session" in reasons["nosession.csv"] and "flagged it" in reasons["flagged.csv"]
    assert "too short" in reasons["short.csv"] and "broken.csv:3: expected 4 comma-separated values" in reasons["broken.csv"]
    assert [f["file"] for f in ds.files] == [good.name]
    assert {f["file"] for f in eg.load_dataset(tmp_path, trim_s=0.5, allow_problems=True).files} == {good.name, "flagged.csv"}


def test_synthetic_and_real_recordings_are_never_mixed(tmp_path):
    _real_tool_file(tmp_path)
    edge_synth.write_synthetic_gestures(tmp_path, sessions=1, seconds=10, labels=("rest",))
    with pytest.raises(ValueError, match="mixed"):
        eg.load_dataset(tmp_path, trim_s=0.5)


def test_a_missing_or_empty_folder_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="record_session"):
        eg.load_dataset(tmp_path / "nope")
    with pytest.raises(ValueError, match="no usable recordings"):
        eg.load_dataset(tmp_path)


def test_design_problems_are_reported(syn_dir, tmp_path):
    assert eg.design_problems(eg.load_dataset(syn_dir)) == []
    one = tmp_path / "one"
    edge_synth.write_synthetic_gestures(one, sessions=1, seconds=12)
    msgs = " | ".join(eg.design_problems(eg.load_dataset(one, trim_s=0.5)))
    assert "only 1 session" in msgs and "'rest' appears in only 1 session" in msgs
    two = tmp_path / "two"
    edge_synth.write_synthetic_gestures(two, sessions=2, seconds=12, labels=("rest", "circle"))
    (two / "s2_circle_synthetic.csv").unlink()
    msgs = " | ".join(eg.design_problems(eg.load_dataset(two, trim_s=0.5)))
    assert "only two sessions" in msgs and "'circle' appears in only 1 session" in msgs and "session 's2' has no recording of: circle" in msgs


# ---------------------------------------------------------------- features
def _random_windows(n=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 50.0
    base = np.stack([300 * np.sin(2 * np.pi * f * t) for f in rng.uniform(1, 6, 3)], axis=-1)
    return np.stack([base * rng.uniform(0.5, 1.5) + rng.normal(0, 5, (128, 3)) + np.array([0, 0, 1000.0]) for _ in range(n)])


def test_motion_features_ignore_gravity_but_raw_features_do_not():
    w = _random_windows()
    c = np.random.default_rng(1).normal(0, 600, (len(w), 1, 3))        # a different gravity direction per window
    assert np.allclose(eg.feature_matrix(w + c, 50.0, "motion"), eg.feature_matrix(w, 50.0, "motion"), atol=1e-6)
    assert not np.allclose(eg.feature_matrix(w + c, 50.0, "raw"), eg.feature_matrix(w, 50.0, "raw"))


def test_the_motion_magnitude_features_do_not_depend_on_how_the_board_is_turned():
    w = _random_windows()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
    names = eg.feature_names_for("motion")
    mag = [i for i, n in enumerate(names) if n.startswith("mag_")]
    a, b = eg.feature_matrix(w, 50.0, "motion"), eg.feature_matrix(w @ R.T, 50.0, "motion")
    assert len(mag) == 10 and np.allclose(a[:, mag], b[:, mag], atol=1e-6)
    assert not np.allclose(a, b)                      # the per-axis features do change with orientation


def test_feature_names_match_the_columns_and_motion_drops_the_zero_means():
    w = _random_windows(3)
    for fset in eg.FEATURE_SETS:
        assert eg.feature_matrix(w, 50.0, fset).shape[1] == len(eg.feature_names_for(fset))
    raw, motion = eg.feature_names_for("raw"), eg.feature_names_for("motion")
    assert len(raw) == 40 and len(motion) == 37 and {"x_mean", "y_mean", "z_mean"}.isdisjoint(motion) and "mag_mean" in motion
    with pytest.raises(ValueError):
        eg.feature_matrix(w, 50.0, "nonsense")


def test_feature_matrices_are_row_contiguous_so_a_row_can_be_handed_to_c():
    """Dropping columns leaves numpy data column-major, and a row of that is not a float array a C function can read."""
    w = _random_windows(5)
    for fset in eg.FEATURE_SETS:
        X = eg.feature_matrix(w, 50.0, fset)
        assert X.flags["C_CONTIGUOUS"] and X[2].flags["C_CONTIGUOUS"]


def test_a_perfectly_still_window_has_no_motion():
    X = eg.feature_matrix(np.full((1, 128, 3), 500.0), 50.0, "motion")
    assert np.allclose(X, 0.0)


# ---------------------------------------------------------------- the synthetic generator
def test_synthetic_recordings_are_marked_deterministic_and_plausible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pa = edge_synth.write_synthetic_gestures(a, sessions=2, seconds=12, seed=3)
    pb = edge_synth.write_synthetic_gestures(b, sessions=2, seconds=12, seed=3)
    assert len(pa) == 2 * len(eg.GESTURES) and all(x.read_bytes() == y.read_bytes() for x, y in zip(pa, pb))
    c = edge_synth.write_synthetic_gestures(tmp_path / "c", sessions=2, seconds=12, seed=4)
    assert any(x.read_bytes() != y.read_bytes() for x, y in zip(pa, c))
    cap = eg.read_capture(pa[0])
    assert cap.synthetic and cap.session == "s1" and cap.verdict == "OK"
    assert np.all(np.abs(cap.xyz_mg) <= 2000) and np.allclose(cap.xyz_mg, np.rint(cap.xyz_mg))
    assert cap.t_s[-1] == pytest.approx(12.0, abs=0.1)
    with pytest.raises(ValueError, match="no synthetic model"):
        edge_synth.write_synthetic_gestures(tmp_path / "d", sessions=1, seconds=5, labels=("wave",))


# ---------------------------------------------------------------- evaluation
def test_session_splits_hold_out_exactly_one_whole_session_each_time():
    sessions = np.array(["s1"] * 5 + ["s2"] * 4 + ["s3"] * 6)
    folds = list(ev.session_splits(sessions))
    assert [h for h, _, _ in folds] == ["s1", "s2", "s3"]
    for held, tr, te in folds:
        assert set(sessions[te]) == {held} and held not in set(sessions[tr])
        assert len(tr) + len(te) == len(sessions) and not set(tr) & set(te)
        ev.assert_session_disjoint(sessions, tr, te)


def test_a_session_on_both_sides_of_a_split_is_caught():
    sessions = np.array(["s1", "s1", "s2", "s2"])
    with pytest.raises(AssertionError, match="s1"):
        ev.assert_session_disjoint(sessions, np.array([0, 2]), np.array([1, 3]))
    X, y = np.random.default_rng(0).normal(size=(4, 3)), np.array(["a", "b", "a", "b"])
    with pytest.raises(AssertionError):                              # and the fold runner refuses such folds
        ev._run_folds(X, y, [("bad", np.array([0, 2]), np.array([1, 3]))], ["a", "b"], 0, groups=sessions)


def test_the_report_has_the_audit_baselines_and_per_session_accuracy(syn_report, syn_dir):
    r = syn_report
    assert r["data"]["synthetic"] and r["data"]["sessions"] == ["s1", "s2", "s3"] and r["data"]["labels"] == list(eg.GESTURES)
    assert r["data"]["windows"] == sum(r["data"]["windows_per_label"].values())
    assert r["baselines"]["chance_accuracy"] == pytest.approx(0.2) and r["baselines"]["majority_accuracy_leave_one_session_out"] <= 0.3
    los = r["protocols"]["leave_one_session_out"]
    assert [f["held_out"] for f in los["folds"]] == ["s1", "s2", "s3"]
    assert all(f["held_out"] not in f["train_sessions"] and f["n_test"] > 0 for f in los["folds"])
    for fset in eg.FEATURE_SETS:
        for model in ("tree", "forest"):
            s = los["results"][fset][model]
            assert len(s["fold_accuracy"]) == 3 and 0 <= s["macro_f1"] <= 1 and np.asarray(s["confusion_matrix"]).sum() == r["data"]["windows"]
    assert (Path(syn_dir) / "s1_rest_synthetic.csv").exists()


def test_results_are_written_to_disk_and_a_one_session_dataset_is_refused(tmp_path):
    d = tmp_path / "d"
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=12)
    out = tmp_path / "res"
    ev.run_gestures(d, out_dir=out, trim_s=0.5, log=lambda *_: None)
    assert json.loads((out / "gestures.json").read_text())["data"]["synthetic"] is True
    assert (out / "gestures.md").read_text().startswith("SYNTHETIC DATA, NOT REAL RECORDINGS")
    one = tmp_path / "one"
    edge_synth.write_synthetic_gestures(one, sessions=1, seconds=12)
    with pytest.raises(ValueError, match="only 1 session"):
        ev.run_gestures(one, out_dir=tmp_path / "r2", trim_s=0.5, log=lambda *_: None)


def _tree_f1(X, y, tr, te):
    from sklearn.metrics import f1_score
    p = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y[te], p, average="macro"), p


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_random_split_overstates_what_a_new_session_would_score(tmp_path, seed):
    """The reason leave-one-session-out exists. When how the board was held varies per recording, windows from one recording
    in both halves of a random split make the task far easier than meeting a session the model has never seen."""
    edge_synth.write_synthetic_gestures(tmp_path, sessions=4, seconds=24, seed=seed, recording_tilt_sd_deg=70)
    ds = eg.load_dataset(tmp_path)
    X = eg.feature_matrix(ds.windows, 50.0, "raw")
    tr, te = train_test_split(np.arange(len(ds.y)), test_size=0.3, stratify=ds.y, random_state=0)
    leaky, _ = _tree_f1(X, ds.y, tr, te)
    from sklearn.metrics import f1_score
    pred = np.empty(len(ds.y), dtype=object)
    for _, a, b in ev.session_splits(ds.session):
        pred[b] = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X[a], ds.y[a]).predict(X[b])
    honest = f1_score(ds.y, pred.astype(str), average="macro")
    assert leaky - honest >= 0.05, (leaky, honest)


def test_labels_keep_the_recommended_order_and_extras_follow():
    assert ev.ordered_labels(np.array(["circle", "wave", "rest", "tilt", "zig"])) == ["rest", "circle", "tilt", "wave", "zig"]


# ---------------------------------------------------------------- README block and figures
def _real_looking_report(syn_report):
    """The synthetic report relabelled as real, ONLY to test the text generation (the numbers mean nothing)."""
    r = json.loads(json.dumps(syn_report))
    r["data"]["synthetic"] = False
    r["data"]["recorded_rate_hz"] = {"min": 51.1, "mean": 51.2, "max": 51.3}
    return r


def test_the_readme_block_refuses_synthetic_results(syn_report):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        rp.render_block(syn_report)


def test_the_readme_block_is_generated_from_the_report_with_no_typed_in_numbers(syn_report):
    r = _real_looking_report(syn_report)
    block = rp.render_block(r)
    assert block.startswith(rp.START) and block.endswith(rp.END)
    for needle in ("s1, s2, s3", "one person and one board", "leave one session out".capitalize(), "Random split of windows (leaky)"):
        assert needle in block
    allowed = set()

    def walk(o):
        if isinstance(o, float):
            allowed.add(f"{o:.3f}")
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(r)
    _, _, best_scores = ev.best_honest(r)             # two numbers the block derives (a lead and a spread) rather than copies
    others = [sc["macro_f1"] for ms in r["protocols"]["leave_one_session_out"]["results"].values() for sc in ms.values() if sc is not best_scores]
    allowed |= {f"{best_scores['macro_f1'] - max(others):.3f}", f"{max(best_scores['fold_accuracy']) - min(best_scores['fold_accuracy']):.3f}"}
    shown = set(re.findall(r"\b\d\.\d{3}\b", block))
    assert shown and shown <= allowed, shown - allowed
    fset, model, best = ev.best_honest(r)
    assert f"{model} on {fset} features" in block and f"{best['macro_f1']:.3f}" in block
    assert "51.1 to 51.3 Hz" in block


def _with_los(report, macro_f1=None, cm=None):
    """Copy of a report with every leave-one-session-out macro-F1 (and optionally confusion matrix) replaced."""
    r = json.loads(json.dumps(report))
    for ms in r["protocols"]["leave_one_session_out"]["results"].values():
        for sc in ms.values():
            if macro_f1 is not None:
                sc["macro_f1"] = macro_f1
            if cm is not None:
                sc["confusion_matrix"] = cm
    return r


def test_the_readme_wording_is_honest_about_ties_perfect_scores_and_the_weakest_gesture(syn_report):
    real = _real_looking_report(syn_report)
    perfect = rp.render_block(_with_los(real, macro_f1=1.0, cm=np.diag([10] * 5).tolist()))
    assert "close to perfect" in perfect and "easy to tell apart" in perfect          # caution on near-perfect scores
    assert "classified every window of the held-out sessions correctly" in perfect and "weakest gesture" not in perfect
    assert "(3 other settings scored the same)" in perfect                               # 2 feature sets x 2 models, all tied
    cm = [[10, 0, 0, 0, 0], [0, 10, 0, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 4, 6], [0, 0, 0, 0, 10]]
    mixed = rp.render_block(_with_los(real, macro_f1=0.7, cm=cm))
    assert "weakest gesture is circle (recall 0.400)" in mixed and "close to perfect" not in mixed
    unique = _with_los(real, macro_f1=0.5, cm=cm)
    top = ev.best_honest(unique)
    unique["protocols"]["leave_one_session_out"]["results"][top[0]][top[1]]["macro_f1"] = 0.6
    assert "scored the same" not in rp.render_block(unique)                              # a unique best: no tie remark


def test_the_recorded_rate_is_described_as_a_single_value_or_a_range(syn_report):
    r = _real_looking_report(syn_report)
    r["data"]["recorded_rate_hz"] = {"min": 51.22, "mean": 51.22, "max": 51.22}
    assert "ran at about 51.2 Hz" in rp.render_block(r)
    r["data"]["recorded_rate_hz"] = {"min": 50.9, "mean": 51.2, "max": 51.6}
    assert "ran at 50.9 to 51.6 Hz" in rp.render_block(r)
    r["data"]["recorded_rate_hz"] = None
    assert "accelerometer itself ran" not in rp.render_block(r)


def test_the_placeholder_block_makes_no_claim_about_results():
    b = rp.placeholder_block()
    assert "synthetic data only" in b and "No real gesture recordings" in b and not re.search(r"\d\.\d{3}", b)


README = """# tinyml-edge-lab

## What is here
| File | What it does |
|---|---|
| `src/tinyml_lab/features.py` | Window features |
| `src/tinyml_lab/cwru*.py` | CWRU bearing pipeline |

## Quick start
```
pip install -r requirements.txt
python scripts/run_cwru.py   # CWRU
```

<!-- cwru:start -->
CWRU results
<!-- cwru:end -->

## Roadmap
1. Things.
"""


def test_update_readme_adds_the_block_row_and_command_once_and_replaces_later(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(README)
    rp.update_readme(p, rp.placeholder_block())
    s = p.read_text()
    assert s.index(rp.START) < s.index("## Roadmap") and s.count(rp.START) == 1 and "CWRU results" in s
    assert s.count("edge_gesture*.py") == 1 and s.count("scripts/run_gestures.py") == 1
    table = s[s.index("## What is here"):s.index("## Quick start")]
    assert table.rstrip().splitlines()[-1].startswith("| `src/tinyml_lab/edge_gesture*.py`")
    quick = s[s.index("## Quick start"):s.index("<!-- cwru:start")]
    assert quick.count("```") == 2 and quick.index("run_gestures.py") < quick.rindex("```")
    rp.update_readme(p, "<!-- gestures:start -->\nNEW\n<!-- gestures:end -->")
    s2 = p.read_text()
    assert "NEW" in s2 and "No real gesture" not in s2 and s2.count(rp.START) == 1
    assert s2.count("edge_gesture*.py") == 1 and s2.count("scripts/run_gestures.py") == 1


def test_update_readme_keeps_windows_line_endings_and_refuses_an_unknown_layout(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(README.replace("\n", "\r\n").encode())
    rp.update_readme(p, rp.placeholder_block())
    raw = p.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")
    q = tmp_path / "other.md"
    q.write_text("# no roadmap here\n")
    with pytest.raises(SystemExit):
        rp.update_readme(q, rp.placeholder_block())


def test_figures_are_drawn(syn_report, syn_dir, tmp_path):
    (tmp_path / "gestures.json").write_text(json.dumps(syn_report))
    names = rp.build_gesture_figures(eg.load_dataset(syn_dir), tmp_path)
    assert names == ["gestures_protocols.png", "gestures_confusion.png", "gestures_examples.png"]
    for n in names:
        assert (tmp_path / "figures" / n).stat().st_size > 5000


# ---------------------------------------------------------------- the C export
@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_the_exported_gesture_tree_in_c_agrees_with_scikit_learn(syn_dir, tmp_path):
    clf, X, y, labels = ev.export_tree(syn_dir, tmp_path, feature_set="motion")
    meta = json.loads((tmp_path / "gesture_model.json").read_text())
    assert meta["classes"] == labels == list(eg.GESTURES) and len(meta["feature_names"]) == X.shape[1]
    assert meta["feature_set"] == "motion" and "no held-out score" in meta["note"]
    so = tmp_path / "t.so"
    subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(so), str(tmp_path / "gesture_tree.c")], check=True)
    lib = ctypes.CDLL(str(so))
    lib.gesture_tree_predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.gesture_tree_predict.restype = ctypes.c_int
    X32 = np.ascontiguousarray(X.astype(np.float32))
    got = np.array([lib.gesture_tree_predict(r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for r in X32])
    assert np.array_equal(got, clf.predict(X32))
    # agreeing with scikit-learn proves nothing if the tree learned the wrong thing: compare with the TRUE gestures too
    ds = eg.load_dataset(syn_dir)
    truth = np.array([labels.index(v) for v in ds.y])
    assert len(truth) == len(got) and np.mean(got == truth) > 0.95


# ---------------------------------------------------------------- the capture tool's new session support
def test_session_is_written_to_the_header_and_the_file_name(tmp_path):
    p = _real_tool_file(tmp_path, session="s3")
    text = p.read_text()
    assert "# session=s3" in text and p.name.startswith("s3_shake_lr_")
    ser = tec.FakeSerial(tec.firmware_lines(800))
    r = ec.save_capture(ec.capture(ser, 60, clock=lambda: ser.t), label="rest", subject="me", session="", note="", port="COM4",
                        baud=115200, out_dir=tmp_path / "nos", now=datetime(2026, 1, 1))
    assert r["path"].name == "rest_20260101_000000.csv" and "# session=\n" in r["path"].read_text()


def test_save_capture_reports_nothing_usable_instead_of_writing_an_empty_file(tmp_path):
    ser = tec.FakeSerial(["garbage line", "more garbage"] * 20)
    r = ec.save_capture(ec.capture(ser, 5, clock=lambda: ser.t), label="rest", subject="me", session="s1", note="", port="COM4",
                        baud=115200, out_dir=tmp_path)
    assert not r["ok"] and "No usable data" in r["reason"] and not list(tmp_path.glob("*.csv"))


# ---------------------------------------------------------------- the session recorder
class StreamFake(tec.FakeSerial):
    """A serial port whose stream starts afresh each time the recorder clears its input buffer."""
    def __init__(self, make_lines):
        super().__init__([])
        self.make_lines = make_lines
        self.resets = 0

    def reset_input_buffer(self):
        self.resets += 1
        self.lines, self.i, self.t = self.make_lines(self.resets), 0, 0.0


def _answers(*a):
    it = iter(a)
    return lambda prompt: next(it)


def test_the_recorder_saves_each_gesture_with_the_session_and_lets_you_redo_one(tmp_path):
    ser = StreamFake(lambda n: tec.firmware_lines(700 if n != 1 else 650))      # take 1 of 'rest' differs from take 2
    log = []
    counter = iter(range(1, 100))
    kept = rs.run_session(ser, session="s1", subject="me", labels=["rest", "circle"], seconds=10, note="palm", port="COM4",
                          baud=115200, out_dir=tmp_path, out=log.append, sleep_fn=lambda s: None, clock=lambda: ser.t,
                          input_fn=_answers("", "r", "", "", "", ""), now_fn=lambda: datetime(2026, 10, 12, 9, 0, 0) + timedelta(minutes=next(counter)))
    assert sorted(kept) == ["circle", "rest"]
    raw = sorted(p.name for p in tmp_path.glob("*.csv") if not p.name.endswith("_50hz.csv"))
    assert len(raw) == 2 and raw[0].startswith("s1_circle_") and raw[1].startswith("s1_rest_")
    assert len(list(tmp_path.glob("*_50hz.csv"))) == 2                     # the first 'rest' take and its copy were deleted
    cap = eg.read_capture(tmp_path / raw[1])
    assert cap.label == "rest" and cap.session == "s1" and cap.note == "palm"
    assert raw[1].endswith("_090200.csv") and not list(tmp_path.glob("*_090100*"))      # the second take stayed, the first is gone
    assert any("Session 's1': 2 of 2 gestures kept" in line for line in log) and ser.resets == 3


def test_the_recorder_can_skip_a_gesture_when_nothing_could_be_saved(tmp_path):
    ser = StreamFake(lambda n: ["garbage"] * 600 if n == 1 else tec.firmware_lines(700))
    kept = rs.run_session(ser, session="s1", subject="me", labels=["rest", "tilt"], seconds=10, note="", port="COM4", baud=115200,
                          out_dir=tmp_path, out=lambda *_: None, sleep_fn=lambda s: None, clock=lambda: ser.t,
                          input_fn=_answers("", "s", "", ""))
    assert list(kept) == ["tilt"]


def test_the_recorder_and_the_loader_agree_on_the_gesture_list():
    assert tuple(g for g, _ in rs.GESTURES) == eg.GESTURES
    assert all(text.strip() for _, text in rs.GESTURES)


def test_what_the_recorder_writes_loads_as_a_dataset(tmp_path):
    ser = StreamFake(lambda n: tec.firmware_lines(900))
    stamps = iter(range(1, 100))
    rs.run_session(ser, session="s1", subject="me", labels=["rest", "circle"], seconds=18, note="", port="COM4", baud=115200,
                   out_dir=tmp_path, out=lambda *_: None, sleep_fn=lambda s: None, clock=lambda: ser.t,
                   input_fn=_answers("", "", "", ""), now_fn=lambda: datetime(2026, 10, 12) + timedelta(minutes=next(stamps)))
    ds = eg.load_dataset(tmp_path, trim_s=1.0)
    assert sorted(set(ds.y.tolist())) == ["circle", "rest"] and set(ds.session.tolist()) == {"s1"} and not ds.synthetic


# ---------------------------------------------------------------- the whole real-data chain
def _gesture_lines(label, n, seed):
    """Firmware-style lines (settings, header, samples at exactly 50 Hz, a statistics line) carrying a simulated gesture."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 50.0
    motion = edge_synth.gesture_motion(label, t, rng, amp=1.0, fscale=1.0)
    a = np.clip(np.rint(motion + np.array([0, 0, 1042.0]) + rng.normal(0, 4, (n, 3))), -2000, 2000).astype(int)
    lines = ["# odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k", "t_ticks,x_mg,y_mg,z_mg"]
    for i, (x, y, z) in enumerate(a):
        lines.append(f"{1000 + round(i * 32768 / 50)},{x},{y},{z}")
        if i == 499:
            lines.append("# stats samples=500 overruns=0 i2c_err=0 rate_hz=50.000 tick_hz=32768 timer=xtal32k")
    return lines


def test_recorder_to_evaluation_to_readme_on_non_synthetic_files(tmp_path):
    """Record 3 sessions x 5 gestures with the real recorder from simulated streams, evaluate them, and publish the README block."""
    data = tmp_path / "edge_gestures"
    stamps = iter(range(1, 1000))
    for si in (1, 2, 3):
        ser = StreamFake(lambda k, si=si: _gesture_lines(list(eg.GESTURES)[(k - 1) % 5], 1250, 100 * si + k))
        rs.run_session(ser, session=f"s{si}", subject="me", labels=list(eg.GESTURES), seconds=25, note="palm", port="COM4",
                       baud=115200, out_dir=data, out=lambda *_: None, sleep_fn=lambda s: None, clock=lambda: ser.t,
                       input_fn=_answers(*([""] * 10)), now_fn=lambda: datetime(2026, 10, 12) + timedelta(minutes=next(stamps)))
    # what Windows Python writes: Windows line endings
    for f in data.glob("*.csv"):
        f.write_bytes(f.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    out = tmp_path / "results"
    report = ev.run_gestures(data, out_dir=out, log=lambda *_: None)
    assert report["data"]["synthetic"] is False and report["data"]["sessions"] == ["s1", "s2", "s3"]
    assert report["data"]["files"] == 15 and report["data"]["design_problems"] == [] and report["data"]["skipped"] == []
    assert not (out / "gestures.md").read_text().startswith("SYNTHETIC")
    assert report["protocols"]["leave_one_session_out"]["results"]["motion"]["forest"]["macro_f1"] > 0.8    # the gestures are separable
    rp.build_gesture_figures(eg.load_dataset(data), out)
    readme = tmp_path / "README.md"
    readme.write_text(README.replace("\n", "\r\n"))
    rp.update_readme(readme, rp.render_block(report))
    text = readme.read_bytes().decode()
    assert "15 recordings of 5 gestures" in text and "s1, s2, s3" in text and "SYNTHETIC" not in text and "No real gesture" not in text
    assert "![Protocols](results/figures/gestures_protocols.png)" in text and text.count("\r\n") == text.count("\n")



def _ranking_report(syn_report, lead, folds=(0.95, 0.80)):
    """A real-looking report in which the best leave-one-session-out setting leads every other by exactly `lead`."""
    r = _real_looking_report(syn_report)
    res = r["protocols"]["leave_one_session_out"]["results"]
    for ms in res.values():
        for sc in ms.values():
            sc["macro_f1"], sc["fold_accuracy"], sc["confusion_matrix"] = 0.60, list(folds), np.diag([10] * 5).tolist()
    res["raw"]["forest"]["macro_f1"] = 0.60 + lead
    return r


def test_the_readme_warns_when_the_best_setting_leads_by_less_than_the_spread_between_sessions(syn_report):
    block = rp.render_block(_ranking_report(syn_report, lead=0.03))            # the spread between sessions is 0.150
    assert "leads the runner-up by 0.030" in block and "less than the 0.150 spread" in block and "should not be over-read" in block


def test_the_readme_does_not_warn_when_the_lead_is_larger_than_the_spread(syn_report):
    assert "over-read" not in rp.render_block(_ranking_report(syn_report, lead=0.20))


def test_the_readme_does_not_talk_about_a_lead_when_settings_are_tied(syn_report):
    block = rp.render_block(_ranking_report(syn_report, lead=0.0))
    assert "over-read" not in block and "leads the runner-up" not in block and "scored the same" in block


def test_the_lead_is_measured_against_the_runner_up_not_the_weakest_setting(syn_report):
    r = _ranking_report(syn_report, lead=0.03)
    res = r["protocols"]["leave_one_session_out"]["results"]
    for fset in res:
        res[fset]["tree"]["macro_f1"] = 0.20                                    # two very weak settings must not inflate the lead
    assert "leads the runner-up by 0.030" in rp.render_block(r)
