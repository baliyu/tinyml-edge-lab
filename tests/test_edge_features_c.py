"""Tests for the C feature code (firmware/sparkfun_edge/gesture_features). All windows are SYNTHETIC or constructed
(NOT real data): they test that the C code computes what the Python code computes, not how it behaves on real motion."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tinyml_lab import edge_features_c as efc
from tinyml_lab import edge_gestures as eg
from tinyml_lab import edge_synth

pytestmark = pytest.mark.skipif(efc.find_gcc() is None, reason="no C compiler")

C_FILE = efc.C_DIR / "edge_features.c"
ROOT = Path(__file__).resolve().parents[1]
ARM_FLAGS = ["-std=c11", "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16", "-Os",
             "-ffunction-sections", "-fdata-sections", "-Wall", "-Wextra", "-Werror", "-pedantic"]


@pytest.fixture(scope="module")
def c64(tmp_path_factory):
    return efc.CFeatures("double", tmp_path_factory.mktemp("c64"))


@pytest.fixture(scope="module")
def c32(tmp_path_factory):
    return efc.CFeatures("float", tmp_path_factory.mktemp("c32"))


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=20, seed=0)
    return eg.load_dataset(d)


def _special_windows():
    """Constructed windows, each with a known character, none with floating-point dust in a 'silent' channel."""
    n = np.arange(128)
    w = {}
    w["zeros"] = np.zeros((128, 3))
    w["gravity only"] = np.tile([0.0, 0.0, 1000.0], (128, 1))
    w["constant, mixed"] = np.tile([123.0, -456.0, 789.0], (128, 1))
    s = np.zeros((128, 3)); s[:, 0] = 500 * np.sin(2 * np.pi * 8 * n / 128); s[:, 2] = 1000
    w["x sine, 8 cycles"] = s
    s = np.zeros((128, 3)); s[:, 1] = 1000 * (-1.0) ** n
    w["y at Nyquist"] = s
    s = np.zeros((128, 3)); s[10, 0] = 2000
    w["x impulse"] = s
    w["square wave, saturated"] = np.tile((2000.0 * np.where((n // 8) % 2 == 0, 1, -1))[:, None], (1, 3))
    w["ramp"] = np.stack([n * 10.0, -n * 4.0, 1000 + n * 0.5], axis=1)
    s = np.zeros((128, 3)); s[:, 0] = 350 * np.sin(2 * np.pi * 2 * n / 128); s[:, 1] = 350 * np.cos(2 * np.pi * 2 * n / 128); s[:, 2] = 1042
    w["circle"] = s
    w["tiny noise"] = np.random.default_rng(5).normal(0, 0.3, (128, 3)) + [0, 0, 1000]
    w["clipped noise"] = np.clip(np.random.default_rng(6).normal(0, 1500, (128, 3)), -2000, 2000)
    return w


def _windows(syn_ds):
    """All the windows used for the comparison with Python, and the position of the one with a flat spectrum (the impulse)."""
    rng = np.random.default_rng(1)
    rand = rng.normal(0, 300, (60, 128, 3)) + np.array([0, 0, 1000.0])
    special = _special_windows()
    flat = len(syn_ds.windows) + len(rand) + list(special).index("x impulse")
    return np.concatenate([syn_ds.windows, rand, np.stack(list(special.values()))]), flat


def _err_vs_size(py, c):
    scale = np.maximum(np.abs(py).max(axis=0), 1e-12)
    return np.abs(c - py) / scale


# ---------------------------------------------------------------- the cosine table
def test_the_cosine_table_in_the_c_file_matches_numpy():
    text = C_FILE.read_text()
    start = text.index("COS_TAB[EF_WINDOW] = {") + len("COS_TAB[EF_WINDOW] = {")
    body = text[start:text.index("};", start)]
    vals = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", body)]
    assert len(vals) == 128
    # the values are printed with full round-trip precision, so they equal numpy's; 3e-16 (about 3 ulp) leaves room for
    # a different maths library, and still catches a wrong digit anywhere in the first fifteen
    assert np.allclose(vals, np.cos(2 * np.pi * np.arange(128) / 128), rtol=0, atol=3e-16)


# ---------------------------------------------------------------- agreement with Python
def _channels(W, fset):
    """The four channels the features are computed from, as numpy sees them: (N, 128, 4)."""
    W = np.asarray(W, dtype=np.float64)
    base = W - W.mean(axis=1, keepdims=True) if fset == "motion" else W
    return np.concatenate([base, np.linalg.norm(base, axis=-1, keepdims=True)], axis=-1)


def _dust_channels(W, fset):
    """(N, 4) True where a channel's spectral energy is at rounding-dust level (below 1e-9 of its energy): the C code is
    meant to report no spectrum there, while Python reads the dust as a spectrum of arbitrary shape."""
    ch = _channels(W, fset)
    xc = ch - ch.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    spec[:, 0, :] = 0
    return spec.sum(axis=1) <= 1e-9 * (ch ** 2).sum(axis=1)


def _checked_errors(fset, W, flat, py, c):
    """Errors vs feature size, with the two inherently arbitrary cases handled explicitly instead of hidden:
    - a dust-level channel: the C output must be exactly zero for its dominant frequency and bands; Python's value is arbitrary;
    - the flat spectrum of an impulse: every bin has the same energy, so which bin is 'largest' is rounding noise."""
    names = eg.feature_names_for(fset)
    err = _err_vs_size(py, c)
    dust = _dust_channels(W, fset)
    assert dust.sum() >= 3                                    # the constructed windows really exercise the guard
    for ci, ch in enumerate(("x", "y", "z", "mag")):
        cols = [i for i, n in enumerate(names) if n.startswith(ch + "_") and (n.endswith("dom_freq") or "_band" in n)]
        rows = np.flatnonzero(dust[:, ci])
        assert np.all(c[np.ix_(rows, cols)] == 0)
        err[np.ix_(rows, cols)] = 0.0
    err[flat, [i for i, n in enumerate(names) if n.endswith("dom_freq")]] = 0.0
    return err


@pytest.mark.parametrize("fset", eg.FEATURE_SETS)
def test_built_as_double_it_matches_python_to_machine_precision(c64, syn, fset):
    W, flat = _windows(syn)
    py = eg.feature_matrix(W, eg.FS, fset)
    c = c64.compute(W, fset)
    assert _checked_errors(fset, W, flat, py, c).max() < 1e-12
    dom = [i for i, n in enumerate(eg.feature_names_for(fset)) if n.endswith("dom_freq")]
    assert np.any(np.abs(py[flat, dom] - c[flat, dom]) > 1)   # the impulse really is the window whose dominant bin is arbitrary


@pytest.mark.parametrize("fset", eg.FEATURE_SETS)
def test_built_as_float_it_matches_python_closely(c32, syn, fset):
    W, flat = _windows(syn)
    py = eg.feature_matrix(W, eg.FS, fset)
    c = c32.compute(W, fset)
    assert _checked_errors(fset, W, flat, py, c).max() < 1e-4
    assert efc.compare_features(py, c, fset)["windows"] == len(W)


@pytest.mark.parametrize("fset", eg.FEATURE_SETS)
def test_the_output_has_the_documented_length_and_order(c64, fset):
    names = eg.feature_names_for(fset)
    assert len(names) == efc.N_FEATURES[fset]
    out = c64.compute(_special_windows()["x sine, 8 cycles"][None], fset)[0]
    assert out.shape == (len(names),)
    i = names.index("x_dom_freq")
    assert out[i] == pytest.approx(3.125)         # the feature at the position the Python names give


# ---------------------------------------------------------------- known signals, independent of Python
def test_a_pure_sine_has_the_expected_frequency_energy_and_size(c64):
    out = dict(zip(eg.feature_names_for("motion"), c64.compute(_special_windows()["x sine, 8 cycles"][None], "motion")[0]))
    assert out["x_dom_freq"] == pytest.approx(8 * 50 / 128) and out["x_band0"] == pytest.approx(1.0, abs=1e-9)
    assert out["x_std"] == pytest.approx(500 / np.sqrt(2), rel=1e-9) and out["x_min"] == pytest.approx(-500, rel=1e-3)
    assert out["x_max"] == pytest.approx(500, rel=1e-3)
    for silent in ("y_std", "y_dom_freq", "y_band0", "y_band3", "z_std", "z_dom_freq", "z_band2"):
        assert out[silent] == 0.0                  # no motion on y and z: no energy, no dominant frequency


def test_a_signal_at_the_nyquist_frequency_lands_in_the_top_band(c64):
    out = dict(zip(eg.feature_names_for("motion"), c64.compute(_special_windows()["y at Nyquist"][None], "motion")[0]))
    assert out["y_dom_freq"] == pytest.approx(25.0) and out["y_band3"] == pytest.approx(1.0, abs=1e-9) and out["y_band0"] < 1e-12


def test_the_four_bands_cover_the_spectrum_and_sum_to_one(c64, syn):
    names = eg.feature_names_for("motion")
    out = c64.compute(syn.windows, "motion")
    for ch in ("x", "y", "z", "mag"):
        bands = out[:, [names.index(f"{ch}_band{b}") for b in range(4)]]
        total = bands.sum(axis=1)
        assert np.all((np.abs(total - 1) < 1e-9) | (total == 0))     # a still channel has no energy at all


def test_a_perfectly_still_window_has_zero_motion_and_a_kept_gravity_in_the_raw_features(c64):
    still = _special_windows()["constant, mixed"][None]
    assert np.all(c64.compute(still, "motion") == 0.0)
    raw = dict(zip(eg.feature_names_for("raw"), c64.compute(still, "raw")[0]))
    assert raw["x_mean"] == 123 and raw["y_mean"] == -456 and raw["z_mean"] == 789 and raw["x_std"] == 0 and raw["x_dom_freq"] == 0
    assert raw["mag_mean"] == pytest.approx(np.linalg.norm([123, -456, 789]))


def test_motion_features_ignore_gravity_in_c_too(c64):
    w = _special_windows()["circle"]
    shifted = w + np.array([300.0, -250.0, 400.0])
    a, b = c64.compute(w[None], "motion"), c64.compute(shifted[None], "motion")
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    assert not np.allclose(c64.compute(w[None], "raw"), c64.compute(shifted[None], "raw"))


# ---------------------------------------------------------------- channels with no variation have no spectrum
@pytest.mark.parametrize("name", ["constant, mixed", "square wave, saturated", "circle"])
def test_a_channel_with_no_variation_has_no_spectrum_in_either_build(c32, c64, name):
    """A perfectly constant magnitude (a circle, a saturated square wave, a stuck sensor) leaves only rounding dust. The C code
    reports no dominant frequency and no band energy for it, instead of letting summation order decide."""
    w = _special_windows()[name][None]
    names = eg.feature_names_for("raw")
    for lib in (c32, c64):
        out = dict(zip(names, lib.compute(w, "raw")[0]))
        assert out["mag_dom_freq"] == 0 and all(out[f"mag_band{b}"] == 0 for b in range(4))


def test_a_stuck_sensor_window_gives_the_same_features_whatever_the_gravity():
    """Identical readings for a whole window: in the motion set every feature is zero, in float and in double."""
    for real in ("float", "double"):
        lib = efc.CFeatures(real)
        for g in ([0.0, 0.0, 1000.0], [50.0, -80.0, 1042.0], [-1.0, 7.0, 1013.0]):
            out = lib.compute(np.tile(g, (128, 1))[None], "motion")[0]
            assert np.all(out[:27] == 0) and np.all(out[28:] == 0)       # all but mag_mean, which is the mean of a zero channel
            assert out[27] == 0


def test_the_dust_threshold_does_not_swallow_real_sensor_noise(c32, c64):
    """The quietest realistic input: whole milli-g readings (the sensor's resolution) of gravity on z plus noise of one count.
    Real noise is several counts, so this is a hard case for the threshold."""
    rng = np.random.default_rng(3)
    w = np.rint(np.array([0, 0, 1000.0]) + rng.normal(0, 1.0, (200, 128, 3)))
    names = eg.feature_names_for("raw")
    for fset in eg.FEATURE_SETS:
        py = eg.feature_matrix(w, eg.FS, fset)
        for lib, tol in ((c64, 1e-12), (c32, 5e-4)):
            c = lib.compute(w, fset)
            assert _err_vs_size(py, c).max() < tol
    c = c64.compute(w, "raw")
    z0 = names.index("z_band0")
    assert np.all(c[:, z0:z0 + 4].sum(axis=1) > 0.999)          # the noise on z was read as a spectrum, not discarded


# ---------------------------------------------------------------- does float rounding change a model's answer?
def test_float_features_do_not_change_a_trained_models_answers(c32, syn):
    for fset in eg.FEATURE_SETS:
        py = eg.feature_matrix(syn.windows, eg.FS, fset)
        agree = efc.prediction_agreement(py, c32.compute(syn.windows, fset), syn.y, syn.session)
        for model, r in agree.items():
            assert r["seen"]["different"] <= max(1, 0.005 * r["seen"]["windows"]), (fset, model, r)
            u = r["unseen sessions"]
            assert u["windows"] == len(syn.y) and u["different"] <= max(1, 0.005 * u["windows"]), (fset, model, r)


def test_prediction_agreement_notices_a_real_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); y = (X[:, 0] > 0).astype(int); s = np.array(["a"] * 100 + ["b"] * 100)
    X_bad = X.copy(); X_bad[:, 0] = -X_bad[:, 0]                    # a feature that flips sign: every answer changes
    r = efc.prediction_agreement(X, X_bad, y, s)["tree (depth 6)"]
    assert r["seen"]["different"] > 150 and r["unseen sessions"]["different"] > 150
    same = efc.prediction_agreement(X, X, y, s)["tree (depth 6)"]
    assert same["seen"]["different"] == 0 and same["unseen sessions"]["different"] == 0


# ---------------------------------------------------------------- the board build
@pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="no ARM compiler")
def test_the_board_build_is_small_uses_little_stack_and_needs_no_maths_library(tmp_path):
    obj = tmp_path / "ef.o"
    subprocess.run(["arm-none-eabi-gcc", *ARM_FLAGS, "-fno-math-errno", "-fstack-usage", "-c", str(C_FILE), "-o", str(obj)],
                   check=True, capture_output=True, text=True, cwd=tmp_path)
    sizes = subprocess.run(["arm-none-eabi-size", str(obj)], check=True, capture_output=True, text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sizes[0]), int(sizes[1]), int(sizes[2])
    assert text < 2048 and data == 0 and bss == 0               # measured: 1504 bytes, all of it code and the cosine table
    undefined = subprocess.run(["arm-none-eabi-nm", "-u", str(obj)], check=True, capture_output=True, text=True).stdout.split()
    assert undefined == []                                      # the square root is a single FPU instruction
    stack = {}
    for line in (tmp_path / "ef.su").read_text().splitlines():
        parts = line.split("\t")
        stack[parts[0].split(":")[-1]] = int(parts[1])
    peak = max(stack["ef_motion_features"], stack["ef_raw_features"]) + stack["channel_features"]
    assert peak < 4096                                          # measured: about 3.5 KB


# ---------------------------------------------------------------- the helper module and the script
def test_the_helper_rejects_bad_input(c32):
    with pytest.raises(ValueError, match="real must be"):
        efc.CFeatures("half")
    with pytest.raises(ValueError, match="shape"):
        c32.compute(np.zeros((3, 100, 3)), "motion")
    with pytest.raises(ValueError, match="shape"):
        c32.compute(np.zeros((128, 3)), "motion")
    assert c32.compute(np.zeros((0, 128, 3)), "raw").shape == (0, 40)


def test_the_check_script_runs_and_writes_its_report(tmp_path):
    d = tmp_path / "syn"
    edge_synth.write_synthetic_gestures(d, sessions=3, seconds=20, seed=2)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_edge_features.py"), "--data", str(d), "--out", str(tmp_path / "res")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SYNTHETIC" in r.stdout and "C as double vs Python" in r.stdout and "never saw" in r.stdout
    rep = json.loads((tmp_path / "res" / "edge_features_check.json").read_text())
    assert rep["data"]["synthetic"] is True
    for fset in eg.FEATURE_SETS:
        assert rep["feature_sets"][fset]["double_vs_python"]["max_error_vs_feature_size"] < 1e-12
        assert rep["feature_sets"][fset]["float_vs_python"]["max_error_vs_feature_size"] < 1e-4
    missing = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_edge_features.py"), "--data", str(tmp_path / "nope")],
                             capture_output=True, text=True)
    assert missing.returncode == 1 and "Cannot run" in missing.stdout
