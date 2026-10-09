import numpy as np
from tinyml_lab.features import feature_names, window_features

FS, N = 50.0, 128


def test_dominant_frequency_of_a_sine():
    t = np.arange(N) / FS
    w = np.sin(2 * np.pi * 5.0 * t)[:, None]
    f = window_features(w, FS)
    dom = f[feature_names(["c"]).index("c_dom_freq")]
    assert abs(dom - 5.0) <= FS / N  # within one FFT bin


def test_constant_signal_has_zero_spread_and_zero_bands():
    f = window_features(np.full((N, 1), 2.0), FS)
    names = feature_names(["c"])
    assert f[names.index("c_std")] == 0.0
    assert f[names.index("c_mean")] == 2.0
    assert all(f[names.index(f"c_band{b}")] == 0.0 for b in range(4))


def test_band_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    f = window_features(rng.normal(size=(N, 1)), FS)
    names = feature_names(["c"])
    assert abs(sum(f[names.index(f"c_band{b}")] for b in range(4)) - 1.0) < 1e-9


def test_feature_count_matches_names():
    f = window_features(np.zeros((N, 3)), FS, n_bands=5)
    assert len(f) == len(feature_names(["x", "y", "z"], n_bands=5))
