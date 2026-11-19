import json

import numpy as np

from litert_quant.figures import confusion, make_model1_figure, wilson_interval


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 1e-3
    assert abs(hi - 0.5962) < 1e-3


def test_wilson_interval_stays_inside_zero_one():
    lo, hi = wilson_interval(10, 10)
    assert 0.0 <= lo < hi <= 1.0 + 1e-9
    lo, hi = wilson_interval(0, 10)
    assert -1e-9 <= lo < hi <= 1.0


def test_confusion_counts():
    y = np.array([0, 1, 1, 2])
    p = np.array([0, 1, 2, 2])
    m = confusion(y, p, 3)
    assert m.sum() == 4
    assert np.trace(m) == 3
    assert m[1, 2] == 1


def test_figure_is_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 60)
    np.savez(tmp_path / "p.npz", y_test=y, keras=y, float=y, int8=y)
    info = {"float_bytes": 1000, "int8_bytes": 500, "parameters": 10, "seed": 0}
    (tmp_path / "r.json").write_text(json.dumps(info))
    out = make_model1_figure(tmp_path / "p.npz", tmp_path / "r.json", tmp_path / "f" / "fig.png")
    assert out.exists()
    assert out.stat().st_size > 5000
