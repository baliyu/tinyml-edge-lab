import numpy as np
from tinyml_lab.features import gravity_mean, gravity_names, with_magnitude


def test_with_magnitude_adds_a_fourth_channel():
    w = np.zeros((2, 5, 3)); w[..., 0] = 3.0; w[..., 1] = 4.0
    out = with_magnitude(w)
    assert out.shape == (2, 5, 4)
    assert np.allclose(out[..., 3], 5.0)


def test_gravity_mean_recovers_a_constant_offset():
    rng = np.random.default_rng(0)
    body = rng.normal(size=(3, 128, 3))
    gravity = np.array([0.1, -0.9, 0.4])
    got = gravity_mean(body + gravity, body)
    assert got.shape == (3, 3)
    assert np.allclose(got, gravity)
    assert len(gravity_names()) == 3
