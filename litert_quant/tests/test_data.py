import numpy as np

from litert_quant.data import fingerprint, load_split


def test_shapes_and_dtypes():
    x_train, x_test, y_train, y_test = load_split()
    assert x_train.shape[1:] == (8, 8, 1)
    assert x_test.shape[1:] == (8, 8, 1)
    assert x_train.dtype == np.float32
    assert len(x_train) + len(x_test) == 1797
    assert len(x_test) == 360


def test_value_range():
    x_train, x_test, _, _ = load_split()
    assert min(x_train.min(), x_test.min()) >= 0.0
    assert max(x_train.max(), x_test.max()) <= 1.0


def test_split_is_deterministic():
    a = load_split()
    b = load_split()
    assert fingerprint(a[1], a[3]) == fingerprint(b[1], b[3])
    c = load_split(seed=1)
    assert fingerprint(a[1], a[3]) != fingerprint(c[1], c[3])


def test_all_classes_in_both_splits():
    _, _, y_train, y_test = load_split()
    assert set(y_train.tolist()) == set(range(10))
    assert set(y_test.tolist()) == set(range(10))


def test_test_set_is_balanced():
    _, _, _, y_test = load_split()
    assert np.bincount(y_test).min() >= 30
