"""Tests for the HAR loader, on a tiny synthetic dataset in the UCI layout (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
import pytest

from litert_quant import har_data
from litert_quant.har_data import fingerprint, load_har, load_split


def _write(base, split, subjects, labels, rng):
    sig = base / split / "Inertial Signals"
    sig.mkdir(parents=True)
    n = len(labels)
    for axis in "xyz":
        np.savetxt(sig / f"total_acc_{axis}_{split}.txt", rng.normal(size=(n, 128)))
    np.savetxt(base / split / f"y_{split}.txt", labels, fmt="%d")
    np.savetxt(base / split / f"subject_{split}.txt", subjects, fmt="%d")


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "UCI HAR Dataset"
    _write(base, "train", [1, 1, 2, 2], [1, 2, 3, 6], rng)
    _write(base, "test", [3, 3], [4, 5], rng)
    return tmp_path


def test_shapes_and_dtypes(root):
    x_tr, x_te, y_tr, y_te, s_tr, s_te = load_har(root)
    assert x_tr.shape == (4, 128, 3)
    assert x_te.shape == (2, 128, 3)
    assert x_tr.dtype == np.float32
    assert len(s_tr) == 4 and len(s_te) == 2


def test_labels_are_zero_based(root):
    _, _, y_tr, y_te, _, _ = load_har(root)
    assert y_tr.tolist() == [0, 1, 2, 5]
    assert y_te.tolist() == [3, 4]
    assert y_tr.dtype == np.int64


def test_channels_are_on_the_last_axis(root):
    x_tr, _, _ = load_split(root, "train")
    raw_y = np.loadtxt(root / "UCI HAR Dataset" / "train" / "Inertial Signals" / "total_acc_y_train.txt")
    np.testing.assert_allclose(x_tr[0, :, 1], raw_y[0], rtol=1e-6)


def test_overlapping_subjects_raise(tmp_path):
    rng = np.random.default_rng(1)
    bad = tmp_path / "bad" / "UCI HAR Dataset"
    _write(bad, "train", [1, 2, 2], [1, 2, 3], rng)
    _write(bad, "test", [2, 3], [4, 5], rng)
    with pytest.raises(AssertionError):
        load_har(tmp_path / "bad")


def test_row_count_mismatch_raises(root):
    subject_file = root / "UCI HAR Dataset" / "train" / "subject_train.txt"
    np.savetxt(subject_file, [1, 1, 2], fmt="%d")
    with pytest.raises(ValueError):
        load_split(root, "train")


def test_fingerprint_is_stable_and_sensitive():
    x = np.zeros((2, 128, 3), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    x2 = x.copy()
    x2[0, 0, 0] += 1.0
    assert fingerprint(x, y) == fingerprint(x.copy(), y.copy())
    assert fingerprint(x, y) != fingerprint(x2, y)
    assert len(fingerprint(x, y)) == 64


def test_default_root_points_into_repo_data_raw():
    parts = har_data.DEFAULT_ROOT.parts
    assert parts[-2:] == ("data", "raw")
    assert (har_data.DEFAULT_ROOT.parents[1] / "litert_quant" / "src").is_dir()
