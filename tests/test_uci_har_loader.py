import numpy as np
import pytest
from tinyml_lab import uci_har


def test_load_split_shapes(fake_uci_root):
    X, y, s = uci_har.load_split(fake_uci_root, "train")
    assert X.shape == (3 * 6 * 6, 128, 3)
    assert set(y) == set(range(1, 7))
    assert set(s) == {1, 2, 3}


def test_find_root_accepts_parent_and_dataset_folder(fake_uci_root):
    a = uci_har.find_root(fake_uci_root)
    b = uci_har.find_root(fake_uci_root / "UCI HAR Dataset")
    assert a == b


def test_find_root_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        uci_har.find_root(tmp_path)


def test_subject_overlap_is_rejected():
    uci_har.assert_subject_disjoint(np.array([1, 2]), np.array([3]))
    with pytest.raises(AssertionError):
        uci_har.assert_subject_disjoint(np.array([1, 2]), np.array([2, 3]))
