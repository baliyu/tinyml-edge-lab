import ctypes
import shutil
import subprocess

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from tinyml_lab.export_c import tree_to_c

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")


def _compile(tmp_path, src):
    c = tmp_path / "tree.c"
    so = tmp_path / "tree.so"
    c.write_text(src)
    subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(so), str(c)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.t_predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.t_predict.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize("depth", [1, 3, 8])
def test_c_matches_sklearn_on_random_data(tmp_path, depth):
    rng = np.random.default_rng(depth)
    X = rng.normal(size=(600, 12)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int) + (X[:, 5] > 1.0).astype(int)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    lib = _compile(tmp_path, tree_to_c(clf, "t_predict"))
    Xt = rng.normal(size=(2000, 12)).astype(np.float32)
    expected = clf.predict(Xt)
    got = np.array([lib.t_predict(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in Xt])
    assert np.array_equal(got, expected)


def test_non_contiguous_labels_are_preserved(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 4, 1)   # labels 1 and 4, like UCI's 1..6
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    lib = _compile(tmp_path, tree_to_c(clf, "t_predict"))
    got = {lib.t_predict(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in X}
    assert got <= {1, 4}


def test_values_exactly_on_a_threshold_go_left_like_sklearn(tmp_path):
    """Random floats almost never hit a threshold, so test the boundary explicitly.

    Training values sit on a 0.25 grid, so every midpoint threshold is exactly representable in float32
    and a test value can be placed exactly on it. sklearn sends x <= threshold left; the C code must too.
    """
    rng = np.random.default_rng(3)
    X = (rng.integers(0, 40, size=(500, 5)) / 4.0).astype(np.float32)
    y = ((X[:, 0] > 5) ^ (X[:, 2] > 3)).astype(int) + (X[:, 4] > 7).astype(int)
    clf = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    lib = _compile(tmp_path, tree_to_c(clf, "t_predict"))
    t = clf.tree_
    rows = []
    for node in range(t.node_count):
        if t.children_left[node] == -1:
            continue
        for _ in range(20):
            row = (rng.integers(0, 40, size=5) / 4.0).astype(np.float32)
            row[t.feature[node]] = np.float32(t.threshold[node])
            assert float(row[t.feature[node]]) == float(t.threshold[node])  # exactly on the boundary
            rows.append(row)
    Xb = np.asarray(rows, dtype=np.float32)
    assert len(Xb) > 50
    got = np.array([lib.t_predict(r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for r in Xb])
    assert np.array_equal(got, clf.predict(Xb))
