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


# ---------------------------------------------------------------- float thresholds, rounded down, are exact
from tinyml_lab.export_c import f32_literal, floor_f32, forest_to_c   # noqa: E402


def _compile_named(tmp_path, src, name):
    c = tmp_path / f"{name}.c"
    so = tmp_path / f"{name}.so"
    c.write_text(src)
    subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(so), str(c)], check=True)
    lib = ctypes.CDLL(str(so))
    getattr(lib, name).argtypes = [ctypes.POINTER(ctypes.c_float)]
    getattr(lib, name).restype = ctypes.c_int
    return getattr(lib, name)


def _predict_all(fn, X):
    return np.array([fn(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in X])


def test_floor_f32_is_the_largest_float32_not_above_the_value():
    rng = np.random.default_rng(0)
    for v in np.concatenate([rng.normal(size=500) * 10.0 ** rng.integers(-6, 6, 500), [0.1, 1e-40, -3.3]]):
        t = floor_f32(float(v))
        assert float(t) <= float(v)
        assert float(np.nextafter(t, np.float32(np.inf))) > float(v)


def test_float_literals_round_trip_exactly():
    rng = np.random.default_rng(1)
    for v in rng.normal(size=300).astype(np.float32):
        lit = f32_literal(v)
        assert lit.endswith("f") and np.float32(float(lit[:-1])) == v
    assert f32_literal(np.float32(3)) == "3.0f"


@pytest.mark.parametrize("depth", [1, 3, 8])
def test_float_threshold_tree_matches_sklearn_on_random_data(tmp_path, depth):
    rng = np.random.default_rng(depth + 10)
    X = rng.normal(size=(600, 12)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int) + (X[:, 5] > 1.0).astype(int)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    fn = _compile_named(tmp_path, tree_to_c(clf, "t_predict", thresholds="float"), "t_predict")
    Xt = rng.normal(size=(2000, 12)).astype(np.float32)
    assert np.array_equal(_predict_all(fn, Xt), clf.predict(Xt))


def test_float_threshold_tree_survives_two_adjacent_floats(tmp_path):
    """The midpoint of two adjacent float32 values is not a float32. Rounding it to nearest can send the upper value left;
    rounding down must not. (Near 1024 adjacent floats are 1.2e-4 apart, far enough apart for scikit-learn to split them.)"""
    a = np.nextafter(np.float32(1024.0), np.float32(2048.0))  # odd mantissa
    b = np.nextafter(a, np.float32(2048.0))                   # even mantissa: the exact midpoint ties up to b
    X = np.array([[a, 0.0]] * 50 + [[b, 0.0]] * 50, dtype=np.float32)
    y = np.array([0] * 50 + [1] * 50)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    probe = np.array([[a, 0.0], [b, 0.0]], dtype=np.float32)
    assert list(clf.predict(probe)) == [0, 1]                 # scikit-learn does separate them
    assert float(np.float32(float(clf.tree_.threshold[0]))) == float(b)   # and the naive float rounding WOULD send b left
    fn = _compile_named(tmp_path, tree_to_c(clf, "t_predict", thresholds="float"), "t_predict")
    assert list(_predict_all(fn, probe)) == [0, 1]


def test_float_threshold_tree_matches_sklearn_exactly_on_every_boundary(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 5)).astype(np.float32)
    y = ((X[:, 0] > 0.1) ^ (X[:, 2] > -0.3)).astype(int) + (X[:, 4] > 0.7).astype(int)
    clf = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    fn = _compile_named(tmp_path, tree_to_c(clf, "t_predict", thresholds="float"), "t_predict")
    t = clf.tree_
    rows = []
    for node in range(t.node_count):
        if t.children_left[node] == -1:
            continue
        f = int(t.feature[node]); thr = float(t.threshold[node]); lo = floor_f32(thr)
        for v in (lo, np.nextafter(lo, np.float32(-np.inf)), np.nextafter(lo, np.float32(np.inf))):
            row = rng.normal(size=5).astype(np.float32); row[f] = v; rows.append(row)
    Xb = np.asarray(rows, dtype=np.float32)
    assert len(Xb) > 30
    assert np.array_equal(_predict_all(fn, Xb), clf.predict(Xb))


# ---------------------------------------------------------------- forest
from sklearn.ensemble import RandomForestClassifier   # noqa: E402


def test_forest_with_pure_leaves_matches_sklearn_exactly_including_ties(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)       # grown fully: leaves are pure
    fn = _compile_named(tmp_path, forest_to_c(rf, "f_predict"), "f_predict")
    Xt = rng.normal(size=(3000, 6)).astype(np.float32)
    assert np.array_equal(_predict_all(fn, Xt), rf.predict(Xt))


def test_forest_with_mixed_leaves_agrees_with_sklearn_almost_everywhere(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    y = ((X[:, 0] + X[:, 2] + rng.normal(0, 0.8, 600)) > 0).astype(int) + (X[:, 5] > 1).astype(int)
    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    fn = _compile_named(tmp_path, forest_to_c(rf, "f_predict"), "f_predict")
    Xt = rng.normal(size=(3000, 8)).astype(np.float32)
    agree = float(np.mean(_predict_all(fn, Xt) == rf.predict(Xt)))
    assert agree >= 0.998


def test_forest_keeps_the_original_class_labels(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 5, 2)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    fn = _compile_named(tmp_path, forest_to_c(rf, "f_predict"), "f_predict")
    assert set(_predict_all(fn, X).tolist()) <= {2, 5}
