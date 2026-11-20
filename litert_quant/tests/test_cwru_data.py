"""Tests for the CWRU loader, on tiny synthetic .mat files (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
import pytest
from scipy.io import savemat

from litert_quant import cwru_data as cd
from litert_quant.cwru_data import (
    Recording, catalogue, check_disjoint, downsample, file_hashes,
    fingerprint, load_de, make_split, windows,
)


def _mat(root, num, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    savemat(str(root / f"{num}.mat"), {f"X{num:03d}_DE_time": x, f"X{num:03d}_FE_time": x * 0 + 9.0})


def test_catalogue_counts():
    recs = catalogue()
    nums = [r.num for r in recs]
    assert len(recs) == 39 and len(set(nums)) == 39
    assert sum(r.kind == "normal" for r in recs) == 4


def test_catalogue_excludes_only_174():
    nums = {r.num for r in catalogue()}
    assert 174 not in nums
    assert {97, 98, 99, 100, 215, 217} <= nums


def test_catalogue_fields_valid():
    for r in catalogue():
        assert r.kind in ("normal", "IR", "B", "OR")
        assert r.load_hp in (0, 1, 2, 3)
        assert (r.size_mil == 0) == (r.kind == "normal")
        if r.kind != "normal":
            assert r.size_mil in (7, 14, 21)


def test_roles_split_normal_by_load():
    recs = catalogue()
    assert {r.num for r in recs if cd.role(r) == "train"} == {97, 98, 99}
    assert {r.num for r in recs if r.kind == "normal" and cd.role(r) == "test"} == {100}
    assert all(cd.role(r) == "test" for r in recs if r.kind != "normal")


def test_load_de_picks_drive_end(tmp_path):
    de = np.random.default_rng(0).normal(size=100)
    _mat(tmp_path, 5, de)
    out = load_de(tmp_path, 5)
    assert out.ndim == 1
    np.testing.assert_allclose(out, de)


def test_load_de_missing_channel_raises(tmp_path):
    savemat(str(tmp_path / "6.mat"), {"X006_FE_time": np.zeros((10, 1))})
    with pytest.raises(KeyError):
        load_de(tmp_path, 6)


def test_load_de_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_de(tmp_path, 7)


def test_downsample_length_and_band():
    t = np.arange(48000) / 48000.0
    low = downsample(np.sin(2 * np.pi * 1000 * t))
    high = downsample(np.sin(2 * np.pi * 9000 * t))
    assert len(low) == 12000 and low.dtype == np.float32
    mid = slice(1000, -1000)
    assert abs(float(np.sqrt(np.mean(low[mid] ** 2))) - np.sqrt(0.5)) < 0.03
    assert float(np.sqrt(np.mean(high[mid] ** 2))) < 0.05


def test_windows_shape_and_stride():
    x = np.arange(1000, dtype=np.float32)
    w = windows(x, 128, 128)
    assert w.shape == (7, 128) and w.dtype == np.float32
    assert w[1, 0] == 128
    assert windows(x, 128, 64).shape == (14, 128)


def test_windows_short_raises():
    with pytest.raises(ValueError):
        windows(np.zeros(50), 128, 128)


def test_make_split_shapes_and_disjoint(tmp_path):
    rng = np.random.default_rng(1)
    recs = [
        Recording(1, "normal", 0, 0), Recording(2, "normal", 0, 1), Recording(3, "normal", 0, 3),
        Recording(4, "IR", 7, 0), Recording(5, "OR", 14, 2),
    ]
    for r in recs:
        _mat(tmp_path, r.num, rng.normal(size=4 * 512))
    s = make_split(tmp_path, recs)
    assert s.x_train.shape == (8, 128) and s.x_train.dtype == np.float32
    assert set(s.rec_train.tolist()) == {1, 2}
    assert set(s.rec_test.tolist()) == {3, 4, 5}
    assert s.x_test.shape == (12, 128)
    assert int(s.y_test.sum()) == 8
    assert set(s.kind_test.tolist()) == {"normal", "IR", "OR"}
    again = make_split(tmp_path, recs)
    assert fingerprint(s.x_test, s.y_test) == fingerprint(again.x_test, again.y_test)


def test_check_disjoint_raises():
    check_disjoint([1, 2], [3])
    with pytest.raises(AssertionError):
        check_disjoint([1, 2], [2, 3])


def test_file_hashes(tmp_path):
    recs = [Recording(1, "normal", 0, 0), Recording(2, "IR", 7, 0)]
    _mat(tmp_path, 1, np.zeros(10))
    _mat(tmp_path, 2, np.ones(10))
    h1 = file_hashes(tmp_path, recs)
    assert set(h1) == {1, 2} and all(len(v) == 64 for v in h1.values())
    _mat(tmp_path, 2, np.ones(10) * 2)
    assert file_hashes(tmp_path, recs)[2] != h1[2]
    assert file_hashes(tmp_path, recs)[1] == h1[1]


def test_fingerprint_is_stable_and_sensitive():
    x = np.zeros((2, 128), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    x2 = x.copy()
    x2[0, 0] += 1.0
    assert fingerprint(x, y) == fingerprint(x.copy(), y.copy())
    assert fingerprint(x, y) != fingerprint(x2, y)
    assert len(fingerprint(x, y)) == 64


def test_default_root_points_into_repo_data_raw_cwru():
    assert cd.DEFAULT_ROOT.parts[-3:] == ("data", "raw", "cwru")
    assert (cd.DEFAULT_ROOT.parents[2] / "litert_quant" / "src").is_dir()


def test_load_de_uses_own_channel_when_extra_present(tmp_path):
    own = np.arange(20, dtype=np.float64)
    savemat(str(tmp_path / "9.mat"), {"X217_DE_time": np.zeros((50, 1)), "X009_DE_time": own.reshape(-1, 1)})
    np.testing.assert_allclose(load_de(tmp_path, 9), own)


def test_normal_recordings_cover_all_four_loads():
    assert sorted(r.load_hp for r in catalogue() if r.kind == "normal") == [0, 1, 2, 3]
