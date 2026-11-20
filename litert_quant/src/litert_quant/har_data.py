"""UCI HAR data for model 2: raw total-acceleration windows and one fixed split.

AI-assisted: drafted with Claude, checked by the author.
The split is UCI's own volunteer-disjoint train/test split, the same one the
decision tree baseline in results/har_baseline.json used (7,352 train and
2,947 test windows), so the CNN and the tree are scored on the same windows.
Windows are (128, 3) in units of g at 50 Hz. Labels are 0..5.
"""
import hashlib
from pathlib import Path

import numpy as np

WINDOW = 128
CHANNELS = 3
LABELS = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"


def find_root(path=DEFAULT_ROOT):
    """Return the folder that holds train/ and test/ (the parent folder is accepted too)."""
    path = Path(path)
    candidates = [path, path / "UCI HAR Dataset"] + sorted(path.glob("*/UCI HAR Dataset"))
    for cand in candidates:
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"No UCI HAR train/ and test/ folders found under {path}")


def load_split(root, split):
    """Return x (N, 128, 3) float32, y (N,) int64 in 0..5, subjects (N,)."""
    base = find_root(root) / split
    sig = base / "Inertial Signals"
    chans = [np.loadtxt(sig / f"total_acc_{axis}_{split}.txt") for axis in "xyz"]
    x = np.stack(chans, axis=-1).astype(np.float32)
    y = np.atleast_1d(np.loadtxt(base / f"y_{split}.txt", dtype=int)).astype(np.int64) - 1
    subjects = np.atleast_1d(np.loadtxt(base / f"subject_{split}.txt", dtype=int))
    if not (len(x) == len(y) == len(subjects)):
        raise ValueError("row counts of signals, labels and subjects do not match")
    return x, y, subjects


def load_har(root=DEFAULT_ROOT):
    """Return x_train, x_test, y_train, y_test, s_train, s_test; fail if a volunteer is in both splits."""
    x_train, y_train, s_train = load_split(root, "train")
    x_test, y_test, s_test = load_split(root, "test")
    overlap = set(np.unique(s_train).tolist()) & set(np.unique(s_test).tolist())
    if overlap:
        raise AssertionError(f"subjects in both train and test: {sorted(overlap)}")
    return x_train, x_test, y_train, y_test, s_train, s_test


def fingerprint(x, y):
    """SHA-256 of the arrays, to prove two runs used the same samples."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()
